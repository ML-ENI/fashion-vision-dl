"""Lectura y evaluación del 07 sin fotos, modelos ni entrenamiento.

Las predicciones se guardan con sus productos. Cambiar una tabla o una métrica
no forma parte de la identidad del entrenamiento y no dispara un ajuste nuevo.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.data.data_contract import sha256
from src.training.transfer_cache import identity


def find_saved(root, experiment_identity):
    """Buscar solo ejecuciones completas compatibles y verificar sus archivos."""
    directory = Path(root) / 'results/metrics/07_transfer/runs' / identity(experiment_identity)
    summary_path = directory / 'summary.json'
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if summary.get('status') != 'completed' or summary.get('identity') != experiment_identity:
        return None
    for name, expected in summary['artifacts_sha256'].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f'Falta un archivo del experimento guardado: {name}')
        if sha256(path) != expected:
            raise ValueError(f'El resultado guardado ha cambiado: {name}')
    return summary, directory


def evaluate_saved(saved, validation, names):
    """Recalcular métricas por semilla, sin seleccionar la semilla con mejor F1."""
    summary, directory = saved
    records, per_class, matrices = [], [], {}
    expected = validation[['article_id', 'group_id', 'label_id']].copy()
    expected['label_id'] = expected.label_id.astype(int)
    classes = np.arange(len(names))
    for run in summary['fits']:
        predictions = pd.read_csv(directory / run['predictions'],
                                  dtype={'article_id': 'string', 'group_id': 'string'})
        if predictions.article_id.duplicated().any():
            raise ValueError('Predicciones duplicadas')
        if set(predictions.article_id) != set(expected.article_id):
            raise ValueError('Las predicciones corresponden a otros productos')
        aligned = expected.merge(predictions, on=['article_id', 'group_id'],
                                 validate='one_to_one', suffixes=('_expected', ''))
        if len(aligned) != len(expected) or not aligned.label_id.eq(aligned.label_id_expected).all():
            raise ValueError('Partición, grupos o etiquetas diferentes en las predicciones')
        prediction = aligned.predicted_id.to_numpy()
        if not np.isin(prediction, classes).all():
            raise ValueError('Predicción fuera del repertorio de clases')
        target = aligned.label_id.to_numpy()
        records.append({
            'Semilla': run['seed'],
            'Accuracy': accuracy_score(target, prediction),
            'Macro-F1': f1_score(target, prediction, labels=classes, average='macro', zero_division=0),
            'Época elegida': run['best_epoch'],
            'Segundos de entrenamiento': run['training_seconds'],
            'Parámetros entrenables': run['trainable_parameters'],
            'Pico RSS del proceso (bytes)': run['peak_rss_bytes'],
        })
        report = classification_report(target, prediction, labels=classes, target_names=names,
                                       output_dict=True, zero_division=0)
        for name in names:
            per_class.append({'Semilla': run['seed'], 'Clase': name, **report[name]})
        matrices[run['seed']] = confusion_matrix(target, prediction, labels=classes)
    return pd.DataFrame(records), pd.DataFrame(per_class), matrices


def product_bootstrap(validation, predictions, classes, *, draws=300, seed=42):
    """IC percentil del 95 %: remuestrear grupos completos, no fotos sueltas.

    Es incertidumbre de validation condicionada al modelo ya elegido. No mide
    el efecto de nuevas semillas ni convierte validation en un test independiente.
    """
    if draws < 2:
        raise ValueError('Se necesitan al menos dos remuestreos')
    target = validation.label_id.astype(int).to_numpy()
    prediction = np.asarray(predictions)
    if len(prediction) != len(target):
        raise ValueError('Predicciones sin alinear con validation')
    if not np.isin(target, np.arange(classes)).all() or not np.isin(prediction, np.arange(classes)).all():
        raise ValueError('Etiquetas fuera del repertorio')
    codes, groups = pd.factorize(validation.group_id, sort=True)
    grouped = np.zeros((len(groups), classes, classes), dtype=np.int64)
    np.add.at(grouped, (codes, target, prediction), 1)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(draws):
        counts = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)))
        matrix = np.tensordot(counts, grouped, axes=(0, 0))
        denominator = matrix.sum(0) + matrix.sum(1)
        f1 = np.divide(2 * matrix.diagonal(), denominator,
                       out=np.zeros(classes, dtype=float), where=denominator > 0)
        scores.append((matrix.trace() / matrix.sum(), f1.mean()))
    bounds = np.quantile(scores, [.025, .975], axis=0)
    return pd.DataFrame({'Métrica': ['Accuracy', 'Macro-F1'],
                         'Límite inferior 95 %': bounds[0], 'Límite superior 95 %': bounds[1]})
