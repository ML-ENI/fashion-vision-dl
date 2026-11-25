"""Cargar experimentos terminados para analizar sus predicciones sin entrenar."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_saved_experiment(context):
    folder = Path(context['root']) / 'results/metrics/04_dense'
    pointer = folder / 'CURRENT.json'
    candidates = []
    if pointer.exists():
        candidates.append(folder / json.loads(pointer.read_text())['run_dir'])
    runs = folder / 'runs'
    if runs.exists():
        candidates.extend(path for path in sorted(runs.iterdir(), reverse=True) if path.is_dir())
    for directory in dict.fromkeys(candidates):
        path = directory / 'summary.json'
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        if summary.get('status') != 'completed':
            continue
        if summary.get('identity', {}).get('manifest_sha256') != context['manifest_sha256']:
            continue
        # Cambiar métricas, figuras o explicaciones no cambia el experimento guardado.
        if any(summary['config'].get(key) != context['config'].get(key)
               for key in ['architecture', 'training', 'variants']):
            continue
        for name, digest in summary['artifacts_sha256'].items():
            artifact = (directory / name).resolve()
            if not artifact.is_relative_to(directory.resolve()):
                raise ValueError('Ruta de artefacto fuera del experimento')
            actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError(f'Predicción o resultado guardado modificado: {name}')
        print(f'Experimento {summary["experiment_id"]} cargado; no se entrena ninguna red.', flush=True)
        return summary, directory
    raise FileNotFoundError(
        'No hay un entrenamiento terminado compatible con este dataset y arquitectura. '
        'NO se ha entrenado una red. Para entrenar por primera vez con estos datos, '
        'activa ENTRENAR_SI_FALTA=True en la celda inicial de configuración.'
    )
