"""07: DINO fijo + clasificador lineal. Cada operación costosa es explícita.

Los resultados completos se leen antes de necesitar pesos o torchvision.
La extracción se confirma por lotes en SQLite; las semillas terminadas se
reutilizan. Este módulo no cambia el dataset ni abre fotografías de test.
"""
from __future__ import annotations

import copy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import urllib.request

import numpy as np
import pandas as pd

from src.data.data_contract import sha256
from src.training.transfer_cache import FeatureStore, identity
from src.training.transfer_results import find_saved


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def load_context(root):
    # Reutilizamos el contrato de 04, no sus arrays RGB64 ni su presupuesto.
    from src.training.dense_experiment import load_context as load_data
    context = load_data(root)
    context['config'] = json.loads((Path(root) / 'config/transfer_v1.json').read_text(encoding='utf-8'))
    provenance = Path(root) / 'results/metrics/03_eda/dataset_provenance.json'
    if not provenance.is_file() or json.loads(provenance.read_text())['manifest_sha256'] != context['manifest_sha256']:
        raise ValueError('Ejecuta y guarda primero 03 con el dataset actual de 02.')
    return context


def experiment_identity(context):
    config = context['config']
    return {'manifest_sha256': context['manifest_sha256'],
            'version': config['version'], 'model': config['model'],
            'preprocessing': config['preprocessing'], 'training': config['training'],
            'selection': config['selection'], 'early_stopping': config['early_stopping']}


def saved_experiment(context):
    found = find_saved(context['root'], experiment_identity(context))
    if found is not None:
        summary = found[0]
        if summary['labels'] != context['names'] or sorted(f['seed'] for f in summary['fits']) != sorted(context['config']['training']['seeds']):
            raise ValueError('Clases o semillas incompletas en el experimento guardado.')
    return found


def weight_paths(context):
    folder = context['root'] / 'tmp/07_transfer/weights'
    return folder / 'dino_resnet50.pth', folder / 'receipt.json'


def prepare_weights(context, *, allow_download=False):
    """Descargar solo bajo bandera; registrar y verificar la huella observada."""
    path, receipt_path = weight_paths(context)
    url = context['config']['model']['url']
    if path.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['url'] != url or receipt['sha256'] != sha256(path):
            raise ValueError('Los pesos locales han cambiado o pertenecen a otro modelo.')
        return receipt
    if path.exists() or receipt_path.exists():
        raise ValueError('Pesos incompletos: falta el archivo o su recibo. Revisar antes de continuar.')
    if not allow_download:
        raise FileNotFoundError('Faltan los pesos. No se ha descargado nada: activa DESCARGAR_PESOS para descargarlos una vez.')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.download')
    print('Descargando los pesos oficiales de DINO; esta operación necesita internet.', flush=True)
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as target:
        while block := response.read(1024 * 1024):
            target.write(block)
    os.replace(temporary, path)
    receipt = {'url': url, 'sha256': sha256(path), 'bytes': path.stat().st_size,
               'checksum_note': 'Huella calculada al descargar; no es una firma ni un checksum publicado por los autores.'}
    write_json(receipt_path, receipt)
    return receipt


def feature_recipe(context):
    _, receipt_path = weight_paths(context)
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text())
    if receipt['url'] != context['config']['model']['url']:
        raise ValueError('Recibo de otro checkpoint.')
    return {'weights_sha256': receipt['sha256'], 'dimension': context['config']['model']['dimension'],
            'preprocessing': context['config']['preprocessing'], 'implementation': 'torchvision-resnet50-eval-v1'}


def all_rows(context):
    return pd.concat([context['rows']['train'], context['rows']['validation']], ignore_index=True)


def feature_status(context):
    recipe = feature_recipe(context)
    rows = all_rows(context)
    if recipe is None:
        return {'required': len(rows), 'cached': 0, 'missing': len(rows)}
    with FeatureStore(context['root'] / 'tmp/07_transfer/features', recipe) as store:
        known = store.keys()
    count = int(rows.prepared_sha256.isin(known).sum())
    return {'required': len(rows), 'cached': count, 'missing': len(rows) - count}


def build_encoder(context):
    import torch
    from torch import nn
    try:
        from torchvision.models import resnet50
    except (ImportError, RuntimeError) as error:
        raise RuntimeError('Se necesita torchvision compatible con PyTorch. Consulta requirements-transfer.txt; no se instala automáticamente.') from error
    torch.set_num_threads(context['config']['training']['threads'])
    torch.use_deterministic_algorithms(True)
    model = resnet50(weights=None)  # No descarga pesos supervisados de ImageNet.
    model.fc = nn.Identity()       # Conserva solo la parte que describe la foto.
    state = torch.load(weight_paths(context)[0], map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)    # Ningún peso del encoder puede entrenarse.
    model.eval()                  # Tampoco cambian las estadísticas de BatchNorm.
    return model


def image_batch(context, rows):
    import torch
    from PIL import Image
    recipe = context['config']['preprocessing']
    size = recipe['image_size']
    values = []
    for row in rows.itertuples():
        path = (context['processed'] / row.image_path).resolve()
        if not path.is_relative_to(context['processed'].resolve()) or sha256(path) != row.prepared_sha256:
            raise ValueError(f'JPEG preparado ausente o modificado: {row.article_id}')
        with Image.open(path) as image:
            rgb = image.convert('RGB').resize((size, size), Image.Resampling.LANCZOS)
            values.append(np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255)
    pixels = np.stack(values)
    mean = np.asarray(recipe['mean'], dtype=np.float32)[None, :, None, None]
    std = np.asarray(recipe['std'], dtype=np.float32)[None, :, None, None]
    return torch.from_numpy((pixels - mean) / std)


def cost_path(context):
    key = identity({'experiment': experiment_identity(context), 'recipe': feature_recipe(context),
                    'batch_size': context['config']['extraction_batch_size']})
    return context['root'] / 'results/metrics/07_transfer/cost' / (key + '.json')


def extract_features(context, *, pilot=False):
    """Llamada explícita del notebook: medir un piloto o completar solo lo que falta."""
    import torch
    from src.training.dense_experiment import PeakRSS
    prepare_weights(context, allow_download=False)
    recipe = feature_recipe(context)
    config = context['config']
    if pilot and cost_path(context).is_file():
        print('Medición de coste ya guardada; no se repite el piloto.', flush=True)
        return json.loads(cost_path(context).read_text())
    if not pilot and not cost_path(context).is_file():
        raise FileNotFoundError('Mide primero el coste con MEDIR_COSTE=True y revisa la estimación antes de activar EXTRAER_FOTOS.')
    rows = all_rows(context)
    batch_size = config['extraction_batch_size']
    with FeatureStore(context['root'] / 'tmp/07_transfer/features', recipe, create=True) as store:
        pending = rows.loc[~rows.prepared_sha256.isin(store.keys())].drop_duplicates('prepared_sha256')
        missing_before = len(pending)
        if pilot:
            pending = pending.iloc[:batch_size * config['pilot_batches']]
        if pending.empty:
            print('Todas las fotos necesarias ya tienen su representación guardada; no se ejecuta DINO.', flush=True)
            return None
        encoder = build_encoder(context)
        started = time.perf_counter()
        with PeakRSS() as memory, torch.inference_mode():
            for start in range(0, len(pending), batch_size):
                block = pending.iloc[start:start + batch_size]
                vectors = encoder(image_batch(context, block)).numpy()
                store.put(block.prepared_sha256.tolist(), vectors)
                done = min(start + batch_size, len(pending))
                elapsed = time.perf_counter() - started
                remaining = elapsed / done * (len(pending) - done)
                print(f'Fotos nuevas guardadas: {done}/{len(pending)}; quedan aprox. {remaining/60:.1f} min', flush=True)
        seconds = time.perf_counter() - started
        report = {'pilot': pilot, 'images_processed': len(pending), 'seconds': seconds,
                  'seconds_per_image': seconds / len(pending), 'peak_rss_bytes': memory.peak,
                  'frozen_encoder_parameters': sum(p.numel() for p in encoder.parameters()),
                  'remaining_estimate_seconds': seconds / len(pending) * (missing_before - len(pending)),
                  'recipe': recipe, 'manifest_sha256': context['manifest_sha256'],
                  'versions': {p: importlib.metadata.version(p) for p in ['torch', 'torchvision', 'numpy', 'Pillow']},
                  'note': 'Estimación orientativa en este ordenador; incluye lectura, verificación, inferencia y escritura. No mide el preentrenamiento externo ni el ajuste del clasificador.'}
        if pilot:
            write_json(cost_path(context), report)
        else:
            write_json(cost_path(context).with_suffix('.extraction.json'), report)
        return report


def train_heads(context):
    """Entrenar solo las semillas que faltan; requiere vectores completos y bandera externa."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from src.training.dense_experiment import PeakRSS, SelectionState, evaluate, train_epoch
    found = saved_experiment(context)
    if found is not None:
        print('Experimento completo cargado; no se vuelve a entrenar.', flush=True)
        return found
    if feature_status(context)['missing']:
        raise FileNotFoundError('Faltan representaciones. No se ha entrenado ni extraído ninguna foto desde esta función.')
    recipe = feature_recipe(context)
    config = context['config']
    budget = config['training']
    torch.set_num_threads(budget['threads'])
    torch.use_deterministic_algorithms(True)
    with FeatureStore(context['root'] / 'tmp/07_transfer/features', recipe) as store:
        tensors = {split: torch.from_numpy(store.read(frame.prepared_sha256.tolist()))
                   for split, frame in context['rows'].items()}
    targets = {split: torch.tensor(frame.label_id.astype(int).to_numpy(), dtype=torch.long)
               for split, frame in context['rows'].items()}
    exp_identity = experiment_identity(context)
    directory = context['root'] / 'results/metrics/07_transfer/runs' / identity(exp_identity)
    directory.mkdir(parents=True, exist_ok=True)
    fits = []
    for seed in budget['seeds']:
        prefix = f'linear_seed{seed}'
        marker = directory / (prefix + '.json')
        if marker.is_file():
            completed = json.loads(marker.read_text())
            if completed['feature_recipe'] != recipe:
                raise ValueError('Esta semilla se calculó con otros pesos o representaciones.')
            for name, digest in completed['artifacts_sha256'].items():
                artifact = (directory / name).resolve()
                if not artifact.is_relative_to(directory.resolve()) or sha256(artifact) != digest:
                    raise ValueError('Resultados de una semilla guardada modificados.')
            fits.append(completed)
            print(f'Semilla {seed} terminada anteriormente: se reutiliza.', flush=True)
            continue
        torch.manual_seed(seed)
        model = nn.Linear(config['model']['dimension'], len(context['names']))
        optimizer = torch.optim.Adam(model.parameters(), lr=budget['learning_rate'], weight_decay=budget['weight_decay'])
        loader = DataLoader(TensorDataset(tensors['train'], targets['train']),
                            batch_size=budget['batch_size'], shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(seed))
        selection = SelectionState(budget['patience'])
        history, best = [], None
        started = time.perf_counter()
        with PeakRSS() as memory:
            for epoch in range(1, budget['max_epochs'] + 1):
                online_loss = train_epoch(model, loader, optimizer, nn.CrossEntropyLoss())
                train, _ = evaluate(model, tensors['train'], targets['train'], budget['batch_size'])
                validation, _ = evaluate(model, tensors['validation'], targets['validation'], budget['batch_size'])
                save, stop = selection.update(epoch, validation['loss'], validation['macro_f1'])
                if save:
                    best = copy.deepcopy(model.state_dict())
                history.append({'epoch': epoch, 'online_loss': online_loss,
                                **{'train_' + k: v for k, v in train.items()},
                                **{'validation_' + k: v for k, v in validation.items()}})
                print(f'Semilla {seed}, época {epoch}: macro-F1 validation={validation["macro_f1"]:.4f}', flush=True)
                if stop:
                    break
        training_seconds = time.perf_counter() - started
        model.load_state_dict(best)
        prediction_start = time.perf_counter()
        _, logits = evaluate(model, tensors['validation'], targets['validation'], budget['batch_size'])
        prediction_seconds = time.perf_counter() - prediction_start
        frame = context['rows']['validation'][['article_id', 'group_id', 'label_id']].copy()
        probabilities = logits.softmax(1).numpy()
        frame['predicted_id'] = probabilities.argmax(1)
        for column in range(len(context['names'])):
            frame[f'p_{column}'] = probabilities[:, column]
        predictions_path = directory / (prefix + '_predictions.csv.gz')
        history_path = directory / (prefix + '_history.csv')
        checkpoint = directory / (prefix + '_head.npz')
        # Solo la pequeña capa final; sin repetir los pesos de ResNet en cada semilla.
        np.savez_compressed(checkpoint, weight=best['weight'].numpy(), bias=best['bias'].numpy())
        with np.load(checkpoint, allow_pickle=False) as restored:
            torch.testing.assert_close(torch.from_numpy(restored['weight']), best['weight'], rtol=0, atol=0)
            torch.testing.assert_close(torch.from_numpy(restored['bias']), best['bias'], rtol=0, atol=0)
        frame.to_csv(predictions_path, index=False, compression='gzip')
        pd.DataFrame(history).to_csv(history_path, index=False)
        completed = {'seed': seed, 'best_epoch': selection.best_epoch,
                     'epochs_run': len(history), 'training_seconds': training_seconds,
                     'trainable_parameters': sum(p.numel() for p in model.parameters()),
                     'peak_rss_bytes': memory.peak, 'predictions': predictions_path.name,
                     'history': history_path.name, 'feature_recipe': recipe,
                     'head_validation_ms_per_image': prediction_seconds * 1000 / len(frame),
                     'artifacts_sha256': {p.name: sha256(p) for p in [predictions_path, history_path, checkpoint]}}
        write_json(marker, completed)
        fits.append(completed)
    artifacts = {name: digest for fit in fits for name, digest in fit['artifacts_sha256'].items()}
    artifacts.update({f'linear_seed{fit["seed"]}.json': sha256(directory / f'linear_seed{fit["seed"]}.json') for fit in fits})
    summary = {'status': 'completed', 'identity': exp_identity, 'labels': context['names'],
               'feature_recipe': recipe, 'fits': fits, 'artifacts_sha256': artifacts,
               'environment': {'python': platform.python_version(),
                               **{p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'pandas', 'scikit-learn']}},
               'source_sha256': {name: sha256(context['root'] / name) for name in
                                 ['src/training/transfer_experiment.py', 'src/training/transfer_cache.py']},
               'cost_note': 'training_seconds y latencia de la cabeza excluyen extracción DINO. Informar ambos costes por separado.'}
    write_json(directory / 'summary.json', summary)
    return saved_experiment(context)
