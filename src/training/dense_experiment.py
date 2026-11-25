"""Experimento Dense local, con procedencia y sin abrir fotografías de test.

El notebook expone el modelo y el bucle didáctico; este módulo evita duplicar
la implementación al repetir las seis ejecuciones y permite probar sus controles.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import threading
import time

import numpy as np
import pandas as pd
from PIL import Image
import psutil
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.dense import DenseClassifier, parameter_count


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def load_context(root):
    root = Path(root)
    processed = root / 'data/processed'
    pointer = processed / 'CURRENT.json'
    if not pointer.exists():
        raise FileNotFoundError('Primero ejecuta el notebook 02 para publicar la cohorte completa; el conjunto anterior no está activo')
    run = processed / json.loads(pointer.read_text())['run_dir']
    preparation = json.loads((run / 'config.json').read_text())
    if 'max_images_per_class' in preparation:
        raise ValueError('La publicación contiene cupos antiguos: ejecutar 02 con la configuración vigente antes de entrenar')
    requested = json.loads((root / 'config/local_images_v1.json').read_text(encoding='utf-8'))
    from src.data.data_contract import require_current_visual_review
    require_current_visual_review(root, run, requested, preparation)
    if (set(preparation.get('include_classes', [])) != set(requested['include_classes']) or
            set(preparation.get('include_product_groups') or []) != set(requested.get('include_product_groups') or [])):
        raise ValueError('El conjunto preparado tiene otro alcance de prendas; ejecutar 02 antes de entrenar')
    checks = json.loads((run / 'checksums.json').read_text())
    for name in ['image_manifest.csv.gz', 'labels.json']:
        if sha256(run / name) != checks[name]:
            raise ValueError(f'La publicación ha cambiado: {name}')
    manifest = pd.read_csv(run / 'image_manifest.csv.gz', keep_default_na=False,
                           dtype={'article_id': 'string', 'product_code': 'string', 'group_id': 'string'})
    active = manifest.loc[manifest.split.isin(['train', 'validation', 'test'])].copy()
    for field in ['group_id', 'product_code', 'prepared_pixel_sha256']:
        if active.groupby(field).split.nunique().max() != 1:
            raise ValueError(f'Fuga de información entre particiones: {field}')
    labels = json.loads((run / 'labels.json').read_text())
    names = sorted(labels, key=labels.get)
    if list(range(len(names))) != [labels[name] for name in names]:
        raise ValueError('Índices de clase no consecutivos')
    rows = {split: active.loc[active.split.eq(split)].reset_index(drop=True)
            for split in ['train', 'validation']}
    for split, frame in rows.items():
        if frame.empty or (split == 'train' and set(frame.label) != set(names)):
            raise ValueError('Partición vacía o clases ausentes')
        if not set(frame.label).issubset(names):
            raise ValueError('Partición con etiquetas desconocidas')
        if not all(int(row.label_id) == labels[row.label] for row in frame.itertuples()):
            raise ValueError('Etiquetas e índices no coinciden')
    config = json.loads((root / 'config/dense_baseline_v1.json').read_text(encoding='utf-8'))
    # La salida sigue las clases que sobreviven al filtro autorizado de 02.
    config['architecture']['classes'] = len(names)
    return {'root': root, 'processed': processed, 'run': run, 'rows': rows,
            'names': names, 'config': config, 'manifest_sha256': checks['image_manifest.csv.gz'],
            'split_counts': active.split.value_counts().to_dict()}


def prepare_arrays(context):
    """Caché uint8 local; hashes de TODOS los JPEG train/validation antes de usarla."""
    size = context['config']['architecture']['image_size']
    cache = context['root'] / 'tmp/dense_rgb' / (context['manifest_sha256'][:16] + f'_{size}_lanczos_v1')
    cache.mkdir(parents=True, exist_ok=True)
    raw = {}
    for split, rows in context['rows'].items():
        path = cache / f'{split}.npy'
        marker = cache / f'{split}.json'
        valid_cache = False
        if path.exists() and marker.exists():
            metadata = json.loads(marker.read_text())
            valid_cache = metadata['sha256'] == sha256(path) and metadata['article_ids'] == rows.article_id.tolist()
        pending = cache / f'{split}.pending.npy'
        array = (np.load(path, mmap_mode='r') if valid_cache else
                 np.lib.format.open_memmap(pending, mode='w+', shape=(len(rows), 3, size, size), dtype=np.uint8))
        if array.shape != (len(rows), 3, size, size) or array.dtype != np.uint8:
            raise ValueError('Caché con forma o tipo incorrectos')
        for i, row in enumerate(rows.itertuples()):
            source = context['processed'] / row.image_path
            if sha256(source) != row.prepared_sha256:
                raise ValueError(f'JPEG preparado modificado: {row.article_id}')
            if not valid_cache:
                with Image.open(source) as image:
                    image = image.convert('RGB').resize((size, size), Image.Resampling.LANCZOS)
                    array[i] = np.asarray(image).transpose(2, 0, 1)
            if (i + 1) % 4000 == 0:
                print(f'{split}: {i+1:,} imágenes verificadas'.replace(',', '.'), flush=True)
        if not valid_cache:
            array.flush()
            del array
            os.replace(pending, path)
            write_json(marker, {'sha256': sha256(path), 'article_ids': rows.article_id.tolist()})
        raw[split] = np.load(path, mmap_mode='r')
    # Momentos poblacionales por canal en la representación RGB64, no en RGB224.
    sums = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    for start in range(0, len(raw['train']), 128):
        block = raw['train'][start:start+128].astype(np.float64) / 255.0
        sums += block.sum(axis=(0, 2, 3))
        squares += np.square(block).sum(axis=(0, 2, 3))
    pixels = len(raw['train']) * size * size
    mean = sums / pixels
    std = np.sqrt(np.maximum(squares / pixels - mean**2, 0))
    if np.any(std <= 0):
        raise ValueError('Canal sin variación')
    return raw, {'mean': mean.tolist(), 'std': std.tolist(), 'image_size': size,
                 'partition': 'train', 'pixels_per_channel': pixels,
                 'method': 'population pixel moments after RGB64 Lanczos resize'}


def normalized_tensors(raw, normalization):
    """Arrays respaldados por disco: la cohorte completa no exige una copia RAM."""
    mean = np.asarray(normalization['mean'], dtype=np.float32)[None, :, None, None]
    std = np.asarray(normalization['std'], dtype=np.float32)[None, :, None, None]
    tensors = {}
    for split, values in raw.items():
        path = Path(values.filename).with_name(split+'_normalized.npy')
        marker = path.with_suffix('.json')
        expected = {'raw_sha256': sha256(Path(values.filename)), 'normalization': normalization,
                    'shape': list(values.shape), 'dtype': 'float32'}
        cached = json.loads(marker.read_text()) if marker.exists() else {}
        if not (path.exists() and cached.get('inputs') == expected and
                cached.get('sha256') == sha256(path)):
            temporary = path.with_name(path.stem + '.pending.npy')
            mapped = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=values.shape)
            for start in range(0, len(values), 128):
                mapped[start:start+128] = (values[start:start+128].astype(np.float32)/255-mean)/std
            mapped.flush()
            del mapped
            os.replace(temporary, path)
            write_json(marker, {'inputs': expected, 'sha256': sha256(path)})
        tensors[split] = torch.from_numpy(np.load(path, mmap_mode='c'))
    return tensors


def measure_feasibility(context, tensors, directory):
    """Una época real, descartada: decidir por coste, nunca por su puntuación."""
    path = directory / 'cost_feasibility.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    config = context['config']
    budget = config['training']
    torch.manual_seed(config['cost_feasibility']['pilot_seed'])
    model = model_from_config(config, 0.0)
    y = {s: torch.tensor(f.label_id.astype(int).to_numpy(), dtype=torch.long) for s, f in context['rows'].items()}
    loader = DataLoader(TensorDataset(tensors['train'], y['train']), batch_size=budget['batch_size'],
                        shuffle=True, generator=torch.Generator().manual_seed(901), num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=budget['learning_rate'])
    started = time.perf_counter()
    with PeakRSS() as memory:
        train_epoch(model, loader, optimizer, nn.CrossEntropyLoss())
        evaluate(model, tensors['train'], y['train'], budget['batch_size'])
        evaluate(model, tensors['validation'], y['validation'], budget['batch_size'])
    seconds = time.perf_counter()-started
    estimate = seconds*budget['max_epochs']*len(budget['seeds'])*len(config['variants'])*config['cost_feasibility']['safety_factor']
    report = {'dataset_run': context['run'].name, 'train_images': len(y['train']), 'validation_images': len(y['validation']),
              'pilot_epoch_seconds': seconds, 'peak_process_rss_bytes': memory.peak,
              'conservative_six_fit_upper_seconds': estimate, 'budget_seconds': config['cost_feasibility']['max_total_training_seconds'],
              'fits_budget': estimate <= config['cost_feasibility']['max_total_training_seconds'],
              'sampling_applied': False,
              'decision': 'keep_all_eligible_images' if estimate <= config['cost_feasibility']['max_total_training_seconds'] else 'review_computational_budget_before_training',
              'note': 'Pilot weights discarded; no pilot prediction or metric used for selection. Includes evaluation time; preprocessing separate.'}
    write_json(path, report)
    print('Coste medido: '+str(round(seconds, 1))+' s/época; estimación máxima conservadora: '+str(round(estimate/60, 1))+' min', flush=True)
    if not report['fits_budget']:
        raise RuntimeError('El coste supera el presupuesto: revisar explícitamente el diseño antes de entrenar, sin eliminar fotografías automáticamente')
    return report


def scores(y, pred, classes):
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=range(classes), average='macro', zero_division=0))}


def grouped_intervals(y, pred, groups, classes, draws=300, seed=42, other=None):
    """Bootstrap por productos; other devuelve diferencia emparejada, no IC entre semillas."""
    _, inverse = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    def cubes_for(values):
        cubes = np.zeros((inverse.max()+1, classes, classes), dtype=np.int64)
        np.add.at(cubes, (inverse, y, values), 1)
        return cubes
    cubes = cubes_for(pred)
    second = cubes_for(other) if other is not None else None
    def calculate(cm):
        tp = cm.diagonal()
        den = cm.sum(0) + cm.sum(1)
        return np.array([tp.sum()/cm.sum(), np.divide(2*tp, den, out=np.zeros(classes), where=den>0).mean()])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        indexes = rng.integers(len(cubes), size=len(cubes))
        result = calculate(cubes[indexes].sum(0))
        if second is not None:
            result -= calculate(second[indexes].sum(0))
        values.append(result)
    return np.quantile(values, [.025, .975], axis=0).tolist()


class SelectionState:
    """Separar el criterio para detener de aquel para guardar el checkpoint."""
    def __init__(self, patience):
        self.patience = patience
        self.best_loss, self.best_f1 = float('inf'), -1.0
        self.stale = 0
        self.best_epoch = None

    def update(self, epoch, loss, f1):
        if not np.isfinite(loss) or not np.isfinite(f1):
            raise ValueError('Entrenamiento no finito')
        if loss < self.best_loss:
            self.best_loss, self.stale = loss, 0
        else:
            self.stale += 1
        save = f1 > self.best_f1
        if save:
            self.best_f1, self.best_epoch = f1, epoch
        return save, self.stale >= self.patience


def train_epoch(model, loader, optimizer, criterion):
    model.train()  # activa dropout únicamente durante entrenamiento
    total = 0.0
    for images, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        if not torch.isfinite(loss):
            raise ValueError('Pérdida no finita')
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(labels)
    return total / len(loader.dataset)


@torch.inference_mode()
def evaluate(model, images, labels, batch_size=32):
    model.eval()  # desactiva dropout: train y validation comparables
    losses = 0.0
    all_logits = []
    for start in range(0, len(images), batch_size):
        logits = model(images[start:start+batch_size])
        target = labels[start:start+batch_size]
        losses += float(nn.functional.cross_entropy(logits, target, reduction='sum'))
        all_logits.append(logits)
    logits = torch.cat(all_logits)
    pred = logits.argmax(1).numpy()
    return {'loss': losses / len(labels), **scores(labels.numpy(), pred, logits.shape[1])}, logits


class PeakRSS:
    """Muestreo RSS del proceso, NO memoria exclusiva de tensores ni VRAM."""
    def __enter__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(.05):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


def model_from_config(config, dropout):
    a = config['architecture']
    return DenseClassifier(a['image_size'], a['hidden_dims'], a['classes'], dropout)


def fit_one(context, tensors, normalization, variant, seed, directory, weights_dir):
    config = context['config']
    budget = config['training']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = model_from_config(config, variant['dropout'])
    y = {s: torch.tensor(frame.label_id.astype(int).to_numpy(), dtype=torch.long)
         for s, frame in context['rows'].items()}
    loader = DataLoader(TensorDataset(tensors['train'], y['train']), batch_size=budget['batch_size'],
                        shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=budget['learning_rate'], weight_decay=budget['weight_decay'])
    criterion = nn.CrossEntropyLoss()  # sin pesos; logits sin softmax
    selection = SelectionState(budget['patience'])
    history = []
    best_state = None
    started = time.perf_counter()
    with PeakRSS() as memory:
        for epoch in range(1, budget['max_epochs']+1):
            epoch_start = time.perf_counter()
            online_loss = train_epoch(model, loader, optimizer, criterion)
            train, _ = evaluate(model, tensors['train'], y['train'], budget['batch_size'])
            val, _ = evaluate(model, tensors['validation'], y['validation'], budget['batch_size'])
            save, stop = selection.update(epoch, val['loss'], val['macro_f1'])
            if save:
                best_state = copy.deepcopy(model.state_dict())
            history.append({'epoch': epoch, 'online_train_loss': online_loss,
                            **{'train_'+k: v for k, v in train.items()},
                            **{'validation_'+k: v for k, v in val.items()},
                            'epoch_seconds': time.perf_counter()-epoch_start})
            print((f"{variant['name']} seed={seed} epoch={epoch:02d} "
                   f"loss_val={val['loss']:.4f} F1_val={val['macro_f1']:.4f} "
                   f"({history[-1]['epoch_seconds']:.1f}s)").replace('.', ','), flush=True)
            if stop:
                break
    train_seconds = time.perf_counter()-started
    model.load_state_dict(best_state)
    train_result, _ = evaluate(model, tensors['train'], y['train'], budget['batch_size'])
    validation_result, logits = evaluate(model, tensors['validation'], y['validation'], budget['batch_size'])
    key = f"{variant['name']}_seed{seed}"
    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = weights_dir / f'{key}.pt'
    torch.save({'state_dict': best_state, 'config': config, 'normalization': normalization,
                'seed': seed, 'variant': variant, 'epoch': selection.best_epoch}, checkpoint)
    # Reabrir el checkpoint y comprobar sus logits antes de publicar los resultados.
    restored = model_from_config(config, variant['dropout'])
    restored.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict'])
    restored.eval()
    with torch.inference_mode():
        if not torch.equal(restored(tensors['validation'][:32]), logits[:32]):
            raise ValueError('Checkpoint no reproduce los logits guardados')
    durations = []
    with torch.inference_mode():
        model(tensors['validation'][:budget['batch_size']])  # calentamiento
        for _ in range(3):
            t = time.perf_counter()
            for start in range(0, len(y['validation']), budget['batch_size']):
                model(tensors['validation'][start:start+budget['batch_size']])
            durations.append(time.perf_counter()-t)
    probabilities = logits.softmax(1).numpy()
    pred = probabilities.argmax(1)
    frame = context['rows']['validation']
    predictions = frame[['article_id', 'group_id', 'label_id']].copy()
    predictions['predicted_label_id'] = pred
    predictions['confidence'] = probabilities.max(1)
    for index in range(len(context['names'])):
        predictions[f'p_{index}'] = probabilities[:, index]
    predictions.to_csv(directory / f'{key}_predictions.csv.gz', index=False)
    pd.DataFrame(history).to_csv(directory / f'{key}_history.csv', index=False)
    report = classification_report(y['validation'].numpy(), pred, labels=range(len(context['names'])),
                                   target_names=context['names'], output_dict=True, zero_division=0)
    write_json(directory / f'{key}_per_class.json', report)
    ci = grouped_intervals(y['validation'].numpy(), pred, frame.group_id, len(context['names']),
                           draws=config['bootstrap']['draws'], seed=config['bootstrap']['seed'])
    linear_weights = sum(layer.in_features*layer.out_features for layer in model.modules() if isinstance(layer, nn.Linear))
    result = {'variant': variant['name'], 'seed': seed, 'dropout': variant['dropout'],
              'parameters': parameter_count(model), 'approx_linear_flops_per_image': 2*linear_weights,
              'epochs_run': len(history), 'selected_epoch': selection.best_epoch,
              'stopped_early': len(history) < budget['max_epochs'],
              'train_seconds': train_seconds, 'inference_ms_per_image': 1000*np.median(durations)/len(frame),
              'peak_process_rss_bytes': memory.peak, 'checkpoint': str(checkpoint.relative_to(context['root'])),
              'checkpoint_sha256': sha256(checkpoint),
              **{'train_'+k: v for k, v in train_result.items()},
              **{'validation_'+k: v for k, v in validation_result.items()},
              'validation_group_bootstrap_ci95_accuracy': [ci[0][0], ci[1][0]],
              'validation_group_bootstrap_ci95_macro_f1': [ci[0][1], ci[1][1]]}
    write_json(directory / f'{key}_metrics.json', result)
    return result


def experiment_signature(context):
    root = context['root']
    files = ['config/dense_baseline_v1.json', 'config/experimental_protocol_v1.json',
             'src/models/dense.py', 'src/training/dense_experiment.py']
    versions = {name: importlib.metadata.version(name) for name in ['torch', 'numpy', 'pandas', 'Pillow', 'scikit-learn', 'psutil']}
    identity = {'manifest_sha256': context['manifest_sha256'], 'sources': {name: sha256(root/name) for name in files},
                'versions': versions, 'platform': platform.platform(), 'processor': platform.processor()}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return signature, identity


def run_or_load(context, tensors, normalization):
    config = context['config']
    if len(context['names']) != config['architecture']['classes']:
        raise ValueError('Número de clases distinto del registrado')
    torch.set_num_threads(config['training']['threads'])
    torch.use_deterministic_algorithms(True)
    signature, identity = experiment_signature(context)
    directory = context['root'] / 'results/metrics/04_dense/runs' / signature
    directory.mkdir(parents=True, exist_ok=True)
    weights_dir = context['root'] / 'results/models/04_dense' / signature
    marker = directory / 'summary.json'
    if marker.exists():
        summary = json.loads(marker.read_text(encoding='utf-8'))
        for name, checksum in summary['artifacts_sha256'].items():
            if sha256(directory / name) != checksum:
                raise ValueError(f'Resultado modificado: {name}')
        for record in summary['runs']:
            if sha256(context['root']/record['checkpoint']) != record['checkpoint_sha256']:
                raise ValueError('Checkpoint ausente o modificado')
        print('Resultados verificados reutilizados: '+signature)
        return summary, directory
    feasibility = measure_feasibility(context, tensors, directory)
    if not feasibility['fits_budget']:
        raise RuntimeError('Es necesario revisar el presupuesto computacional')
    write_json(directory / 'design.json', {'identity': identity, 'config': config, 'normalization': normalization,
                                         'dataset_run': context['run'].name, 'test_images_opened': 0})
    results = []
    for variant in config['variants']:
        for seed in config['training']['seeds']:
            # Recuperar ejecuciones terminadas si el proceso se interrumpió entre semillas.
            key = f"{variant['name']}_seed{seed}"
            metrics = directory / f'{key}_metrics.json'
            if metrics.exists():
                result = json.loads(metrics.read_text(encoding='utf-8'))
                if sha256(context['root']/result['checkpoint']) != result['checkpoint_sha256']:
                    raise ValueError('Checkpoint parcial modificado')
                print('Recuperada ejecución completa: '+key, flush=True)
            else:
                result = fit_one(context, tensors, normalization, variant, seed, directory, weights_dir)
            results.append(result)
    pd.DataFrame(results).to_csv(directory / 'all_runs.csv', index=False)
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.name != 'summary.json')
    summary = {'status': 'completed', 'experiment_id': signature, 'dataset_run': context['run'].name,
               'config': config, 'identity': identity, 'normalization': normalization,
               'hardware': {'device': 'cpu', 'processor': platform.processor(), 'physical_cores': psutil.cpu_count(logical=False),
                            'threads_used': torch.get_num_threads(), 'ram_total_bytes': psutil.virtual_memory().total},
               'test_images_opened': 0, 'test_predictions_generated': 0, 'cost_feasibility': feasibility,
               'cost_notes': {'train_seconds': 'includes all epoch train and train/validation evaluation; excludes resize and final artifact export',
                              'latency': 'median of 3 warmed CPU passes over disk-backed normalized arrays; excludes JPEG decode, resize and normalization, but may include OS paging or disk reads',
                              'peak_process_rss_bytes': '50ms sampled entire Python process including mapped file pages and datasets; not incremental tensor memory',
                              'FLOPs': '2 per dense weight per image; excludes activations, bias, normalization and softmax'},
               'runs': results, 'artifacts_sha256': {p.name: sha256(p) for p in files}}
    write_json(marker, summary)
    write_json(directory.parents[1] / 'CURRENT.json', {'run_dir': 'runs/'+signature, 'status': 'completed'})
    return summary, directory
