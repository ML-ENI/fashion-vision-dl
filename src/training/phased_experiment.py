"""Experimentos por fases (05 CNN, 06 secuencias): mismas fotos y presupuesto que la Dense de 04.

Reutiliza la carga de datos, las métricas, la selección de checkpoint y el
bootstrap de ``dense_experiment``; cada notebook solo aporta su modelo. Cada
fase (p. ej. ``h1`` y después ``batchnorm``) se guarda por separado, para que
una fase posterior no obligue a repetir las anteriores. Nunca se abren fotos de test.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.dense import parameter_count
from src.training.dense_experiment import (
    PeakRSS, SelectionState, evaluate, grouped_intervals, load_context, sha256, train_epoch, write_json,
)


class PhasedExperiment:
    """Una configuración, una carpeta de resultados y una forma de construir el modelo."""

    def __init__(self, config, results, weights, sources, build_model, multiply_adds):
        self.config_path = config
        self.results = results
        self.weights = weights
        self.sources = list(sources)
        self.build_model = build_model          # (config, variant) -> nn.Module
        self.multiply_adds = multiply_adds      # (model, config) -> int

    # ---------- contexto y modelo ----------
    def load_context(self, root):
        """Mismo contexto de datos que 04 (publicación, filas y controles), con esta configuración."""
        root = Path(root)
        context = load_context(root)
        config = json.loads((root / self.config_path).read_text(encoding='utf-8'))
        # La salida sigue las clases de la publicación vigente, igual que en 04.
        config['architecture']['classes'] = len(context['names'])
        context['dense_config'] = context['config']
        context['config'] = config
        return context

    def model(self, config, variant):
        return self.build_model(config, variant)

    @staticmethod
    def labels(context):
        return {s: torch.tensor(f.label_id.astype(int).to_numpy(), dtype=torch.long)
                for s, f in context['rows'].items()}

    # ---------- coste ----------
    def measure_phase_cost(self, context, tensors, directory, phase):
        """Una época real descartada: se decide por coste, nunca por su puntuación."""
        path = directory / 'cost_feasibility.json'
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
        config = context['config']
        budget, cost = config['training'], config['cost_feasibility']
        variants = config['phases'][phase]['variants']
        y = self.labels(context)
        seconds_per_variant, peak = [], 0
        # Se mide cada variante de la fase: pueden tener costes muy distintos (p. ej. RNN frente a GRU).
        for variant in variants:
            torch.manual_seed(cost['pilot_seed'])
            model = self.model(config, variant)
            loader = DataLoader(TensorDataset(tensors['train'], y['train']), batch_size=budget['batch_size'],
                                shuffle=True, generator=torch.Generator().manual_seed(cost['pilot_seed']),
                                num_workers=0)
            optimizer = torch.optim.Adam(model.parameters(), lr=budget['learning_rate'])
            started = time.perf_counter()
            with PeakRSS() as memory:
                train_epoch(model, loader, optimizer, nn.CrossEntropyLoss())
                evaluate(model, tensors['train'], y['train'], budget['batch_size'])
                evaluate(model, tensors['validation'], y['validation'], budget['batch_size'])
            seconds_per_variant.append(time.perf_counter() - started)
            peak = max(peak, memory.peak)
        estimate = sum(seconds_per_variant) * budget['max_epochs'] * len(budget['seeds']) * cost['safety_factor']
        limit = config['phases'][phase]['max_total_training_seconds']
        report = {'phase': phase, 'dataset_run': context['run'].name, 'train_images': len(y['train']),
                  'validation_images': len(y['validation']),
                  'pilot_epoch_seconds': float(np.mean(seconds_per_variant)),
                  'pilot_epoch_seconds_by_variant': {v['name']: s for v, s in zip(variants, seconds_per_variant)},
                  'peak_process_rss_bytes': peak, 'fits': len(budget['seeds']) * len(variants),
                  'conservative_upper_seconds': estimate, 'budget_seconds': limit, 'fits_budget': estimate <= limit,
                  'sampling_applied': False,
                  'decision': ('keep_all_eligible_images' if estimate <= limit
                               else 'review_computational_budget_before_training'),
                  'note': 'Pilot weights discarded; no pilot metric used. Includes train and validation '
                          'evaluation; preprocessing separate.'}
        write_json(path, report)
        print(f'Coste medido: {np.mean(seconds_per_variant):.1f} s/época de media; estimación máxima '
              f'conservadora: {estimate/60:.1f} min'.replace('.', ','), flush=True)
        if not report['fits_budget']:
            raise RuntimeError('El coste supera el presupuesto de la fase: revisar explícitamente el diseño antes '
                               'de entrenar, sin eliminar fotografías automáticamente')
        return report

    # ---------- un entrenamiento ----------
    def fit_one(self, context, tensors, normalization, variant, seed, directory, weights_dir):
        """Mismo bucle que la Dense de 04: Adam, parada por pérdida y checkpoint por macro-F1."""
        config = context['config']
        budget = config['training']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = self.model(config, variant)
        y = self.labels(context)
        loader = DataLoader(TensorDataset(tensors['train'], y['train']), batch_size=budget['batch_size'],
                            shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=budget['learning_rate'],
                                     weight_decay=budget['weight_decay'])
        criterion = nn.CrossEntropyLoss()  # sin pesos de clase; logits sin softmax
        selection = SelectionState(budget['patience'])
        history, best_state = [], None
        started = time.perf_counter()
        with PeakRSS() as memory:
            for epoch in range(1, budget['max_epochs'] + 1):
                epoch_start = time.perf_counter()
                online_loss = train_epoch(model, loader, optimizer, criterion)
                train, _ = evaluate(model, tensors['train'], y['train'], budget['batch_size'])
                val, _ = evaluate(model, tensors['validation'], y['validation'], budget['batch_size'])
                save, stop = selection.update(epoch, val['loss'], val['macro_f1'])
                if save:
                    best_state = copy.deepcopy(model.state_dict())
                history.append({'epoch': epoch, 'online_train_loss': online_loss,
                                **{'train_' + k: v for k, v in train.items()},
                                **{'validation_' + k: v for k, v in val.items()},
                                'epoch_seconds': time.perf_counter() - epoch_start})
                print((f"{variant['name']} seed={seed} epoch={epoch:02d} loss_val={val['loss']:.4f} "
                       f"F1_val={val['macro_f1']:.4f} ({history[-1]['epoch_seconds']:.1f}s)").replace('.', ','),
                      flush=True)
                if stop:
                    break
        train_seconds = time.perf_counter() - started
        model.load_state_dict(best_state)
        train_result, _ = evaluate(model, tensors['train'], y['train'], budget['batch_size'])
        validation_result, logits = evaluate(model, tensors['validation'], y['validation'], budget['batch_size'])
        key = f"{variant['name']}_seed{seed}"
        weights_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = weights_dir / f'{key}.pt'
        torch.save({'state_dict': best_state, 'config': config, 'normalization': normalization,
                    'seed': seed, 'variant': variant, 'epoch': selection.best_epoch}, checkpoint)
        # Reabrir el checkpoint y comprobar sus logits antes de publicar resultados.
        restored = self.model(config, variant)
        restored.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict'])
        restored.eval()
        with torch.inference_mode():
            # Mismos batches que evaluate: el resultado exacto puede depender del tamaño del batch.
            size = budget['batch_size']
            check = min(len(logits), 2 * size)
            again = torch.cat([restored(tensors['validation'][s:min(s + size, check)]) for s in range(0, check, size)])
            if not torch.equal(again, logits[:check]):
                raise ValueError('El checkpoint no reproduce los logits guardados')
        durations = []
        with torch.inference_mode():
            model(tensors['validation'][:budget['batch_size']])  # calentamiento
            for _ in range(3):
                t = time.perf_counter()
                for start in range(0, len(y['validation']), budget['batch_size']):
                    model(tensors['validation'][start:start + budget['batch_size']])
                durations.append(time.perf_counter() - t)
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
        result = {'variant': variant['name'], 'seed': seed,
                  **{k: v for k, v in variant.items() if k != 'name'},
                  'parameters': parameter_count(model),
                  'approx_flops_per_image': 2 * self.multiply_adds(model, config),
                  'epochs_run': len(history), 'selected_epoch': selection.best_epoch,
                  'stopped_early': len(history) < budget['max_epochs'],
                  'train_seconds': train_seconds, 'inference_ms_per_image': 1000 * np.median(durations) / len(frame),
                  'peak_process_rss_bytes': memory.peak,
                  'checkpoint': checkpoint.relative_to(context['root']).as_posix(),
                  'checkpoint_sha256': sha256(checkpoint),
                  **{'train_' + k: v for k, v in train_result.items()},
                  **{'validation_' + k: v for k, v in validation_result.items()},
                  'validation_group_bootstrap_ci95_accuracy': [ci[0][0], ci[1][0]],
                  'validation_group_bootstrap_ci95_macro_f1': [ci[0][1], ci[1][1]]}
        write_json(directory / f'{key}_metrics.json', result)
        return result

    # ---------- fases ----------
    def phase_signature(self, context, phase):
        root = context['root']
        versions = {name: importlib.metadata.version(name)
                    for name in ['torch', 'numpy', 'pandas', 'Pillow', 'scikit-learn', 'psutil']}
        identity = {'phase': phase, 'manifest_sha256': context['manifest_sha256'],
                    'sources': {name: sha256(root / name) for name in self.sources},
                    'versions': versions, 'platform': platform.platform(), 'processor': platform.processor()}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        return f'{phase}_{digest}', identity

    @staticmethod
    def _same_design(summary, context, phase):
        config = context['config']
        return (summary.get('status') == 'completed' and summary.get('phase') == phase
                and summary.get('identity', {}).get('manifest_sha256') == context['manifest_sha256']
                and all(summary['config'].get(key) == config.get(key) for key in ['architecture', 'training'])
                and summary['config']['phases'][phase]['variants'] == config['phases'][phase]['variants'])

    def load_saved_phase(self, context, phase):
        """Cargar una fase terminada y compatible sin entrenar; None si no existe."""
        runs = Path(context['root']) / self.results / 'runs'
        if not runs.exists():
            return None
        for directory in sorted(runs.iterdir(), reverse=True):
            path = directory / 'summary.json'
            if not (directory.is_dir() and path.exists()):
                continue
            summary = json.loads(path.read_text(encoding='utf-8'))
            if not self._same_design(summary, context, phase):
                continue
            for name, digest in summary['artifacts_sha256'].items():
                if sha256(directory / name) != digest:
                    raise ValueError(f'Resultado guardado modificado: {name}')
            print(f"Fase {phase} cargada ({summary['experiment_id']}); no se entrena ninguna red.", flush=True)
            return summary, directory
        return None

    def run_phase(self, context, tensors, normalization, phase):
        """Entrenar una fase completa (todas sus variantes × tres semillas), recuperando lo ya terminado."""
        config = context['config']
        required = config['phases'][phase].get('requires')
        if required and self.load_saved_phase(context, required) is None:
            raise RuntimeError(f'Primero hay que terminar y guardar la fase {required}')
        torch.set_num_threads(config['training']['threads'])
        torch.use_deterministic_algorithms(True)
        signature, identity = self.phase_signature(context, phase)
        directory = Path(context['root']) / self.results / 'runs' / signature
        directory.mkdir(parents=True, exist_ok=True)
        weights_dir = Path(context['root']) / self.weights / signature
        if (directory / 'summary.json').exists():
            saved = self.load_saved_phase(context, phase)
            if saved is not None:
                return saved
        feasibility = self.measure_phase_cost(context, tensors, directory, phase)
        write_json(directory / 'design.json', {'identity': identity, 'config': config, 'normalization': normalization,
                                               'dataset_run': context['run'].name, 'test_images_opened': 0})
        results = []
        for variant in config['phases'][phase]['variants']:
            for seed in config['training']['seeds']:
                key = f"{variant['name']}_seed{seed}"
                metrics = directory / f'{key}_metrics.json'
                if metrics.exists():
                    result = json.loads(metrics.read_text(encoding='utf-8'))
                    if sha256(Path(context['root']) / result['checkpoint']) != result['checkpoint_sha256']:
                        raise ValueError('Checkpoint parcial modificado')
                    print('Recuperada ejecución completa: ' + key, flush=True)
                else:
                    result = self.fit_one(context, tensors, normalization, variant, seed, directory, weights_dir)
                results.append(result)
        pd.DataFrame(results).to_csv(directory / 'all_runs.csv', index=False)
        files = sorted(p for p in directory.iterdir() if p.is_file() and p.name != 'summary.json')
        summary = {'status': 'completed', 'phase': phase, 'experiment_id': signature,
                   'dataset_run': context['run'].name, 'config': config, 'identity': identity,
                   'normalization': normalization,
                   'hardware': {'device': 'cpu', 'processor': platform.processor(),
                                'physical_cores': psutil.cpu_count(logical=False),
                                'threads_used': torch.get_num_threads(),
                                'ram_total_bytes': psutil.virtual_memory().total},
                   'test_images_opened': 0, 'test_predictions_generated': 0, 'cost_feasibility': feasibility,
                   'cost_notes': {'train_seconds': 'all epochs, including train and validation evaluation; '
                                                   'excludes resize',
                                  'FLOPs': '2 per multiply-add in convolutions, recurrent cells and linear layers; '
                                           'excludes bias, activations, pooling, normalization and softmax'},
                   'runs': results, 'artifacts_sha256': {p.name: sha256(p) for p in files}}
        write_json(directory / 'summary.json', summary)
        return summary, directory
