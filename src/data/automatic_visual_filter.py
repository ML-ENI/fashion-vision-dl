"""Filtro de detalles en train, validation y test con una regla visual común.

Las decisiones del modelo no son anotaciones humanas ni predicciones de prendas.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.data.data_contract import sha256


SCORE_COLUMNS = [f'clip_similarity_{i}' for i in range(6)]


def validate_policy(policy):
    if not policy.get('enabled', False):
        return
    if set(policy.get('splits', [])) != {'train', 'validation', 'test'}:
        raise ValueError('El filtro visual debe incluir train, validation y test')
    pairs = policy.get('prompt_pairs', [])
    if (len(pairs) != 3 or any(len(pair) != 2 or any(not isinstance(p, str) or not p.strip() for p in pair)
                               for pair in pairs)):
        raise ValueError('El filtro automático necesita tres pares de textos: vista reconocible / detalle')
    if policy.get('decision_rule') != 'unanimous_detail':
        raise ValueError('Regla automática desconocida; no cambiarla silenciosamente')
    if (not isinstance(policy.get('batch_size'), int) or policy['batch_size'] < 1 or
            not isinstance(policy.get('cpu_threads'), int) or policy['cpu_threads'] < 1):
        raise ValueError('batch_size y cpu_threads deben ser enteros positivos')
    revision = policy.get('revision', '')
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Fijar una revisión SHA completa del modelo, no main')
    if policy.get('device') not in ['cpu', 'cuda']:
        raise ValueError('Dispositivo del filtro: cpu o cuda explícito')


def dependency_versions():
    try:
        versions = {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'Pillow', 'numpy']}
    except importlib.metadata.PackageNotFoundError as error:
        raise ImportError(
            'Faltan dependencias del filtro automático. En una celda del mismo kernel ejecuta '
            '%pip install -r "../requirements-visual-filter.txt" y reinicia el kernel antes de ejecutar 02.'
        ) from error
    if versions['transformers'] != '4.57.1':
        raise ImportError('Instala requirements-visual-filter.txt en este kernel: la API fijada es Transformers 4.57.1.')
    return versions


def decisions_from_scores(scores):
    """Solo descartar si los tres pares prefieren detalle; conservar desacuerdos."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 6 or not np.isfinite(scores).all():
        raise ValueError('Se esperaban seis similitudes finitas por fotografía')
    differences = scores[:, 1::2] - scores[:, 0::2]
    votes = (differences > 0).sum(axis=1)
    decisions = np.where(votes == 3, 'exclude_detail', np.where((differences < 0).all(axis=1), 'keep', 'uncertain'))
    return decisions, votes, differences.mean(axis=1)


class ClipScorer:
    def __init__(self, policy, root):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = policy['device']
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('Se pidió CUDA, pero este kernel no dispone de CUDA')
        torch.set_num_threads(policy['cpu_threads'])
        common = {'revision': policy['revision'], 'cache_dir': str(root / 'tmp/model_cache'),
                  'local_files_only': policy.get('local_files_only', False)}
        print('Cargando CLIP local; si no está en caché se descargarán sus pesos, no se enviarán fotos.', flush=True)
        self.processor = CLIPProcessor.from_pretrained(policy['model_id'], use_fast=False, **common)
        self.model = CLIPModel.from_pretrained(policy['model_id'], use_safetensors=False, **common).to(self.device).eval()
        from huggingface_hub import try_to_load_from_cache
        weights = try_to_load_from_cache(policy['model_id'], 'pytorch_model.bin', **{
            'revision': policy['revision'], 'cache_dir': common['cache_dir']})
        self.provenance = {'checkpoint_revision': getattr(self.model.config, '_commit_hash', policy['revision']),
                           'weights_sha256': sha256(Path(weights)) if isinstance(weights, str) else None}
        prompts = [text for pair in policy['prompt_pairs'] for text in pair]
        text = self.processor.tokenizer(prompts, padding=True, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            features = self.model.get_text_features(**text)
            self.text_features = features / features.norm(dim=-1, keepdim=True)

    def score(self, paths):
        images = []
        for path in paths:
            with Image.open(path) as source:
                if source.size != (224, 224):
                    raise ValueError('El filtro CLIP espera las copias RGB224 de 02')
                images.append(source.convert('RGB'))
        # No introducir otro recorte central: ya tenemos RGB224 con proporción conservada.
        inputs = self.processor.image_processor(images=images, do_center_crop=False,
                                                 do_resize=False, return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            return (features @ self.text_features.T).cpu().numpy()


def score_dataset_images(candidate, output, policy, root, scorer_factory=None, versions=None):
    """Cada lote completo se guarda atómicamente. Una interrupción no publica datos."""
    validate_policy(policy)
    images = candidate.loc[candidate.split.isin(policy['splits'])].sort_values('article_id').copy()
    versions = dependency_versions() if versions is None else versions
    identity = {'policy': policy, 'versions': versions, 'source_sha256': sha256(Path(__file__)),
                'images': images[['article_id', 'file_sha256', 'prepared_sha256', 'split']].to_dict('records')}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    cache = root / 'tmp/automatic_visual_filter' / fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    factory = ClipScorer if scorer_factory is None else scorer_factory
    scorer = None
    blocks = []
    new_images = 0
    inference_seconds = 0.0
    started = time.perf_counter()
    size = policy['batch_size']
    print(f'Filtro automático: {len(images):,} fotos de train, validation y test; batch={size}; dispositivo={policy["device"]}.', flush=True)
    for offset in range(0, len(images), size):
        part = images.iloc[offset:offset + size]
        # Verificar cada copia antes de usarla o reutilizar una predicción de caché.
        paths = []
        for row in part.itertuples():
            path = (output / row.image_path).resolve()
            if not path.is_relative_to(output.resolve()) or sha256(path) != row.prepared_sha256:
                raise ValueError(f'Imagen preparada modificada: {row.article_id}')
            paths.append(path)
        path = cache / f'batch_{offset // size:06d}.csv'
        checksum_path = path.with_suffix('.json')
        if path.exists() and checksum_path.exists():
            if json.loads(checksum_path.read_text())['sha256'] != sha256(path):
                raise ValueError('Caché automática modificada: checksum del lote incorrecto')
            block = pd.read_csv(path, dtype={'article_id': 'string'}, keep_default_na=False)
            if (block.article_id.tolist() != part.article_id.tolist() or
                    block.prepared_sha256.tolist() != part.prepared_sha256.tolist()):
                raise ValueError('Caché automática incompatible con las fotografías; no reutilizar')
            scores = block[SCORE_COLUMNS].to_numpy(dtype=float)
        else:
            if scorer is None:
                scorer = factory(policy, root)
                model_info = cache / 'model_info.json'
                temporary_info = model_info.with_suffix('.pending')
                temporary_info.write_text(json.dumps(getattr(scorer, 'provenance', {})), encoding='utf-8')
                temporary_info.replace(model_info)
            tick = time.perf_counter()
            scores = np.asarray(scorer.score(paths))
            inference_seconds += time.perf_counter() - tick
            if scores.shape != (len(part), 6):
                raise ValueError('CLIP devolvió un número de imágenes o similitudes inesperado')
            decisions_from_scores(scores)
            block = part[['article_id', 'file_sha256', 'prepared_sha256', 'label', 'split']].reset_index(drop=True)
            for index, column in enumerate(SCORE_COLUMNS):
                block[column] = scores[:, index]
            temporary = path.with_suffix('.pending')
            block.to_csv(temporary, index=False, float_format='%.17g')
            temporary.replace(path)
            temporary = checksum_path.with_suffix('.pending')
            temporary.write_text(json.dumps({'sha256': sha256(path)}), encoding='utf-8')
            temporary.replace(checksum_path)
            new_images += len(part)
        decision, votes, margin = decisions_from_scores(scores)
        block['auto_visual_decision'] = decision
        block['auto_detail_votes'] = votes
        block['auto_similarity_margin'] = margin
        blocks.append(block)
        completed = offset + len(part)
        if offset == 0 or (offset // size + 1) % 10 == 0 or completed == len(images):
            eta = ((len(images) - completed) * inference_seconds / new_images) if new_images else 0
            estimate = f' · estimación restante {eta / 60:.1f} min' if new_images else ' · caché'
            print(f'CLIP: {completed:,}/{len(images):,}{estimate}', flush=True)
    scores = pd.concat(blocks, ignore_index=True)
    report = {'status': 'completed', 'fingerprint': fingerprint, 'model_id': policy['model_id'],
              'revision': policy['revision'], 'prompt_pairs': policy['prompt_pairs'],
              'decision_rule': policy['decision_rule'], 'versions': versions,
              'images_scored': len(scores), 'prediction_counts': scores.auto_visual_decision.value_counts().to_dict(),
              'new_inferences_this_call': new_images, 'elapsed_seconds_this_call': time.perf_counter() - started,
              'images_scored_by_split': scores.split.value_counts().to_dict(),
              'scope': policy['splits'],
              'interpretation': 'Prompt similarities and votes, not calibrated probabilities or verified ground truth'}
    info_path = cache / 'model_info.json'
    report['checkpoint'] = json.loads(info_path.read_text()) if info_path.exists() else {}
    return scores, report


def apply_automatic_decisions(candidate, scores):
    """Respetar decisiones humanas y la partición original de cada foto retenida."""
    result = candidate.copy()
    expected = result.loc[result.assigned_split.isin(['train', 'validation', 'test']), 'article_id']
    if scores.article_id.duplicated().any() or set(scores.article_id) != set(expected):
        raise ValueError('El filtro automático debe cubrir exactamente TODAS las fotos elegibles de las tres particiones')
    automatic = scores.set_index('article_id')
    for column in ['auto_visual_decision', 'auto_detail_votes', 'auto_similarity_margin']:
        result[column] = result.article_id.map(automatic[column])
    unreviewed = result.visual_review.eq('unreviewed')
    rejected = unreviewed & result.auto_visual_decision.eq('exclude_detail')
    result.loc[unreviewed, 'visual_review'] = result.loc[unreviewed, 'auto_visual_decision'].map({
        'keep': 'auto_keep', 'exclude_detail': 'auto_exclude_detail', 'uncertain': 'auto_uncertain'})
    result.loc[unreviewed, 'visual_review_reason'] = 'CLIP zero-shot; similitudes guardadas, no anotación humana'
    result.loc[rejected, 'split'] = 'excluded'
    for split in ['train', 'validation', 'test']:
        if set(result.loc[result.split.eq(split), 'label']) != set(candidate.loc[candidate.assigned_split.eq(split), 'label']):
            raise ValueError(f'El filtro deja clases sin ejemplos en {split}; revisar descartes antes de publicar')
    return result
