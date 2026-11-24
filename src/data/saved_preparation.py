"""Leer el análisis ya publicado y calcular selecciones sin ejecutar CLIP.

Este módulo no importa el modelo ni llama al analizador de imágenes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from src.data.data_contract import sha256, publish, write_json

SPLITS = ['train', 'validation', 'test']
CALCULATION_RECIPE = 'saved-scores-selection-v1'


def _verify(run):
    for name, digest in json.loads((run / 'checksums.json').read_text()).items():
        path = (run / name).resolve()
        if not path.is_relative_to(run.resolve()) or sha256(path) != digest:
            raise ValueError(f'Resultado guardado modificado: {name}')


def _read(run, name):
    return pd.read_csv(run / f'{name}.csv.gz', keep_default_na=False,
                       dtype={'article_id': 'string', 'product_code': 'string',
                              'group_id': 'string', 'raw_group_id': 'string'})


def _compatible(requested, saved):
    keys = ['target', 'include_product_groups', 'include_classes', 'min_side',
            'image_size', 'padding_rgb', 'jpeg_quality', 'seed', 'val_size', 'test_size']
    if any(requested.get(key) != saved.get(key) for key in keys):
        return False
    # El dispositivo y batch no cambian las puntuaciones que ya están guardadas.
    left = requested.get('automatic_visual_filter', {})
    right = saved.get('automatic_visual_filter', {})
    return all(left.get(key) == right.get(key) for key in
               ['enabled', 'model_id', 'revision', 'prompt_pairs', 'decision_rule', 'splits'])


def load_saved_analysis(root, config, run=None):
    """Solo lectura: falta/incompatibilidad provoca un aviso, nunca inferencia."""
    root = Path(root)
    choices = []
    if run is not None:
        choices.append(Path(run))
    else:
        for folder in [root / 'data/processed', root / 'results/data_preparation']:
            pointer = folder / 'CURRENT.json'
            if pointer.exists():
                choices.append(folder / json.loads(pointer.read_text())['run_dir'])
    for path in choices:
        if not (path / 'summary.json').exists():
            continue
        summary = json.loads((path / 'summary.json').read_text())
        # Las selecciones derivadas conservan una referencia al análisis completo.
        source_name = summary.get('saved_analysis_source')
        if source_name:
            candidates = [root / 'data/processed/runs' / source_name,
                          root / 'results/data_preparation' / source_name]
            path = next((p for p in candidates if p.is_dir()), path)
            summary = json.loads((path / 'summary.json').read_text())
        saved_config = json.loads((path / 'config.json').read_text())
        if (summary.get('automatic_visual_filter') or {}).get('status') != 'completed':
            continue
        if not _compatible(config, saved_config):
            continue
        _verify(path)
        manifest = _read(path, 'image_manifest')
        scores = _read(path, 'automatic_visual_scores')
        eligible = manifest.loc[manifest.assigned_split.isin(SPLITS)].set_index('article_id')
        if (scores.article_id.duplicated().any() or set(scores.article_id) != set(eligible.index)):
            raise ValueError('Las puntuaciones no cubren todas las fotos del análisis guardado')
        ordered = scores.set_index('article_id').reindex(eligible.index)
        if not (ordered.prepared_sha256.eq(eligible.prepared_sha256).all() and
                ordered.split.eq(eligible.assigned_split).all()):
            raise ValueError('Las puntuaciones pertenecen a otras fotos o particiones')
        print(f'Cargadas {len(scores):,} puntuaciones guardadas. CLIP no se ejecuta.', flush=True)
        return {'root': root, 'source': path, 'config': copy.deepcopy(config),
                'manifest': manifest, 'scores': scores, 'summary': summary}
    raise FileNotFoundError(
        'No hay un análisis completo compatible guardado. NO se ha lanzado CLIP. '
        'Descarga los resultados compartidos en results/data_preparation o activa explícitamente '
        'la celda opcional de análisis. No es necesario repetirlo si ya existe la copia completa.'
    )


def _install_metadata(root, source):
    """Instalar metadatos compartidos; no copiar fotos ni pesos de modelos."""
    output = root / 'data/processed'
    runs = output / 'runs'
    runs.mkdir(parents=True, exist_ok=True)
    target = runs / source.name
    if target.resolve() != source.resolve():
        if target.exists():
            _verify(target)
            if sha256(target / 'checksums.json') != sha256(source / 'checksums.json'):
                raise ValueError('Existe otra publicación con el mismo identificador; no se sobrescribe')
        else:
            with tempfile.TemporaryDirectory(dir=runs, prefix='.import-') as temporary:
                stage = Path(temporary) / source.name
                shutil.copytree(source, stage)
                _verify(stage)
                stage.rename(target)
    pointer = output / '.CURRENT.saved.pending'
    write_json(pointer, {'run_dir': f'runs/{target.name}', 'pipeline_version': '2.0.0', 'preview': False})
    pointer.replace(output / 'CURRENT.json')
    return target


def finalize_saved_analysis(context, excluded_article_ids=()):
    """Aplicar selecciones y calcular estadísticas. No hay ruta de llamada a CLIP."""
    from src.data.prepare_local import read_visual_review, apply_visual_review, channel_statistics, validate_manifest
    from src.data.automatic_visual_filter import decisions_from_scores, apply_automatic_decisions, SCORE_COLUMNS

    root, source, config = context['root'], context['source'], context['config']
    _verify(source)
    source_config = json.loads((source / 'config.json').read_text())
    source_signature = json.loads((source / 'signature.json').read_text())
    manual_path = config.get('training_visual_review_path')
    manual_hash = sha256(root / manual_path) if manual_path else None
    excluded = sorted(set(str(value) for value in excluded_article_ids))
    unchanged = (not excluded and config == source_config and
                 manual_hash == source_signature.get('training_visual_review_sha256'))
    if unchanged:
        print('Selección sin cambios: se reutilizan manifiesto, pesos y estadísticas. No se calculan otra vez.', flush=True)
        return _install_metadata(root, source)

    identity = {'source_checksums': sha256(source / 'checksums.json'), 'config': config,
                'excluded_article_ids': excluded, 'manual_sha256': manual_hash, 'recipe': CALCULATION_RECIPE}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = root / 'data/processed'
    pointer = output / 'CURRENT.json'
    if pointer.exists():
        current = output / json.loads(pointer.read_text())['run_dir']
        if json.loads((current / 'signature.json').read_text()).get('saved_result_key') == key:
            _verify(current)
            return current

    frame = context['manifest'].copy(deep=True)
    candidate = frame.loc[frame.assigned_split.isin(SPLITS)].copy()
    candidate['split'] = candidate.assigned_split
    if not set(excluded).issubset(set(candidate.article_id)):
        raise ValueError('La selección incluye identificadores ajenos al análisis guardado')
    scores = context['scores'].copy(deep=True)
    decisions, votes, margins = decisions_from_scores(scores[SCORE_COLUMNS].to_numpy(dtype=float))
    scores['auto_visual_decision'], scores['auto_detail_votes'], scores['auto_similarity_margin'] = decisions, votes, margins
    review = read_visual_review(config, root)
    candidate, visual_report = apply_visual_review(candidate, review)
    candidate = apply_automatic_decisions(candidate, scores)
    for column in ['split', 'visual_review', 'visual_review_reason', 'auto_visual_decision',
                   'auto_detail_votes', 'auto_similarity_margin']:
        frame.loc[candidate.index, column] = candidate[column]
    frame.loc[candidate.index, 'reason'] = ''
    frame.loc[frame.visual_review.eq('exclude_detail'), 'reason'] = 'detail_only_image'
    frame.loc[frame.visual_review.eq('auto_exclude_detail'), 'reason'] = 'automatic_detail_image'
    selection = frame.article_id.isin(excluded) & frame.split.isin(SPLITS)
    frame.loc[selection, 'split'] = 'excluded'
    frame.loc[selection, 'reason'] = 'explicit_metadata_selection'
    labels = json.loads((source / 'labels.json').read_text())
    for split in SPLITS:
        if set(frame.loc[frame.split.eq(split), 'label']) != set(labels):
            raise ValueError(f'La selección deja clases sin ejemplos en {split}; no se publica ni se elimina una clase automáticamente')
    counts = frame.loc[frame.split.eq('train'), 'label'].value_counts()
    weights = (counts.sum() / (len(counts) * counts)).to_dict()
    frame['training_weight'] = 0.0
    frame.loc[frame.split.eq('train'), 'training_weight'] = frame.loc[frame.split.eq('train'), 'label'].map(weights)
    before = context['manifest'].loc[context['manifest'].split.eq('train'), ['article_id', 'prepared_sha256']]
    after = frame.loc[frame.split.eq('train'), ['article_id', 'prepared_sha256']]
    if before.reset_index(drop=True).equals(after.reset_index(drop=True)):
        normalization = json.loads((source / 'normalization.json').read_text())
    else:
        print('Recalculando únicamente las estadísticas del train modificado; sin CLIP.', flush=True)
        normalization = channel_statistics(frame, output)
    selected = frame.loc[frame.split.isin(SPLITS)]
    summary = copy.deepcopy(context['summary'])
    summary.update({'images_usable': len(selected), 'split_counts': selected.split.value_counts().to_dict(),
                    'class_distribution_by_split': pd.crosstab(selected.label, selected.split).to_dict(),
                    'excluded_reasons': frame.loc[frame.split.eq('excluded'), 'reason'].value_counts().to_dict(),
                    'prepared_bytes': int(selected.prepared_bytes.sum()), 'training_visual_review': visual_report,
                    'saved_analysis_source': source.name, 'explicit_metadata_selection': excluded})
    summary['automatic_visual_filter']['new_inferences_this_call'] = 0
    summary['automatic_visual_filter']['automatic_exclusions_applied'] = int(candidate.visual_review.eq('auto_exclude_detail').sum())
    summary['automatic_visual_filter']['human_overrides'] = len(review)
    signature = copy.deepcopy(source_signature)
    signature.update({'config': config, 'saved_result_key': key})
    if manual_path:
        signature['training_visual_review_sha256'] = manual_hash
    provenance = json.loads((source / 'provenance.json').read_text())
    provenance['derived_from_saved_analysis'] = identity
    # La selección usa archivos existentes; esta comprobación no es inferencia.
    validate_manifest(frame, output, verify_assets=True)
    return publish(output, {'image_manifest': frame, 'automatic_visual_scores': scores, 'training_visual_review': review},
                   {'summary': summary, 'labels': labels, 'training_class_weights': weights,
                    'normalization': normalization, 'config': config, 'signature': signature, 'provenance': provenance})
