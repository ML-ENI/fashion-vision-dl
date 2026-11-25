"""Reutilización con datos ficticios: nunca CLIP ni entrenamiento real."""
import copy
import ast
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/data'))
from src.data.data_contract import publish, sha256
from src.data.saved_preparation import load_saved_analysis, finalize_saved_analysis
from src.training.saved_experiments import load_saved_experiment


class SavedPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        output = self.root / 'data/processed'
        images = output / 'images'
        images.mkdir(parents=True)
        self.config = {'target': 'product_type_name', 'include_classes': ['A'], 'image_size': 2,
                       'automatic_visual_filter': {'enabled': True, 'model_id': 'test-only',
                       'revision': '1' * 40, 'prompt_pairs': [['whole', 'detail']] * 3,
                       'decision_rule': 'unanimous_detail', 'splits': ['train', 'validation', 'test']}}
        rows = []
        for i in range(12):
            path = images / f'{i:010d}.png'
            Image.new('RGB', (2, 2), (i * 10, i * 8, i * 6)).save(path)
            split = ['train', 'validation', 'test'][i // 4]
            rows.append({'article_id': f'{i:010d}', 'product_code': f'{i:07d}',
                         'group_id': f'{i:010d}', 'raw_group_id': f'{i:010d}', 'history_split': '',
                         'label': 'A', 'label_id': 0, 'split': split, 'assigned_split': split,
                         'file_sha256': sha256(path), 'pixel_sha256': f'raw-{i}',
                         'prepared_sha256': sha256(path), 'prepared_pixel_sha256': f'prepared-{i}',
                         'image_path': f'images/{path.name}', 'prepared_bytes': path.stat().st_size,
                         'visual_review': 'auto_keep', 'visual_review_reason': '', 'reason': '',
                         'training_weight': 1.0 if split == 'train' else 0.0})
        frame = pd.DataFrame(rows)
        scores = frame[['article_id', 'file_sha256', 'prepared_sha256', 'label', 'split']].copy()
        for i in range(6):
            scores[f'clip_similarity_{i}'] = .3 if i % 2 == 0 else .1
        scores['auto_visual_decision'] = 'keep'
        scores['auto_detail_votes'] = 0
        scores['auto_similarity_margin'] = -.2
        summary = {'images_usable': 12, 'classes_usable': 1,
                   'automatic_visual_filter': {'status': 'completed', 'images_scored': 12,
                    'new_inferences_this_call': 12}}
        self.run = publish(output, {'image_manifest': frame, 'automatic_visual_scores': scores},
                           {'summary': summary, 'config': self.config, 'signature': {}, 'labels': {'A': 0},
                            'normalization': {'mean': [.1] * 3, 'std': [.1] * 3, 'pixels': 16},
                            'training_class_weights': {'A': 1.0}, 'provenance': {}})

    def tearDown(self):
        self.temp.cleanup()

    def test_repeated_load_does_not_call_model_or_pixel_statistics(self):
        with patch('src.data.automatic_visual_filter.ClipScorer', side_effect=AssertionError('No CLIP')) as model:
            with patch('src.data.prepare_local.channel_statistics', side_effect=AssertionError('No recalcular')) as stats:
                for _ in range(2):
                    saved = load_saved_analysis(self.root, self.config)
                    self.assertEqual(finalize_saved_analysis(saved), self.run)
                model.assert_not_called()
                stats.assert_not_called()

    def test_shared_bundle_loads_without_raw_audit_images_or_model(self):
        friend = self.root / 'friend'
        bundle = friend / 'results/data_preparation' / self.run.name
        shutil.copytree(self.run, bundle)
        (bundle.parent / 'CURRENT.json').write_text(json.dumps({'run_dir': self.run.name}))
        with patch('src.data.automatic_visual_filter.ClipScorer', side_effect=AssertionError('No CLIP')):
            saved = load_saved_analysis(friend, self.config)
            installed = finalize_saved_analysis(saved)
        self.assertTrue((installed / 'automatic_visual_scores.csv.gz').exists())
        self.assertFalse((friend / 'data/raw').exists())
        self.assertFalse((friend / 'data/processed/images').exists())

    def test_missing_or_incompatible_results_stop_instead_of_inference(self):
        changed = copy.deepcopy(self.config)
        changed['image_size'] = 4
        with patch('src.data.automatic_visual_filter.ClipScorer', side_effect=AssertionError('No CLIP')):
            with self.assertRaises(FileNotFoundError):
                load_saved_analysis(self.root, changed)
            with self.assertRaises(FileNotFoundError):
                load_saved_analysis(self.root / 'empty', self.config)

    def test_corrupt_scores_are_not_reused(self):
        with (self.run / 'automatic_visual_scores.csv.gz').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'modificado'):
            load_saved_analysis(self.root, self.config)

    def test_selection_recalculates_once_without_model_and_keeps_source(self):
        from src.data.prepare_local import channel_statistics
        before = sha256(self.run / 'checksums.json')
        saved = load_saved_analysis(self.root, self.config)
        with patch('src.data.automatic_visual_filter.ClipScorer', side_effect=AssertionError('No CLIP')) as model:
            with patch('src.data.prepare_local.channel_statistics', wraps=channel_statistics) as stats:
                first = finalize_saved_analysis(saved, excluded_article_ids=['0000000000'])
                second = finalize_saved_analysis(saved, excluded_article_ids=['0000000000'])
                self.assertEqual(first, second)
                self.assertEqual(stats.call_count, 1)
                model.assert_not_called()
        self.assertEqual(sha256(self.run / 'checksums.json'), before)
        summary = json.loads((first / 'summary.json').read_text())
        self.assertEqual(summary['images_usable'], 11)
        self.assertEqual(summary['automatic_visual_filter']['new_inferences_this_call'], 0)


class SavedExperimentTests(unittest.TestCase):
    def test_metric_changes_reuse_predictions_but_other_dataset_stops(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'results/metrics/04_dense/runs/test-run'
            directory.mkdir(parents=True)
            predictions = directory / 'predictions.csv'
            predictions.write_text('y,prediction\n0,0\n')
            config = {'architecture': {'classes': 2}, 'training': {'seeds': [42]},
                      'variants': ['baseline'], 'bootstrap': {'draws': 300}}
            summary = {'status': 'completed', 'experiment_id': 'test-run', 'config': config,
                       'identity': {'manifest_sha256': 'dataset-a'},
                       'artifacts_sha256': {'predictions.csv': hashlib.sha256(predictions.read_bytes()).hexdigest()}}
            (directory / 'summary.json').write_text(json.dumps(summary))
            changed = copy.deepcopy(config)
            changed['bootstrap']['draws'] = 500
            context = {'root': root, 'manifest_sha256': 'dataset-a', 'config': changed}
            self.assertEqual(load_saved_experiment(context)[1], directory)
            context['manifest_sha256'] = 'dataset-b'
            with self.assertRaises(FileNotFoundError):
                load_saved_experiment(context)

    def test_normalized_arrays_are_not_rewritten_on_repeat(self):
        from src.training.dense_experiment import normalized_tensors
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'train.npy'
            np.save(path, np.arange(96, dtype=np.uint8).reshape(2, 3, 4, 4))
            raw = {'train': np.load(path, mmap_mode='r')}
            normalization = {'mean': [.1, .2, .3], 'std': [.2, .2, .2]}
            tensors = normalized_tensors(raw, normalization)
            expected = tensors['train'].clone()
            del tensors
            original = np.lib.format.open_memmap
            def read_only(*args, **kwargs):
                if kwargs.get('mode', 'r+') not in ['r', 'c']:
                    raise AssertionError('No reescribir')
                return original(*args, **kwargs)
            with patch('numpy.lib.format.open_memmap', side_effect=read_only):
                tensors = normalized_tensors(raw, normalization)
            self.assertTrue(np.array_equal(tensors['train'].numpy(), expected.numpy()))
            del tensors, expected, raw


class NotebookCostGuardsTests(unittest.TestCase):
    def test_syntax_and_expensive_calls_are_explicitly_guarded(self):
        root = Path(__file__).resolve().parents[1]
        for filename, flag, costly in [('02_data_preparation.ipynb', 'CREAR_ANALISIS_CLIP', 'prepare'),
                                       ('04_dense_mlp.ipynb', 'ENTRENAR_SI_FALTA', 'run_or_load')]:
            notebook = json.loads((root / 'notebooks' / filename).read_text(encoding='utf-8'))
            found_default = False
            found_call = False
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code':
                    continue
                source = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
                tree = ast.parse(source)
                parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == flag for t in node.targets):
                        self.assertIsInstance(node.value, ast.Constant)
                        self.assertIs(node.value.value, False)
                        found_default = True
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == costly:
                        found_call = True
                        ancestor = parents.get(node)
                        guarded = False
                        while ancestor is not None:
                            if isinstance(ancestor, ast.If) and isinstance(ancestor.test, ast.Name) and ancestor.test.id == flag:
                                guarded = True
                            ancestor = parents.get(ancestor)
                        self.assertTrue(guarded, f'{costly} sin bandera explícita en {filename}')
            self.assertTrue(found_default)
            self.assertTrue(found_call)


if __name__ == '__main__':
    unittest.main()
