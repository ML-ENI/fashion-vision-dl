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



if __name__ == '__main__':
    unittest.main()
