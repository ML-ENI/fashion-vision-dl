"""Pruebas del protocolo automático sin descargar CLIP ni ejecutar inferencia real."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.data.automatic_visual_filter import (
    apply_automatic_decisions, decisions_from_scores, score_dataset_images,
)
from src.data.data_contract import sha256


class AutomaticVisualFilterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / 'data/processed'
        self.output.mkdir(parents=True)
        self.policy = {'enabled': True, 'model_id': 'test-only', 'revision': 'a' * 40,
                       'device': 'cpu', 'cpu_threads': 1, 'batch_size': 2,
                       'decision_rule': 'unanimous_detail',
                       'splits': ['train', 'validation', 'test'],
                       'prompt_pairs': [['overview', 'detail']] * 3}
        rows = []
        for i in range(12):
            path = self.output / f'{i:010d}.jpg'
            Image.new('RGB', (224, 224), (i * 20, 30, 40)).save(path)
            rows.append({'article_id': f'{i:010d}', 'label': 'A' if i % 2 == 0 else 'B',
                         'file_sha256': sha256(path), 'prepared_sha256': sha256(path),
                         'image_path': path.name, 'split': ['train', 'validation', 'test'][i // 4]})
        self.candidate = pd.DataFrame(rows)

    def tearDown(self):
        self.temporary.cleanup()

    def test_unanimity_disagreements_ties_and_invalid_scores(self):
        decision, votes, _ = decisions_from_scores([
            [.1, .3, .1, .3, .1, .3],  # detalle en los tres pares
            [.3, .1, .3, .1, .3, .1],  # vista en los tres pares
            [.1, .3, .1, .3, .3, .1],  # desacuerdo: conservar
            [.1, .1, .1, .1, .1, .1],  # empate: conservar como dudosa
        ])
        self.assertEqual(decision.tolist(), ['exclude_detail', 'keep', 'uncertain', 'uncertain'])
        self.assertEqual(votes.tolist(), [3, 0, 2, 0])
        with self.assertRaises(ValueError):
            decisions_from_scores([[float('nan')] * 6])

    def test_all_partitions_and_cached_results_do_not_load_a_model(self):
        seen = []
        class FakeScorer:
            def score(self, paths):
                seen.extend(path.name for path in paths)
                return np.tile([.3, .1] * 3, (len(paths), 1))
        scores, report = score_dataset_images(self.candidate, self.output, self.policy, self.root,
                                               scorer_factory=lambda *_: FakeScorer(), versions={'test': '1'})
        self.assertEqual(set(seen), set(self.candidate.image_path))
        self.assertEqual(report['images_scored'], 12)
        self.assertEqual(report['images_scored_by_split'], {'train': 4, 'validation': 4, 'test': 4})
        def forbidden(*_):
            raise AssertionError('No debe volver a cargarse un modelo con caché completa')
        again, report = score_dataset_images(self.candidate, self.output, self.policy, self.root,
                                               scorer_factory=forbidden, versions={'test': '1'})
        self.assertEqual(report['new_inferences_this_call'], 0)
        pd.testing.assert_frame_equal(scores, again, check_dtype=False)

    def test_interruption_resumes_completed_batches(self):
        seen = []
        class InterruptedScorer:
            def score(self, paths):
                if seen:
                    raise RuntimeError('Interrupción simulada')
                seen.extend(path.name for path in paths)
                return np.tile([.3, .1] * 3, (len(paths), 1))
        with self.assertRaisesRegex(RuntimeError, 'Interrupción'):
            score_dataset_images(self.candidate, self.output, self.policy, self.root,
                                  scorer_factory=lambda *_: InterruptedScorer(), versions={'test': '1'})
        resumed = []
        class ResumeScorer:
            def score(self, paths):
                resumed.extend(path.name for path in paths)
                return np.tile([.3, .1] * 3, (len(paths), 1))
        _, report = score_dataset_images(self.candidate, self.output, self.policy, self.root,
                                          scorer_factory=lambda *_: ResumeScorer(), versions={'test': '1'})
        self.assertEqual(report['new_inferences_this_call'], 10)
        self.assertTrue(set(resumed).isdisjoint(seen))

    def test_same_rule_filters_each_partition_and_preserves_assignment(self):
        candidate = self.candidate.copy()
        candidate['assigned_split'] = candidate.split
        candidate['visual_review'] = 'unreviewed'
        candidate['visual_review_reason'] = ''
        candidate.loc[0, 'visual_review'] = 'keep'  # humano conserva aunque CLIP descarte
        scores = candidate[['article_id']].copy()
        scores['auto_visual_decision'] = ['exclude_detail', 'exclude_detail', 'uncertain', 'keep', 'exclude_detail', 'keep', 'keep', 'keep', 'exclude_detail', 'keep', 'keep', 'keep']
        scores['auto_detail_votes'] = [3, 3, 2, 0, 3, 0, 0, 0, 3, 0, 0, 0]
        scores['auto_similarity_margin'] = [.1, .1, .02, -.1, .1, -.1, -.1, -.1, .1, -.1, -.1, -.1]
        result = apply_automatic_decisions(candidate, scores)
        self.assertEqual(result.loc[0, 'split'], 'train')
        self.assertEqual(result.loc[1, 'split'], 'excluded')
        self.assertEqual(result.loc[2, 'split'], 'train')
        self.assertEqual(result.loc[4, 'split'], 'excluded')
        self.assertEqual(result.loc[8, 'split'], 'excluded')
        kept = result.split.ne('excluded')
        self.assertEqual(result.loc[kept, 'split'].tolist(), candidate.loc[kept, 'split'].tolist())
        self.assertEqual(result.assigned_split.tolist(), candidate.split.tolist())
        with self.assertRaisesRegex(ValueError, 'TODAS'):
            apply_automatic_decisions(candidate, scores.iloc[:2])


if __name__ == '__main__':
    unittest.main()
