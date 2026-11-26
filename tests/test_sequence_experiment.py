"""Controles de 06: la imagen se lee bien como secuencia y el ciclo entrenar → guardar → cargar funciona."""
import copy
import json
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd
import torch

from src.models.sequence import SequenceClassifier, image_to_sequence, multiply_adds
from src.models.dense import parameter_count
from src.training import sequence_experiment

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / sequence_experiment.CONFIG).read_text(encoding='utf-8'))


class SequenceModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_columns_and_rows_are_read_in_order(self):
        images = torch.arange(2 * 3 * 64 * 64, dtype=torch.float32).reshape(2, 3, 64, 64)
        columns = image_to_sequence(images, 'columns')
        rows = image_to_sequence(images, 'rows')
        self.assertEqual(tuple(columns.shape), (2, 64, 192))
        # Paso 5 por columnas = columna 5 de los tres canales; paso 5 por filas = fila 5.
        torch.testing.assert_close(columns[1, 5], images[1, :, :, 5].reshape(-1))
        torch.testing.assert_close(rows[1, 5], images[1, :, 5, :].reshape(-1))

    def test_every_variant_has_expected_shapes_and_hidden_size(self):
        config = copy.deepcopy(CONFIG)
        config['architecture']['classes'] = 25
        for phase in config['phases'].values():
            for variant in phase['variants']:
                model = sequence_experiment.model_from_config(config, variant)
                self.assertEqual(model.recurrent.hidden_size, 128)
                self.assertEqual(tuple(model(torch.zeros(2, 3, 64, 64)).shape), (2, 25))

    def test_attention_weights_sum_to_one(self):
        model = SequenceClassifier('lstm', readout='attention')
        weights = model.attention_weights(model.states(torch.rand(3, 3, 64, 64)))
        torch.testing.assert_close(weights.sum(1), torch.ones(3))

    def test_multiply_adds_formula(self):
        model = SequenceClassifier('gru', input_size=192, hidden_size=128, classes=25)
        expected = 3 * (192 * 128 + 128 * 128) * 64 + 128 * 25
        self.assertEqual(multiply_adds(model, 64), expected)
        self.assertLess(parameter_count(model), parameter_count(SequenceClassifier('lstm')))


class SequenceExperimentTests(unittest.TestCase):
    def test_train_save_reload_and_phase_dependency(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = {split: pd.DataFrame({'article_id': [f'{split}{i}' for i in range(8)],
                                         'group_id': [f'{split}g{i}' for i in range(8)],
                                         'label_id': [0, 1] * 4, 'label': ['a', 'b'] * 4})
                    for split in ['train', 'validation']}
            tensors = {split: torch.cat([torch.zeros(4, 3, 8, 8), torch.ones(4, 3, 8, 8)])[[0, 4, 1, 5, 2, 6, 3, 7]]
                       for split in rows}
            config = copy.deepcopy(CONFIG)
            config['architecture'].update({'image_size': 8, 'steps': 8, 'features_per_step': 24,
                                           'hidden_size': 6, 'classes': 2})
            config['training'].update({'seeds': [42], 'max_epochs': 2, 'batch_size': 4, 'threads': 1})
            config['bootstrap']['draws'] = 10
            context = {'root': root, 'rows': rows, 'names': ['a', 'b'], 'config': config,
                       'manifest_sha256': 'x' * 64, 'run': types.SimpleNamespace(name='fake')}
            experiment = sequence_experiment.EXPERIMENT
            sources, experiment.sources = experiment.sources, []
            try:
                with self.assertRaises(RuntimeError):          # atención exige la fase de celdas
                    experiment.run_phase(context, tensors, {}, 'attention')
                summary, folder = experiment.run_phase(context, tensors, {'mean': [0] * 3}, 'cells')
                self.assertEqual([r['variant'] for r in summary['runs']], ['rnn', 'lstm', 'gru'])
                self.assertEqual(summary['test_images_opened'], 0)
                self.assertEqual(experiment.load_saved_phase(context, 'cells')[1], folder)
                summary, _ = experiment.run_phase(context, tensors, {'mean': [0] * 3}, 'attention')
                self.assertEqual(summary['runs'][0]['readout'], 'attention')
                self.assertIsNone(experiment.load_saved_phase(context, 'rows'))
            finally:
                experiment.sources = sources


if __name__ == '__main__':
    unittest.main()
