"""Controles de la CNN de 05: tamaño comparable, formas y ciclo entrenar → guardar → cargar."""
import copy
import json
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd
import torch

from src.models.cnn import CNNClassifier, multiply_adds
from src.models.dense import DenseClassifier, parameter_count
from src.training import cnn_experiment

ROOT = Path(__file__).resolve().parents[1]


class CNNModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = json.loads((ROOT / cnn_experiment.CONFIG).read_text(encoding='utf-8'))
        cls.config['architecture']['classes'] = 25

    def test_parameters_within_ten_percent_of_dense(self):
        dense = parameter_count(DenseClassifier(classes=25))
        for variant in [{'batchnorm': False}, {'batchnorm': True}]:
            cnn = parameter_count(cnn_experiment.model_from_config(self.config, variant))
            self.assertLessEqual(abs(cnn / dense - 1), self.config['parameter_matching']['tolerance'])

    def test_shapes_and_feature_maps(self):
        model = cnn_experiment.model_from_config(self.config, {'batchnorm': True})
        model.eval()
        self.assertEqual(tuple(model(torch.zeros(2, 3, 64, 64)).shape), (2, 25))
        sizes = [tuple(m.shape[1:]) for m in model.feature_maps(torch.zeros(1, 3, 64, 64))]
        self.assertEqual(sizes, [(16, 32, 32), (32, 16, 16), (64, 8, 8), (128, 4, 4)])

    def test_multiply_adds_counts_convolution_and_linear(self):
        model = CNNClassifier(image_size=4, conv_channels=(2,), hidden_dims=(), classes=3)
        # conv 3→2, 3×3, salida 4×4: 2·3·9·16 = 864; lineal 2·2·2 → 3: 24.
        self.assertEqual(multiply_adds(model, image_size=4), 864 + 24)


class CNNExperimentTests(unittest.TestCase):
    def test_train_save_and_reload_phase_without_training(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ['a', 'b']
            rows = {split: pd.DataFrame({'article_id': [f'{split}{i}' for i in range(8)],
                                         'group_id': [f'{split}g{i}' for i in range(8)],
                                         'label_id': [0, 1] * 4, 'label': ['a', 'b'] * 4})
                    for split in ['train', 'validation']}
            tensors = {split: torch.cat([torch.zeros(4, 3, 8, 8), torch.ones(4, 3, 8, 8)])[[0, 4, 1, 5, 2, 6, 3, 7]]
                       for split in rows}
            config = copy.deepcopy(CNNModelTests.config if hasattr(CNNModelTests, 'config') else
                                   json.loads((ROOT / cnn_experiment.CONFIG).read_text(encoding='utf-8')))
            config['architecture'].update({'image_size': 8, 'conv_channels': [4], 'hidden_dims': [8], 'classes': 2})
            config['training'].update({'seeds': [42], 'max_epochs': 2, 'patience': 5, 'batch_size': 4, 'threads': 1})
            config['bootstrap']['draws'] = 10
            context = {'root': root, 'rows': rows, 'names': names, 'config': config,
                       'manifest_sha256': 'x' * 64, 'run': types.SimpleNamespace(name='fake')}
            sources = cnn_experiment.SOURCES
            cnn_experiment.SOURCES = []  # el proyecto real no está en la carpeta temporal
            try:
                summary, folder = cnn_experiment.run_phase(context, tensors, {'mean': [0] * 3}, 'h1')
                self.assertEqual(summary['test_images_opened'], 0)
                self.assertEqual(len(summary['runs']), 1)
                loaded, loaded_folder = cnn_experiment.load_saved_phase(context, 'h1')
                self.assertEqual(loaded_folder, folder)
                self.assertIsNone(cnn_experiment.load_saved_phase(context, 'batchnorm'))
                # Cambiar el diseño impide reutilizar resultados guardados.
                context['config']['training']['learning_rate'] = .01
                self.assertIsNone(cnn_experiment.load_saved_phase(context, 'h1'))
            finally:
                cnn_experiment.SOURCES = sources

    def test_batchnorm_requires_saved_h1(self):
        with tempfile.TemporaryDirectory() as directory:
            context = {'root': Path(directory), 'config': json.loads((ROOT / cnn_experiment.CONFIG).read_text()),
                       'manifest_sha256': 'x' * 64}
            with self.assertRaises(RuntimeError):
                cnn_experiment.run_phase(context, {}, {}, 'batchnorm')


if __name__ == '__main__':
    unittest.main()
