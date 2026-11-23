"""Controles de entrenamiento que pueden cambiar la interpretación del experimento."""
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.dense import DenseClassifier, parameter_count
from src.training.dense_experiment import SelectionState, evaluate, grouped_intervals, train_epoch


class DenseExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shapes_parameter_budget_and_cross_entropy(self):
        model = DenseClassifier()
        self.assertEqual(parameter_count(model), 986108)
        logits = model(torch.zeros(4, 3, 64, 64))
        self.assertEqual(tuple(logits.shape), (4, 12))
        loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1, 2, 3]))
        expected = -logits.log_softmax(1)[range(4), [0, 1, 2, 3]].mean()
        torch.testing.assert_close(loss, expected)

    def test_stopping_loss_is_separate_from_checkpoint_f1(self):
        state = SelectionState(patience=2)
        self.assertEqual(state.update(1, 1.0, .2), (True, False))
        self.assertEqual(state.update(2, 1.1, .3), (True, False))
        self.assertEqual(state.update(3, 1.2, .3), (False, True))
        self.assertEqual(state.best_epoch, 2)
        with self.assertRaises(ValueError):
            state.update(4, float('nan'), .5)

    def test_gradient_learns_tiny_problem_and_eval_disables_dropout(self):
        torch.manual_seed(42)
        model = DenseClassifier(image_size=4, hidden_dims=(12,), classes=2, dropout=.3)
        images = torch.cat([torch.zeros(8, 3, 4, 4), torch.ones(8, 3, 4, 4)])
        labels = torch.tensor([0]*8 + [1]*8)
        loader = DataLoader(TensorDataset(images, labels), batch_size=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=.03)
        before, _ = evaluate(model, images, labels)
        for _ in range(30):
            train_epoch(model, loader, optimizer, nn.CrossEntropyLoss())
        after, logits1 = evaluate(model, images, labels)
        _, logits2 = evaluate(model, images, labels)
        self.assertLess(after['loss'], before['loss']/3)
        self.assertEqual(after['accuracy'], 1.0)
        self.assertTrue(torch.equal(logits1, logits2))

    def test_paired_group_bootstrap_identical_predictions_is_zero(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        pred = np.array([0, 1, 1, 1, 0, 0])
        groups = ['a', 'a', 'b', 'b', 'c', 'd']
        ci = grouped_intervals(y, pred, groups, 2, draws=50, other=pred)
        np.testing.assert_array_equal(ci, [[0, 0], [0, 0]])


if __name__ == '__main__':
    unittest.main()
