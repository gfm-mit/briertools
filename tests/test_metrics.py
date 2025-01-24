import unittest
from metrics.brier import brier_score
from metrics.logloss import log_loss

class TestCustomMetric(unittest.TestCase):
    def test_brier_score(self):
        y_true = [0, 1, 1]
        y_pred = [0.1, 0.5, 0.9]
        score = brier_score(y_true, y_pred)
        self.assertAlmostEqual(score, 0.09)

    def test_brier_score_thresholded(self):
        y_true = [0, 1, 1]
        y_pred = [0.1, 0.5, 0.9]
        score = brier_score(y_true, y_pred, threshold_range=(0.2, 0.8))
        self.assertAlmostEqual(score, 0.09 -  0.04)  # Example assertion (adapt as needed)

    def test_log_loss(self):
        y_true = [0, 1, 1]
        y_pred = [0.1, 0.5, 0.9]
        score = log_loss(y_true, y_pred)
        self.assertAlmostEqual(score, -0.30128940395)

    def test_log_loss_thresholded(self):
        y_true = [0, 1, 1]
        y_pred = [0.1, 0.5, 0.9]
        score = log_loss(y_true, y_pred, threshold_range=(0.2, 0.8))
        self.assertAlmostEqual(score, -0.30128940395 - -0.22314355131)  # Example assertion (adapt as needed)