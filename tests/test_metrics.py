import unittest
from metrics.brier import brier_score
from metrics.logloss import log_loss

class TestCustomMetric(unittest.TestCase):
    def test_brier_score(self):
        y_true = [1, 2, 3]
        y_pred = [1, 2, 3]
        score = brier_score(y_true, y_pred)
        self.assertEqual(score, 0)  # Example assertion (adapt as needed)

    def test_log_loss(self):
        y_true = [1, 2, 3]
        y_pred = [1, 2, 3]
        score = log_loss(y_true, y_pred)
        self.assertEqual(score, 0)  # Example assertion (adapt as needed)