import unittest
from metrics.brier import brier_score

class TestCustomMetric(unittest.TestCase):
    def test_custom_metric(self):
        y_true = [1, 2, 3]
        y_pred = [1, 2, 3]
        score = brier_score(y_true, y_pred)
        self.assertEqual(score, 0)  # Example assertion (adapt as needed)