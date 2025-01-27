import unittest

import numpy as np
from ..briertools.brier import brier_score
from ..briertools.logloss import log_loss

class TestCustomMetric(unittest.TestCase):
    def test_brier_score(self):
        y_true = [0, 1, 1]
        y_pred = [0.1, 0.5, 0.9]
        score = brier_score(y_true, y_pred)
        self.assertAlmostEqual(score, 0.09)

    def test_brier_score_thresholded(self):
        y_true = [0, 1, 1]
        y_pred = [0.1, 0.5, 0.9]
        threshold_range = (0.2, 0.8)

        y_thresh = [0.2, 0.8, 0.8]
        thresh_score = brier_score(y_true, y_thresh)
        self.assertAlmostEqual(thresh_score, 0.04)
        y_clip = np.clip(y_pred, threshold_range[0], threshold_range[1])

        clip_score = brier_score(y_true, y_clip)
        self.assertAlmostEqual(clip_score, 0.11)

        score = brier_score(y_true, y_pred, threshold_range=(0.2, 0.8))
        self.assertAlmostEqual(score, 0.11 - 0.04)

    def test_log_loss(self):
        y_true = [0, 1, 1]
        y_pred = [0.1, 0.5, 0.9]
        score = log_loss(y_true, y_pred)
        self.assertAlmostEqual(score, 0.30128940395)

    def qtest_log_loss_thresholded(self):
        y_true = [0, 1, 1]
        y_pred = [0.1, 0.5, 0.9]
        score = log_loss(y_true, y_pred, threshold_range=(0.2, 0.8))
        self.assertAlmostEqual(score, 0.30128940395 - 0.22314355131)  # Example assertion (adapt as needed)