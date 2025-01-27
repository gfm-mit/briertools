import unittest

import numpy as np

from ..briertools.utils import get_regret
from ..briertools.brier import brier_score
from ..briertools.logloss import log_loss

class TestBrier(unittest.TestCase):
    def setUp(self):
        self.y_true = [0, 1, 1]
        self.y_pred = [0.1, 0.5, 0.9]
        self.threshold_range = (0.2, 0.8)

    def test_brier_score(self):
        score = brier_score(self.y_true, self.y_pred)
        self.assertAlmostEqual(score, 0.09)
    
    def test_uniform_regret(self):
        taus = np.linspace(0, 1, 10000)
        regret = get_regret(self.y_true, self.y_pred, taus)
        regret = np.trapezoid(regret, taus)
        self.assertAlmostEqual(regret * 2, 0.09, places=4)

    def test_thresholded_brier_score(self):
        y_thresh = [0.2, 0.8, 0.8]
        thresh_score = brier_score(self.y_true, y_thresh)
        self.assertAlmostEqual(thresh_score, 0.04)

        y_clip = np.clip(self.y_pred, self.threshold_range[0], self.threshold_range[1])
        clip_score = brier_score(self.y_true, y_clip)
        self.assertAlmostEqual(clip_score, 0.11)

        score = brier_score(self.y_true, self.y_pred, threshold_range=self.threshold_range)
        self.assertAlmostEqual(score, 0.11 - 0.04)

    def test_thresholded_regret(self):
        taus = np.linspace(*self.threshold_range, 10000)

        regret = get_regret(self.y_true, self.y_pred, taus)
        regret = np.trapezoid(regret, taus)
        self.assertAlmostEqual(regret * 2, 0.11 - 0.04, places=4)

class TestLogLoss(unittest.TestCase):
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