import unittest
import scipy

import numpy as np

from ..briertools.utils import get_regret
from ..briertools.brier import brier_score
from ..briertools.logloss import log_loss

class TestBrier(unittest.TestCase):
    def setUp(self):
        self.y_true = [0, 1, 1]
        self.y_pred = [0.1, 0.5, 0.9]
        self.threshold_range = (0.2, 0.8)

    def test_pointwise(self):
        score = brier_score(self.y_true, self.y_pred)
        self.assertAlmostEqual(score, 0.09)
    
    def test_uniform_regret(self):
        taus = np.linspace(0, 1, 10000)
        regret = get_regret(self.y_true, self.y_pred, taus)
        regret = np.trapezoid(regret, taus)
        self.assertAlmostEqual(regret * 2, 0.09, places=4)

    def test_thresholded_pointwise(self):
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
    def setUp(self):
        self.y_true = [0, 1, 1]
        self.y_pred = [0.1, 0.5, 0.9]
        self.threshold_range = (0.2, 0.8)

    def test_pointwise(self):
        score = log_loss(self.y_true, self.y_pred)
        expected = -np.sum(np.log([0.9, 0.5, 0.9])) / 3
        self.assertAlmostEqual(score, expected)
    
    def test_uniform_regret(self):
        taus = scipy.special.expit(np.linspace(-13, 13, 10000))
        weights = np.ones_like(taus) / taus / (1 - taus)
        regret = get_regret(self.y_true, self.y_pred, taus)
        regret = np.trapezoid(regret * weights, taus)
        expected = -np.sum(np.log([0.9, 0.5, 0.9])) / 3
        self.assertAlmostEqual(regret, expected, places=4)

    def test_thresholded_pointwise(self):
        y_thresh = [0.2, 0.8, 0.8]
        thresh_score = log_loss(self.y_true, y_thresh)
        expected_thresh = -np.sum(np.log([0.8, 0.8, 0.8])) / 3
        self.assertAlmostEqual(thresh_score, expected_thresh)

        y_clip = np.clip(self.y_pred, self.threshold_range[0], self.threshold_range[1])
        clip_score = log_loss(self.y_true, y_clip)
        expected_clip = -np.sum(np.log([0.8, 0.5, 0.8])) / 3
        self.assertAlmostEqual(clip_score, expected_clip)

        score = log_loss(self.y_true, self.y_pred, threshold_range=self.threshold_range)
        self.assertAlmostEqual(score, expected_clip - expected_thresh)

    def test_thresholded_regret(self):
        taus = scipy.special.expit(np.linspace(*scipy.special.logit(self.threshold_range), 10000))

        weights = np.ones_like(taus) / taus / (1 - taus)
        regret = get_regret(self.y_true, self.y_pred, taus)
        regret = np.trapezoid(regret * weights, taus)
        expected_thresh = -np.sum(np.log([0.8, 0.8, 0.8])) / 3
        expected_clip = -np.sum(np.log([0.8, 0.5, 0.8])) / 3
        self.assertAlmostEqual(regret, expected_clip - expected_thresh, places=4)