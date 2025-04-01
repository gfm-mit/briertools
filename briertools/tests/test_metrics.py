import unittest

import matplotcheck.base as mpc
import matplotcheck.notebook as nb
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
import scipy

from scorers import DCAScorer, LogLossScorer, BrierScorer, MetricScorer


class TestMetricScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = MetricScorer()
        self.y_true = [0, 1, 1]
        self.y_pred = [0.1, 0.5, 0.8]
        self.pointwise_l1_loss = np.array([0.1, 0.5, 0.2])
        self.threshold_range = (0.2, 0.8)

    def test_assert_valid(self):
        y_bad = [-1, 2, 3, 0, 0, 0]
        y_good_int = [0, 1, 0, 1, 0, 1]
        y_good_float = [1.0, 0.0, 0.5, 0.3, 0.2, 0.9]

        with self.assertRaises(AssertionError):
            self.scorer._assert_valid(y_bad, y_good_int)
        with self.assertRaises(AssertionError):
            self.scorer._assert_valid(y_good_float, y_bad)

        self.assertEqual(self.scorer._assert_valid(y_good_float, y_good_int), None)

    def test_pointwise_l1_loss(self):
        loss_scorer = self.scorer._pointwise_l1_loss(
            self.y_true,
            self.y_pred,
        )
        assert_almost_equal(
            loss_scorer.tolist(),
            self.pointwise_l1_loss.tolist(),
        )

    def test_l1_to_total_log_loss(self):
        self.assertAlmostEqual(
            self.scorer._l1_to_total_log_loss(self.pointwise_l1_loss),
            np.mean(-np.log(np.ones(len(self.pointwise_l1_loss)) - self.pointwise_l1_loss))
        )

    def test_l1_to_total_l2_loss(self):
        self.assertAlmostEqual(
            self.scorer._l1_to_total_l2_loss(self.pointwise_l1_loss), 
            np.mean(np.array([i ** 2 for i in self.pointwise_l1_loss])))

    def test_clip_loss(self):
        score_clip, score_near = self.scorer._clip_loss(
            self.y_true, 
            self.y_pred, 
            self.threshold_range,
        )
        assert_almost_equal(
            score_clip,
            np.array([0.2, 0.2, 0.2])
        )
        assert_almost_equal(
            score_near,
            np.array([0.2, 0.5, 0.2])
        )

    def test_uniform_regret(self):
        taus = np.linspace(0, 1, 10000)
        regret = self.scorer._get_regret(self.y_true, self.y_pred, taus)
        regret = np.trapezoid(regret, taus)
        self.assertAlmostEqual(regret * 2, 0.1, places=4)

        taus = scipy.special.expit(np.linspace(-13, 13, 10000))
        weights = np.ones_like(taus) / taus / (1 - taus)
        regret = self.scorer._get_regret(self.y_true, self.y_pred, taus)
        regret = np.trapezoid(regret * weights, taus)
        expected = -np.sum(np.log([0.8, 0.5, 0.9])) / 3
        self.assertAlmostEqual(regret, expected, places=4)

    def test_thresholded_regret(self):
        taus = np.linspace(*self.threshold_range, 10000)
        regret = self.scorer._get_regret(self.y_true, self.y_pred, taus)
        regret = np.trapezoid(regret, taus)
        self.assertAlmostEqual(regret * 2, 0.11 - 0.04, places=4)

        taus = scipy.special.expit(
            np.linspace(*scipy.special.logit(self.threshold_range), 10000)
        )

        weights = np.ones_like(taus) / taus / (1 - taus)
        regret = self.scorer._get_regret(self.y_true, self.y_pred, taus)
        regret = np.trapezoid(regret * weights, taus)
        expected_thresh = -np.sum(np.log([0.8, 0.8, 0.8])) / 3
        expected_clip = -np.sum(np.log([0.8, 0.5, 0.8])) / 3
        self.assertAlmostEqual(regret, expected_clip - expected_thresh, places=4)

class TestBrier(unittest.TestCase):
    
    def setUp(self):
        self.y_true = [0, 1, 1]
        self.y_pred = [0.1, 0.5, 0.8]
        self.threshold_range = (0.2, 0.8)
        self.scorer = BrierScorer()
        self.scorer.n_points = 5
        self.x_correct = np.array([0.2,  0.35, 0.5,  0.65, 0.8 ])
        self.y_correct = np.array([0, 0, 0, 0.35, 0.2]) / 3

    def test_pointwise(self):
        score = self.scorer.score(self.y_true, self.y_pred)
        self.assertAlmostEqual(score, 0.1)

    def test_thresholded_pointwise(self):
        y_thresh = [0.2, 0.8, 0.8]
        thresh_score = self.scorer.score(self.y_true, y_thresh)
        self.assertAlmostEqual(thresh_score, 0.04)

        y_clip = np.clip(self.y_pred, self.threshold_range[0], self.threshold_range[1])
        clip_score = self.scorer.score(self.y_true, y_clip)
        self.assertAlmostEqual(clip_score, 0.11)

        score = self.scorer.score(
            self.y_true, self.y_pred, threshold_range=self.threshold_range
        )
        self.assertAlmostEqual(score, 0.11 - 0.04)

    def test_partition_loss(self):
        calibration_loss, discrimination_loss = self.scorer._partition_loss(
            self.y_true,
            self.y_pred,
            self.scorer.score,
            self.threshold_range,

        )
        self.assertAlmostEqual(calibration_loss, 0.07)
        self.assertAlmostEqual(discrimination_loss, 0.0)

    def test_make_x_and_y_curves(self):
        x_to_plot, y_to_plot, label = self.scorer._make_x_and_y_curves(
            self.y_true,
            self.y_pred,
            self.threshold_range,
            (0.3, 0.7),
        )
        assert_almost_equal(x_to_plot, self.x_correct)
        assert_almost_equal(y_to_plot, self.y_correct)
        label_correct = "MSE: 0.05 | $\mathbb{E}$ R(f): 0.04"
        self.assertEqual(label, label_correct)
        
    def test_plot_curve_and_get_colors(self):
        fig, ax = plt.subplots()
        self.scorer._plot_curve_and_get_colors(
            ax,
            self.x_correct,
            self.y_correct, 
            "this is a test",
        )
        plt.legend()
        plot_tester = mpc.PlotTester(ax)
        legend_given = [[ll.get_text() for ll in l.get_texts()] for l in plot_tester.get_legends()]
        df_data = pd.DataFrame({
            'x': self.x_correct,
            'y': self.y_correct,
        })
        plot_tester.assert_xydata(df_data, xcol='x', ycol='y')
        self.assertEqual(legend_given, [['this is a test']]) 

    def test_plot_curve(self):
        fig, ax = plt.subplots()
        self.scorer.plot_curve(
            ax,
            self.y_true,
            self.y_pred,
            self.threshold_range,
            fill_range=(0.1, 0.9),
            ticks=[i * 0.2 - 0.01 for i in range(1, 5)],
        )
        plt.legend()
        plot_tester = mpc.PlotTester(ax)
        legend_given = [[ll.get_text() for ll in l.get_texts()] for l in plot_tester.get_legends()]
        df_data = pd.DataFrame({
            'x': self.x_correct,
            'y': self.y_correct,
        })
        plot_tester.assert_xydata(df_data, xcol='x', ycol='y')
        label_correct = "MSE: 0.09 | $\mathbb{E}$ R(f): 0.04"
        self.assertEqual(legend_given, [[label_correct]])

    def test_get_fill_between_params(self):
        xs, ylow, yhigh, fill_kwargs = self.scorer._get_fill_between_params(
            self.y_true,
            self.y_pred,
            self.threshold_range,
            0.3,
            fill_range=(0.1, 0.9),
        )
        assert_almost_equal(xs, self.x_correct)
        assert_almost_equal(ylow, self.y_correct)
        assert_almost_equal(yhigh, np.zeros(5))
        self.assertEqual(fill_kwargs, {'alpha': 0.3})

class TestLogLoss(unittest.TestCase):
    def setUp(self):
        self.y_true = [0, 1, 1]
        self.y_pred = [0.1, 0.5, 0.8]
        self.threshold_range = (0.2, 0.8)
        self.scorer = LogLossScorer()
        self.scorer.n_points = 5
        self.x_correct = scipy.special.logit(np.array([0.6,  1.0, 1.5,  2.0, 2.4 ] )/ 3)
        self.y_correct = np.array([0, 0, 1.5, 1.0, 0.6]) / 9

    def test_pointwise(self):
        score = self.scorer.score(self.y_true, self.y_pred)
        expected = -np.sum(np.log([0.9, 0.5, 0.8])) / 3
        self.assertAlmostEqual(score, expected)

    def test_thresholded_pointwise(self):
        y_thresh = [0.2, 0.8, 0.8]
        thresh_score = self.scorer.score(self.y_true, y_thresh)
        expected_thresh = -np.sum(np.log([0.8, 0.8, 0.8])) / 3
        self.assertAlmostEqual(thresh_score, expected_thresh)

        y_clip = np.clip(self.y_pred, self.threshold_range[0], self.threshold_range[1])
        clip_score = self.scorer.score(self.y_true, y_clip)
        expected_clip = -np.sum(np.log([0.8, 0.5, 0.8])) / 3
        self.assertAlmostEqual(clip_score, expected_clip)

        score = self.scorer.score(self.y_true, self.y_pred, threshold_range=self.threshold_range)
        self.assertAlmostEqual(score, expected_clip - expected_thresh)

    def test_partition_loss(self):
        y_pred_iso = np.array([0, 1, 1])
        loss_near, loss_clip = (np.array([0.2, 0.2, 0.2]), np.array([0.2, 0.5, 0.2]))
        near_score = np.mean(-np.log(1 - loss_near))
        far_score = np.mean(-np.log(1 - loss_clip))

        correct_cl = far_score - near_score
        correct_dl = 0
        calibration_loss, discrimination_loss = self.scorer._partition_loss(
            self.y_true,
            self.y_pred,
            self.scorer.score,
            self.threshold_range,

        )
        self.assertAlmostEqual(calibration_loss, correct_cl)
        self.assertAlmostEqual(discrimination_loss, correct_dl)

    def test_make_x_and_y_curves(self):
        x_to_plot, y_to_plot, label = self.scorer._make_x_and_y_curves(
            self.y_true,
            self.y_pred,
            self.threshold_range,
            (0.3, 0.7),
        )
        assert_almost_equal(x_to_plot, self.x_correct)
        assert_almost_equal(y_to_plot, self.y_correct)
        label_correct = "LL: 0.112"
        self.assertEqual(label, label_correct)
        
    def test_plot_curve_and_get_colors(self):
        fig, ax = plt.subplots()
        self.scorer._plot_curve_and_get_colors(
            ax,
            self.x_correct,
            self.y_correct, 
            "this is a test",
        )
        plt.legend()
        plot_tester = mpc.PlotTester(ax)
        legend_given = [[ll.get_text() for ll in l.get_texts()] for l in plot_tester.get_legends()]
        df_data = pd.DataFrame({
            'x': self.x_correct,
            'y': self.y_correct,
        })
        plot_tester.assert_xydata(df_data, xcol='x', ycol='y')
        self.assertEqual(legend_given, [['this is a test']]) 

    def test_plot_curve(self):
        fig, ax = plt.subplots()
        self.scorer.plot_curve(
            ax,
            self.y_true,
            self.y_pred,
            self.threshold_range,
            fill_range=(0.1, 0.9),
        )
        plt.legend()
        plot_tester = mpc.PlotTester(ax)
        legend_given = [[ll.get_text() for ll in l.get_texts()] for l in plot_tester.get_legends()]
        df_data = pd.DataFrame({
            'x': self.x_correct,
            'y': self.y_correct,
        })
        data = plot_tester.get_xy().iloc[:-1]
        assert_almost_equal(np.array(df_data['x']), np.array(data['x']))
        label_correct = "LL: 0.235"
        self.assertEqual(legend_given, [[label_correct]])

class TestDCAScorer(unittest.TestCase):
    def setUp(self):
        self.y_true = [0, 1, 1]
        self.y_pred = [0.1, 0.5, 0.8]
        self.threshold_range = (0.2, 0.8)
        self.scorer = DCAScorer()
        self.scorer.n_points = 5
        self.x_correct = np.array([0.2,  0.35, 0.5,  0.65, 0.8])
        self.y_correct = np.array([2, 2, 2, 1, 1]) / 3

    def test_pointwise(self):
        score = self.scorer.score(self.y_true, self.y_pred)
        self.assertAlmostEqual(score, 0.1)

    def test_thresholded_pointwise(self):
        y_thresh = [0.2, 0.8, 0.8]
        thresh_score = self.scorer.score(self.y_true, y_thresh)
        expected_thresh = 0.04
        self.assertAlmostEqual(thresh_score, expected_thresh)

        y_clip = np.clip(self.y_pred, self.threshold_range[0], self.threshold_range[1])
        clip_score = self.scorer.score(self.y_true, y_clip)
        expected_clip = 0.11
        self.assertAlmostEqual(clip_score, expected_clip)

        score = self.scorer.score(self.y_true, self.y_pred, threshold_range=self.threshold_range)
        self.assertAlmostEqual(score, expected_clip - expected_thresh)

    def test_partition_loss(self):
        calibration_loss, discrimination_loss = self.scorer._partition_loss(
            self.y_true,
            self.y_pred,
            self.scorer.score,
            self.threshold_range,

        )
        self.assertAlmostEqual(calibration_loss, 0.07)
        self.assertAlmostEqual(discrimination_loss, 0.0)

    def test_make_x_and_y_curves(self):
        x_to_plot, y_to_plot, label = self.scorer._make_x_and_y_curves(
            self.y_true,
            self.y_pred,
            self.threshold_range,
            (0.3, 0.7),
        )
        print(x_to_plot, y_to_plot)
        assert_almost_equal(x_to_plot, self.x_correct)
        assert_almost_equal(y_to_plot, self.y_correct)
        label_correct = "Net Benefit: 0.07"
        self.assertEqual(label, label_correct)
        
    def test_plot_curve_and_get_colors(self):
        fig, ax = plt.subplots()
        self.scorer._plot_curve_and_get_colors(
            ax,
            self.x_correct,
            self.y_correct, 
            "this is a test",
        )
        plt.legend()
        plot_tester = mpc.PlotTester(ax)
        legend_given = [[ll.get_text() for ll in l.get_texts()] for l in plot_tester.get_legends()]
        df_data = pd.DataFrame({
            'x': self.x_correct,
            'y': self.y_correct,
        })
        plot_tester.assert_xydata(df_data, xcol='x', ycol='y')
        self.assertEqual(legend_given, [['this is a test']]) 

    def test_plot_curve(self):
        fig, ax = plt.subplots()
        self.scorer.plot_curve(
            ax,
            self.y_true,
            self.y_pred,
            self.threshold_range,
            fill_range=(0.1, 0.9),
        )
        plt.legend()
        plot_tester = mpc.PlotTester(ax)
        legend_given = [[ll.get_text() for ll in l.get_texts()] for l in plot_tester.get_legends()]
        df_data = pd.DataFrame({
            'x': self.x_correct,
            'y': self.y_correct,
        })
        data = plot_tester.get_xy()
        assert_almost_equal(np.array(df_data['x']), np.array(data['x']))
        label_correct = "Net Benefit: 0.07"
        self.assertEqual(legend_given, [[label_correct]])


