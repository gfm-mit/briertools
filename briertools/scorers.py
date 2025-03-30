import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import make_scorer


class MetricScorer(object):
    """
    Base class for scorers, each of which implements its own evaluation metric.
    """

    default_threshold_range = None
    special_xticks = False

    def __init__(self):
        """
        Initialize the class.

        The scorer is a sklearn scorer object that can be used to evaluate a classifier
        using the call signature metric_scorer.scorer(clf, y_true, y_pred).
        """
        self.scorer = make_scorer(self.score, greater_is_better=True)

    def _assert_valid(self, y_true: list[int], y_pred: list[float]) -> None:
        """
        Check validity of true & predicted labels (are they in the range of 0 and 1?)
        ----------
        Parameters:
        - y_true: array-like of shape (n_samples,)
          Ground truth (correct) labels.
        - y_pred: array-like of shape (n_samples,)
          Predicted labels, as returned by a classifier.

        Returns:
        None; raises AssertionError if y_true or y_pred contain invalid entries.
        ----------
        """
        assert np.min(y_true) >= 0
        assert np.min(y_pred) >= 0
        assert np.max(y_true) <= 1
        assert np.max(y_pred) <= 1

    def _pointwise_l1_loss(self, y_true: list[int], y_pred: list[float]) -> np.ndarray:
        """
        Return the pointwise L1 loss between true & predicted labels.
        ----------
        Parameters:
        - y_true: array-like of shape (n_samples,)
          Ground truth (correct) labels.
        - y_pred: array-like of shape (n_samples,)
          Predicted labels, as returned by a classifier.

        Returns:
        - loss: A numpy array of shape (n_samples,)
          The itemwise L1 loss between y_true and y_pred.
        ----------
        """
        self._assert_valid(y_true, y_pred)

        return np.abs(np.array(y_true) - np.array(y_pred))

    def _l1_to_total_log_loss(self, l1_loss: list[float]) -> float:
        """
        Return the average log loss given the pointwise L1 loss.
        ----------
        Parameters:
        - l1_loss: array-like of shape (n_samples,)
          Pointwise loss between two sets of labels.

        Returns:
        - score: float
          The average log-loss
        ----------
        """

        return np.mean(-np.log(1 - l1_loss))

    def _l1_to_total_l2_loss(self, l1_loss: list[float]) -> float:
        """
        Convert the pointwise L1 loss to L2  loss(squared error)
        ----------
        Parameters:
        - l1_loss: array-like of shape (n_samples,)
          Pointwise loss between two sets of labels.

        Returns:
        - score: float
          The average L2 loss (the average of the squared values in l1_loss)
        ----------
        """
        return np.mean(l1_loss ** 2)

    def _clip_loss(
        self,
        y_true: list[int],
        y_pred: list[float],
        threshold_range: tuple[float, float],
    ) -> tuple[float]:
        """
        Calculate the pointwise L1 loss of y_pred, relative to y_true,
        when predictions are clipped to the interval of threshold_range.
        ----------
        Parameters:
        - y_true: array-like of shape (n_samples,)
          Ground truth (correct) labels.
        - y_pred: array-like of shape (n_samples,)
          Predicted labels, as returned by a classifier.
        - threshold_range: tuple of two floats in the range [0, 1].
          The first entry should be strictly less than the second entry.

        Returns:
        - loss_near: array of shape (n_samples,)
          The pointwise L1 loss when predictions are clipped to the nearest endpoint of the threshold_range
        - loss_clip: array of shape (n_samples,)
          The pointwise L1 loss to the interval of threshold_range.
        ----------
        """
        y_near = np.array(threshold_range)[np.array(y_true, dtype=int)]
        y_far = np.array(threshold_range)[1 - np.array(y_true, dtype=int)]

        loss_near = self._pointwise_l1_loss(y_true, y_near)
        loss_far = self._pointwise_l1_loss(y_true, y_far)
        loss_pred = self._pointwise_l1_loss(y_true, y_pred)
        assert np.all(loss_near <= loss_far)

        loss_clip = np.clip(loss_pred, loss_near, loss_far)

        return loss_near, loss_clip

    def _partition_loss(
        self,
        y_true: list[int],
        y_pred: list[float],
        loss_fn: callable,
        thresholds: tuple[float, float] = None,
    ) -> tuple[float]:
        # TODO: document what this does better
        """
        Calculate the calibration & discrimination losses of y_pred, relative to y_true,
        using loss_fn to determine loss.
        ----------
        Parameters:
        - y_true: array-like of shape (n_samples,)
          Ground truth (correct) labels.
        - y_pred: array-like of shape (n_samples,)
          Predicted labels, as returned by a classifier.
        - loss_fn: callable
          A loss function that compares y_pred to y_true.
        - thresholds: tuple of two floats in range [0, 1]
          The first item should be strictly less than the second item.

        Returns:
        - calibration_loss: float
          Expresses how well the classifier that produced y_pred is calibrated
          (i.e. how well observed & predicted risk are in agreement)
        - discrimination_loss: float
          Expresses how well-separated the predictions for positive examples are
          from those for negative examples.
        ----------
        """
        assert (
            np.cov(y_pred, y_true)[0, 1] > 0
        ), "y_pred and y_true must be positively correlated"
        loss = loss_fn(y_true, y_pred, thresholds)
        ir = IsotonicRegression()
        y_pred_iso = ir.fit_transform(y_pred, y_true)
        discrimination_loss = loss_fn(y_true, y_pred_iso, thresholds)
        calibration_loss = loss - discrimination_loss

        return calibration_loss, discrimination_loss

    def _get_regret(
        self, y_true: list[int], y_pred: list[float], thresholds: np.ndarray
    ) -> list[float]:
        # TODO: document what this does better
        """
        Calculate the regret of using y_pred to predict the ground truth y_true
        over a range of thresholds.
        ----------
        Parameters:
        - y_true: array-like of shape (n_samples,)
          Ground truth (correct) labels.
        - y_pred: array-like of shape (n_samples,)
          Predicted labels, as returned by a classifier.
        - thresholds: numpy array of shape (n_items,)
          Should be generated using np.linspace(low_threshold, high_threshold, n_items)

        Returns:
        - costs: array-like of shape (n_items,) with total regret for each value in thresholds.
        ----------
        """
        if not isinstance(y_true, np.ndarray):
            y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)

        idx = np.argsort(y_pred)
        insertion_indices = np.searchsorted(y_pred[idx], thresholds)
        false_neg = np.concatenate([[0], np.cumsum(y_true[idx])])[insertion_indices]
        false_pos = (
            np.sum(1 - y_true[idx])
            - np.concatenate([[0], np.cumsum(1 - y_true[idx])])[insertion_indices]
        )

        costs = thresholds * false_pos + (1 - thresholds) * false_neg
        costs /= y_true.shape[0]

        return costs

    def _make_x_and_y_curves(
        y_true: list[int],
        y_pred: list[float],
        threshold_range: tuple[float, float],
        fill_range: tuple[float, float] = None,
    ) -> tuple:
        """
        Makes the X and Y values for the curves to be plotted.

        ----------
        Parameters:
        - threshold_range: tuple of (float, float)
          Strictly increasing tuple of floats in range [0, 1].
        - y_true: array-like of shape (n_samples,)
          Ground truth (correct) labels.
        - y_pred: array-like of shape (n_samples,)
          Predicted labels, as returned by a classifier.
        - fill_range : tuple of (float, float), optional
            Range to fill under the curve. Should be strictly increasing in the range [0, 1].

        Returns:
        - x_to_plot: array-like of shape (n_samples,)
          x-values of the curve to be plotted
        - y_to_plot: array-like of shape (n_samples,)
          y-values of the curve to be plotted
        - label: string
          label for the curve to be plotted (used in building a legend)
        -------
        """
        raise NotImplementedError("This needs to be implemented for a child class!")

    def _plot_curve_and_get_colors(
        self, x_to_plot: list[float], y_to_plot: list[float], label: str
    ) -> list:
        """
        Plot the curve described by x_to_plot and y_to_plot.
        Label it with label.
        Return a list of the colors used to plot.
        ----------
        Parameters:
        - x_to_plot: array-like of shape (n_samples,)
          x-values of the points in the plot.
        - y_to_plot: array-like of shape (n_samples,)
          y-values of the points in the plot
        - label: str
          Labels the curve being plotted (for the plot legend)

        Returns:
        - colors: iterable of the color used to plot x_to_plot and y_to_plot
        ----------
        """
        return plt.plot(x_to_plot, y_to_plot, label=label)[0].get_color()

    def _get_fill_between_params(
        self,
        y_true: list[int],
        y_pred: list[float],
        threshold_range: tuple[float, float],
        alpha: float,
        fill_range: tuple[float] = (0, 1),
    ):
        """
        Provides the values necessary to feed to plt.fill_between
        ----------
        Parameters:
        - y_true: array-like of shape (n_samples,)
          Ground truth (correct) labels.
        - y_pred: array-like of shape (n_samples,)
          Predicted labels, as returned by a classifier.
        - threshold_range: strictly increasing tuple of two floats in range [0, 1].
          The range of thresholds to be considered when scoring.
        - alpha: float.
          the transparency of the filled-in sections.
        - fill_range: strictly increasing tuple of two floats in range [0, 1].
          The range in which the plot will be filled in

        Returns:
        - fill_x: array-like of shape (n_thresholds,)
          For each value across the range of thresholds at which the scores are computed,
          this provides the threshold value.
        - fill_y_low: array-like of shape (n_thresholds,)
          For each threshold value, this is the lower bound of the filled-in area.
        - fill_y_high: array-like of shape (n_thresholds,)
          For each threshold value, this is the upper bound of the filled-in area.
        - fill_kwargs: dict of {str: object}
          Keyword arguments for the plt.fill_between function.
        ----------
        """
        raise NotImplementedError(
            "This is implemented individually for each child class."
        )

    def score(
        self,
        y_true: list[float],
        y_pred: list[float],
        threshold_range: tuple[float, float] = (0, 1),
    ) -> None:
        """
        A custom metric function, unique to each child class.
        ----------
        Parameters:
        - y_true: array-like of shape (n_samples,)
          Ground truth (correct) labels.
        - y_pred: array-like of shape (n_samples,)
          Predicted labels, as returned by a classifier.
        - threshold_range: strictly increasing tuple of two floats in range [0, 1].
          Optional; the range of thresholds to be considered when scoring.

        Returns:
        - score: float
          The computed metric.
        ----------
        """
        raise NotImplementedError("The scoring function needs to be implemented.")

    def plot_curve(
        self,
        y_true: list[float],
        y_pred: list[float],
        threshold_range: tuple[float, float] = None,
        fill_range: tuple[float, float] = None,
        ticks: list[float] = None,
        alpha: float = 0.3,
    ) -> None:
        """
        Plots a curve indicating this class's scoring function evaluated on y_pred
        with y_true as the ground truth.
        ----------
        Parameters:
        - y_true: array-like of shape (n_samples,)
          Ground truth (correct) labels.
        - y_pred: array-like of shape (n_samples,)
          Predicted labels, as returned by a classifier.
        - threshold_range: A strictly increasing tuple of two floats in range [0, 1]
          Optional; the range of thresholds to be considered when scoring.
        - fill_range: A strictly increasing tuple of two floats in range [0, 1]
          Optional; the range in which the plot will be filled in.
        - ticks: a list of floats of arbitrary length.
          Indicates where the x-ticks will be on the final plot.
        - alpha: float in the range [0, 1]
          Indicates how transparent any filled-in areas on the plot will be.

        Returns:
        None (produces a plot)
        ----------
        """
        if threshold_range is None:
            threshold_range = self.default_threshold_range
        x_to_plot, y_to_plot, label = self._make_x_and_y_curves(
            y_true,
            y_pred,
            threshold_range,
            fill_range=fill_range,
        )
        color = self._plot_curve_and_get_colors(x_to_plot, y_to_plot, label)
        if fill_range:
            (
                fill_x,
                fill_y_low,
                fill_y_high,
                fill_kwargs,
            ) = self._get_fill_between_params(
                y_true, y_pred, threshold_range, alpha, fill_range=fill_range
            )
            plt.fill_between(
                fill_x, fill_y_low, fill_y_high, color=color, **fill_kwargs
            )
        if self.special_xticks:
            ticks, tick_labels = self._get_xtick_labels(ticks, threshold_range)
            plt.xticks(ticks, tick_labels)

        plt.ylabel(self.ylabel)
        plt.xlabel(self.xlabel)
        plt.title(self.title)
        plt.legend()
        plt.tight_layout()


class DCAScorer(MetricScorer):
    # TODO: add more info
    """
    Decision curve analysis scorer object.
    """
    default_threshold_range = (0, 1)
    xlabel = "C/L"
    ylabel = "Regret"
    title = "DCA Curve"

    def _make_x_and_y_curves(
        self,
        y_true: list[int],
        y_pred: list[float],
        threshold_range: tuple[float, float],
        fill_range: tuple[float] = None,
    ):
        """
        Makes the X and Y values for DCA curves; produces a label indicating net benefit.

        ----------
        Parameters:
        - y_true: array-like of shape (n_samples,)
          Ground truth (correct) labels.
        - y_pred: array-like of shape (n_samples,)
          Predicted labels, as returned by a classifier.
        - threshold_range: tuple of (float, float)
          Strictly increasing tuple of floats in range [0, 1].
        - fill_range : tuple of (float, float), optional
            Range to fill under the curve. Should be strictly increasing in the range [0, 1].

        Returns:
        - x_to_plot: array-like of shape (n_samples,)
          x-values of the curve to be plotted
        - y_to_plot: array-like of shape (n_samples,)
          y-values of the curve to be plotted
        - label: string
          label for the curve to be plotted (used in building a legend)
        -------
        """
        thresholds = np.linspace(*threshold_range, 100)
        costs = self._get_regret(y_true, y_pred, thresholds)
        loss = self.score(y_true, y_pred, threshold_range)
        pi = np.mean(y_true)

        return thresholds, pi - costs / (1 - thresholds), f"Net Benefit: {loss:.2f}"

    def _get_fill_between_params(
        self,
        y_true: list[int],
        y_pred: list[float],
        threshold_range: tuple[float, float],
        alpha: float,
        fill_range: tuple[float] = (0, 1),
    ):
        """
        Provides the values necessary to feed to plt.fill_between
        ----------
        Parameters:
        - y_true: array-like of shape (n_samples,)
          Ground truth (correct) labels.
        - y_pred: array-like of shape (n_samples,)
          Predicted labels, as returned by a classifier.
        - threshold_range: strictly increasing tuple of two floats in range [0, 1].
          The range of thresholds to be considered when scoring.
        - alpha: float.
          the transparency of the filled-in sections.
        - fill_range: strictly increasing tuple of two floats in range [0, 1].
          The range in which the plot will be filled in.

        Returns:
        - fill_x: array-like of shape (100,)
          For each value across the range of thresholds at which the scores are computed,
          this provides the threshold value.
        - fill_y_low: array-like of shape (100,)
          For each threshold value, this is the lower bound of the filled-in area.
        - fill_y_high: array-like of shape (100,)
          For each threshold value, this is the upper bound of the filled-in area.
        - fill_kwargs: dict of {'alpha': alpha, 'zorder': -10}.
        ----------
        """
        thresholds = np.linspace(*threshold_range, 100)
        costs = self._get_regret(y_true, y_pred, thresholds)
        pi = np.mean(y_true)

        fill_kwargs = {
            "alpha": alpha,
            "zorder": -10,
        }

        return thresholds, pi - costs / (1 - thresholds), pi, fill_kwargs

    def score(
        self,
        y_true: list[float],
        y_pred: list[float],
        threshold_range: tuple[float] = None,
    ) -> float:
        """
        Calculates the DCA (Decision Curve Analysis) score.
        ----------
        Parameters:
        - y_true: array-like of shape (n_samples,)
          Ground truth (correct) labels.
        - y_pred: array-like of shape (n_samples,)
          Predicted labels, as returned by a classifier.
        - threshold_range: tuple of two strictly increasing floats in range [0, 1]; optional.
          The range to clip the true values to.

        Returns:
        - score: float
          The computed DCA score.
        ----------
        """
        self._assert_valid(y_true, y_pred)

        if threshold_range is None:
            return self._l1_to_total_l2_loss(self._pointwise_l1_loss(y_true, y_pred))

        loss_near, loss_clip = self._clip_loss(y_true, y_pred, threshold_range)
        near_score = self._l1_to_total_l2_loss(loss_near)
        far_score = self._l1_to_total_l2_loss(loss_clip)

        return far_score - near_score


class LogLossScorer(MetricScorer):
    # TODO: add more info here
    """
    Log loss scorer object.
    """
    default_threshold_range = (0.001, 0.999)
    xlabel = "C/L (as C:L-C odds)"
    ylabel = "Regret (lower is better)"
    title = "Brier Curve (Log Loss Version)"
    special_xticks = True

    def _make_x_and_y_curves(
        self,
        y_true: list[int],
        y_pred: list[float],
        threshold_range: tuple[float, float],
        fill_range: tuple[float] = None,
    ):
        """
        Makes the X and Y values for log loss curves; produces a label indicating overall log-loss.
        ----------
        Parameters:
        - y_true: array-like of shape (n_samples,)
          Ground truth (correct) labels.
        - y_pred: array-like of shape (n_samples,)
          Predicted labels, as returned by a classifier.
        - threshold_range: tuple of (float, float)
          Strictly increasing tuple of floats in range [0, 1].
        - fill_range : tuple of (float, float), optional
            Range to fill under the curve. Should be strictly increasing in the range [0, 1].

        Returns:
        - x_to_plot: array-like of shape (n_samples,)
          x-values of the curve to be plotted
        - y_to_plot: array-like of shape (n_samples,)
          y-values of the curve to be plotted
        - label: string
          label for the curve to be plotted (used in building a legend)
        -------
        """
        zscore = np.linspace(*scipy.special.logit(threshold_range), 1000)
        expit = scipy.special.expit(zscore)
        costs = self._get_regret(y_true, y_pred, expit)
        loss = self.score(y_true, y_pred, threshold_range=fill_range)
        return zscore, costs, f"LL: {loss:.3g}"

    def _get_fill_between_params(
        self,
        y_true: list[int],
        y_pred: list[float],
        threshold_range: tuple[float, float],
        alpha: float,
        fill_range: tuple[float] = (0.001, 0.999),
    ):
        """
        Provides the values necessary to feed to plt.fill_between
        ----------
        Parameters:
        - y_true: array-like of shape (n_samples,)
          Ground truth (correct) labels.
        - y_pred: array-like of shape (n_samples,)
          Predicted labels, as returned by a classifier.
        - threshold_range: strictly increasing tuple of two floats in range [0, 1].
          The range of thresholds to be considered when scoring.
        - alpha: float.
          the transparency of the filled-in sections.
        - fill_range: strictly increasing tuple of two floats in range [0, 1]; optional & defaults to (0.001, 0.999).
          The range in which the plot will be filled in.

        Returns:
        - fill_x: array-like of shape (1000,)
          For each value across the range of thresholds at which the scores are computed,
          this provides the threshold value.
        - fill_y_low: array-like of shape (1000,)
          For each threshold value, this is the lower bound of the filled-in area.
        - fill_y_high: array-like of shape (1000,)
          For each threshold value, this is the upper bound of the filled-in area.
        - fill_kwargs: dict of {'alpha': 0.3}.
        ----------
        """
        low, high = scipy.special.logit(fill_range)
        zscore = np.linspace(*scipy.special.logit(threshold_range), 1000)
        expit = scipy.special.expit(zscore)
        costs = self._get_regret(y_true, y_pred, expit)
        fill_idx = (low < zscore) & (zscore < high)
        fill_kwargs = {"alpha": 0.3}

        return zscore[fill_idx], costs[fill_idx], costs[fill_idx] * 0, fill_kwargs

    def _get_xtick_labels(self, ticks, threshold_range):
        """
        Helper function to get xticks for log-loss plots.

        ----------
        Parameters:
        - ticks: iterable of floats in the range [0, 1]. Optional.
          These values will be used as the xticks for the final plot.
        - threshold_range: strictly increasing tuple of floats in the range [0, 1]. Optional.
          Indicates the range of thresholds we're dealing with for our plot.

        Returns:
        - final_ticks: iterable of floats in range [0, 1]
          xticks for a logit-scaled plot
        - tick_labels: iterable of strings
          xtick labels for the plot
        ----------
        """
        if ticks is not None:
            tick_labels = np.round(
                np.where(
                    np.array(ticks) <= 0.5,
                    1.0 / np.array(ticks) - 1,
                    1 - 1.0 / (1 - np.array(ticks)),
                )
            )

            def format_tick(tick):
                """
                Format ticks to produce human-readable odds ratios.
                """
                if tick == 0.5:
                    return "(1:1)\nAccuracy"
                if tick > 0.5:
                    odds = 1.0 / (1 - tick) - 1
                    return f"{odds:.0f}:1"
                else:
                    odds = 1.0 / tick - 1
                    return f"1:{odds:.0f}"

            tick_labels = map(format_tick, ticks)

        elif threshold_range is not None:
            ticks = self._get_logit_ticks(threshold_range[0], threshold_range[1])
            tick_labels = ticks
        else:
            ticks = [0.01, 0.1, 0.5, 0.9, 0.99]
            tick_labels = ticks

        return scipy.special.logit(ticks), tick_labels

    def _get_logit_ticks(self, min_val, max_val):
        """
        Generate tick marks for logit-scaled plots using append/prepend operations.
        ----------
        Parameters:
        - min_val: float in [0, 1], strictly less than max_val
        - max_val: float in [0, 1], strictly greater than min_val.

        Returns:
        - ticks: iterable of floats usable as xticks for final plot.
        ----------
        """
        assert 0 <= min_val < max_val <= 1

        ticks = [0.5] if min_val <= 0.5 <= max_val else []

        bound = np.log10(min(min_val, 1 - max_val))
        bound = 2 - int(bound)

        for power in range(1, bound):
            val = 10.0 ** -power
            if min_val <= val <= max_val:
                ticks.insert(0, val)
            if min_val <= 1 - val <= max_val:
                ticks.insert(0, 1 - val)
        ticks = np.round(ticks, 16)

        return ticks

    def score(
        self,
        y_true: list[float],
        y_pred: list[float],
        threshold_range: tuple[float] = None,
    ) -> float:
        """
        Calculates the log loss score.
        ----------
        Parameters:
        - y_true: array-like of shape (n_samples,)
          Ground truth (correct) labels.
        - y_pred: array-like of shape (n_samples,)
          Predicted labels, as returned by a classifier.
        - threshold_range: tuple of two strictly increasing floats in range [0, 1]; optional.
          The range to clip the true values to.

        Returns:
        - score: float
          The computed log loss, optionally clipped to threshold_range.
        ----------
        """
        self._assert_valid(y_true, y_pred)
        if threshold_range is None:
            return self._l1_to_total_log_loss(self._pointwise_l1_loss(y_true, y_pred))
        loss_near, loss_clip = self._clip_loss(y_true, y_pred, threshold_range)
        near_score = self._l1_to_total_log_loss(loss_near)
        far_score = self._l1_to_total_log_loss(loss_clip)

        return far_score - near_score


class BrierScorer(MetricScorer):
    # TODO: say more here
    """
    Brier scorer object
    """
    default_threshold_range = (0, 1)
    special_xticks = False

    def _make_x_and_y_curves(
        self,
        y_true: list[int],
        y_pred: list[float],
        threshold_range: tuple[float, float],
        fill_range: tuple[float] = None,
    ):
        """
        Makes the X and Y values for Brier curves; produces a label indicating overall MSE.
        ----------
        Parameters:
        - y_true: array-like of shape (n_samples,)
          Ground truth (correct) labels.
        - y_pred: array-like of shape (n_samples,)
          Predicted labels, as returned by a classifier.
        - threshold_range: tuple of (float, float)
          Strictly increasing tuple of floats in range [0, 1].
        - fill_range : tuple of (float, float), optional
            Range to fill under the curve. Should be strictly increasing in the range [0, 1].

        Returns:
        - x_to_plot: array-like of shape (n_samples,)
          x-values of the curve to be plotted
        - y_to_plot: array-like of shape (n_samples,)
          y-values of the curve to be plotted
        - label: string
          label for the curve to be plotted (used in building a legend)
        -------
        """
        thresholds = np.linspace(*threshold_range, 100)
        costs = self._get_regret(y_true, y_pred, thresholds)

        loss = self.score(y_true, y_pred, threshold_range)
        integral = np.trapz(costs, thresholds) * 2
        return (
            thresholds,
            costs,
            f"MSE: {loss:.2f} | $\mathbb{{E}}$ R(f): {integral:.2f}",
        )

    def score(
        self,
        y_true: list[float],
        y_pred: list[float],
        threshold_range: tuple[float] = None,
    ) -> float:
        """
        Calculates the Brier score.
        ----------
        Parameters:
        - y_true: array-like of shape (n_samples,)
          Ground truth (correct) labels.
        - y_pred: array-like of shape (n_samples,)
          Predicted labels, as returned by a classifier.
        - threshold_range: tuple of two strictly increasing floats in range [0, 1]; optional.
          The range to clip the true values to.

        Returns:
        - score: float
          The computed Brier score, optionally clipped to threshold_range.
        ----------
        """
        self._assert_valid(y_true, y_pred)
        if threshold_range is None:
            return self._l1_to_total_l2_loss(self._pointwise_l1_loss(y_true, y_pred))
        loss_near, loss_clip = self._clip_loss(y_true, y_pred, threshold_range)
        near_score = self._l1_to_total_l2_loss(loss_near)
        far_score = self._l1_to_total_l2_loss(loss_clip)

        return far_score - near_score
