from matplotlib import pyplot as plt
import numpy as np
import scipy
from briertools.logloss import log_loss_curve


def simulate_binormal(loc, scale=1, scale_neg=1, loc_neg=None, n=3000, fix=True):
    if loc_neg is None:
        loc_neg = -loc
    neg = np.random.normal(loc=loc_neg, scale=scale_neg, size=n)
    pos = np.random.normal(loc=loc, scale=scale, size=n)
    if fix:
        pos, neg = neg, pos
    y_pred = scipy.special.expit(np.concatenate([pos, neg]))
    y_true = np.concatenate([pos * 0 + 1, neg * 0])

    return y_pred, y_true


def draw_curve(y_true, y_pred, **kwargs):
    return log_loss_curve(y_true, y_pred, **kwargs)


def jail():
    plt.figure(figsize=(4, 2))
    y_pred, y_true = simulate_binormal(1, 2, scale_neg=2)
    draw_curve(
        y_true,
        y_pred,
        draw_range=(0.003, 0.55),
        fill_range=(1.0 / 101, 1.0 / 6),
        ticks=[1.0 / 101, 1.0 / 6, 1.0 / 2],
    )

    y_pred, y_true = simulate_binormal(1, 1)
    draw_curve(
        y_true,
        y_pred,
        draw_range=(0.003, 0.55),
        fill_range=(1.0 / 101, 1.0 / 6),
        ticks=[1.0 / 101, 1.0 / 6, 1.0 / 2],
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


def fraud():
    plt.figure(figsize=(4, 2))
    y_pred, y_true = simulate_binormal(1, 1.5)
    draw_curve(
        y_true,
        y_pred,
        draw_range=(0.333, 0.995),
        fill_range=(0.925, 0.983),
        ticks=[1.0 / 2, 10.0 / 11, 100.0 / 101],
    )

    y_pred, y_true = simulate_binormal(1, 5)
    draw_curve(
        y_true,
        y_pred,
        draw_range=(0.333, 0.995),
        fill_range=(0.925, 0.983),
        ticks=[1.0 / 2, 10.0 / 11, 100.0 / 101],
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


def cancer():
    plt.figure(figsize=(4, 2))
    y_pred, y_true = simulate_binormal(1, 1, scale_neg=8)
    draw_curve(
        y_true,
        y_pred,
        draw_range=(0.03, 0.66),
        fill_range=(1.0 / 11, 1.0 / 3),
        ticks=[1.0 / 11, 1.0 / 3, 1.0 / 2],
    )

    y_pred, y_true = simulate_binormal(1, 1)
    draw_curve(
        y_true,
        y_pred,
        draw_range=(0.03, 0.66),
        fill_range=(1.0 / 11, 1.0 / 3),
        ticks=[1.0 / 11, 1.0 / 3, 1.0 / 2],
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


jail()
cancer()
fraud()
