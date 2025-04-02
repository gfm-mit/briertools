import numpy as np
import matplotlib.pyplot as plt
from matplotlib.scale import FuncScale


def scale_x_one_minus_one_minus_x_2(ax):
    def forward(x):
        return 1 - (1 - x) ** 2  # This is the actual transformation for labels

    def inverse(y):
        one_minus = 1 - y
        return 1 - np.sign(one_minus) * np.sqrt(np.abs(one_minus))

    ax.set_xscale(FuncScale(ax.xaxis, (forward, inverse)))
    ax.set_title("1-(1-x)^2")
