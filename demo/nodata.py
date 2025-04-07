import argparse
from matplotlib import pyplot as plt
import numpy as np
import scipy
from scipy.stats import beta

X_TICKS = [1.0 / 100, 1.0 / 10, 1.0 / 4.0, 1.0 / 2, 3.0 / 4, 9.0 / 10, 99.0 / 100]
X_TICK_LABELS = "\n1/100 1/10 1/4 1/2 3/4 9/10 \n99/100".split(" ")

def weights():
    z = np.linspace(-5, 5, 100)
    x = scipy.special.expit(z)
    one = x * 0 + 1
    beta00 = one / x / (1 - x) / 40
    lower = np.minimum(one, beta00)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3), gridspec_kw={'width_ratios': [2, 2, 1]})
    plt.sca(axs[0])
    
    # Add panel label A
    axs[0].text(0.08, 0.95, 'A', transform=axs[0].transAxes, 
                fontsize=12, fontweight='bold', va='top')

    color1 = plt.plot(x, one, label="Brier Score")[0].get_color()
    color2 = plt.plot(x, beta00, label="Log Loss")[0].get_color()
    plt.fill_between(x, one, lower, color=color1, alpha=0.2, zorder=-10)
    plt.fill_between(x, beta00, lower, color=color2, alpha=0.2, zorder=-10)
    plt.fill_between(x, one * 0, lower, color="black", alpha=0.2, zorder=-10)
    color3 = plt.plot(
        [0, 0.5, 0.5, 0.5, 1],
        [0.01, 0.01, 3, 0.01, 0.01],
        label="Accuracy",
        linewidth=2,
    )[0].get_color()
    plt.ylim([0, 3])
    plt.yticks([])
    plt.xlabel("C/L\n(Linear Scale)")
    plt.xticks(X_TICKS, X_TICK_LABELS)
    plt.title("Distribution of $c$")

    plt.sca(axs[1])
    
    # Add panel label B
    axs[1].text(0.05, 0.95, 'B', transform=axs[1].transAxes, 
                fontsize=12, fontweight='bold', va='top')

    w = x * (1 - x)
    one = w
    beta00 = x * 0 + 0.1
    lower = np.minimum(one, beta00)
    plt.plot(z, one, color=color1, label="Brier Score")
    plt.plot(z, beta00, color=color2, label="Log Loss")
    plt.fill_between(z, one, lower, color=color1, alpha=0.2, zorder=-10)
    plt.fill_between(z, beta00, lower, color=color2, alpha=0.2, zorder=-10)
    plt.fill_between(z, one * 0, lower, color="black", alpha=0.2, zorder=-10)
    plt.plot(
        [-5, 0, 0, 0, 5],
        [0.001, 0.001, 2, 0.001, 0.001],
        color=color3,
        label="Accuracy",
        linewidth=2,
    )
    plt.ylim([0, 0.3])
    plt.yticks([])
    plt.xlabel("\nC / L\n(Logistic Scale)")
    plt.xticks(scipy.special.logit(X_TICKS), X_TICK_LABELS)
    plt.title(r"Distribution of $\log \frac{c}{1-c}$")

    handles, labels = axs[0].get_legend_handles_labels()
    axs[2].legend(handles, labels, loc='center', frameon=False)
    axs[2].axis('off')

    for ax in axs[:2]:
        plt.sca(ax)
        plt.ylabel("Averaging Weight")
    plt.suptitle("Two Ways of Thinking about Cost Distributions")
    plt.tight_layout()
    return plt.gca()


def weights_hand():
    z = np.linspace(-5, 5, 100)
    x = scipy.special.expit(z)
    one = x * 0 + 1
    beta00 = one / x / (1 - x) / 40
    beta22 = beta.pdf(x, 2, 2)
    beta28 = beta.pdf(x, 2, 8)
    beta315 = beta.pdf(x, 3, 15)
    shifted = 7 / (7 * x + 1 * (1 - x)) ** 2
    lower = np.minimum(one, beta00)

    fig, axs = plt.subplots(1, 3, figsize=(10, 3), gridspec_kw={'width_ratios': [2, 2, 1]})
    plt.sca(axs[0])
    
    # Add panel label A
    axs[0].text(0.02, 0.95, 'A', transform=axs[0].transAxes, 
                fontsize=12, fontweight='bold', va='top')

    color1 = plt.plot(x, one, label="Beta(1,1) [Brier]")[0].get_color()
    plt.plot(x, beta22, label="Beta(2,2) [Hand]", linestyle=":", color=color1, alpha=0.7)
    color4 = plt.plot(x, beta28, label="Beta(2,8) [Zhu, et al.]")[0].get_color()
    plt.plot(x, beta315, label="Beta(3,15) [Zhu, et al.]", linestyle=":", color=color4, alpha=0.7)
    color5 = plt.plot(
        x,
        shifted,
        label="Shifted Brier",
        alpha=0.5,
        linewidth=2,
        zorder=-10,
    )[0].get_color()
    plt.axvline(x=1 / 8, color=color4, linestyle="--", lw=0.5, zorder=-10)
    plt.text(
        1 / 8,
        -1.25,
        "1/8",
        color=color4,
        fontsize=8,
        ha="center",
        va="center",
        rotation=0,
    )
    plt.ylim([0, 5.2])
    plt.xlabel("C/L")
    plt.xticks(X_TICKS, X_TICK_LABELS)
    plt.yticks([])
    plt.xlabel("C/L\n(Linear Scale)")
    plt.title("Distribution of $c$")

    plt.sca(axs[1])
    
    # Add panel label B
    axs[1].text(0.02, 0.95, 'B', transform=axs[1].transAxes, 
                fontsize=12, fontweight='bold', va='top')

    w = x * (1 - x)
    one = w
    beta00 = x * 0 + 0.1
    lower = np.minimum(one, beta00)
    plt.plot(z, one, color=color1)
    plt.plot(z, beta22 * w, color=color1, linestyle=":", alpha=0.7)
    plt.plot(z, beta28 * w, color=color4)
    plt.plot(z, beta315 * w, color=color4, linestyle=":", alpha=0.7)
    plt.plot(
        z,
        shifted * w,
        color=color5,
        alpha=0.5,
        linewidth=2,
        zorder=-10,
    )
    plt.axvline(
        x=scipy.special.logit(1 / 8), color=color4, linestyle="--", lw=0.5, zorder=-10
    )
    plt.text(
        scipy.special.logit(1 / 8),
        -0.155,
        "1/8",
        color=color4,
        fontsize=8,
        ha="center",
        va="center",
        rotation=0,
    )
    plt.ylim([0, 0.65])
    plt.xlabel("C / L\n(Logistic Scale)")
    plt.xticks(scipy.special.logit(X_TICKS), X_TICK_LABELS)
    plt.yticks([])
    plt.title(r"Distribution of $\log \frac{c}{1-c}$")

    handles, labels = axs[0].get_legend_handles_labels()
    axs[2].legend(handles, labels, loc='center', frameon=False)
    axs[2].axis('off')

    for ax in axs[:2]:
        plt.sca(ax)
        plt.ylabel("Averaging Weight")
    plt.suptitle("Priors over Thresholds")
    plt.tight_layout()
    return plt.gca()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--flag", action="store_true")
  parser.add_argument("--out", type=str)
  args = parser.parse_args()
  if args.flag:
    ax = weights()
  else:
    ax = weights_hand()
  if args.out:
    ax.figure.savefig(args.out)
  else:
    ax.figure.show()