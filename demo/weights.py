from matplotlib import pyplot as plt
import numpy as np
import scipy
from scipy.stats import beta

def weights():
  z = np.linspace(-5, 5, 100)
  x = scipy.special.expit(z)
  one = x * 0 + 1
  beta00 = one/x/(1-x)/40
  lower = np.minimum(one, beta00)
  fig, axs = plt.subplots(1, 2, figsize=(8, 3))
  plt.sca(axs[0])

  color1 = plt.plot(x, one, label="Brier Score")[0].get_color()
  color2 = plt.plot(x, beta00, label="Log Loss")[0].get_color()
  plt.fill_between(x, one, lower, color=color1, alpha=0.2, zorder=-10)
  plt.fill_between(x, beta00, lower, color=color2, alpha=0.2, zorder=-10)
  plt.fill_between(x, one * 0, lower, color="black", alpha=0.2, zorder=-10)
  color3 = plt.plot([0, 0.5, 0.5, 0.5, 1], [0.01, 0.01, 3, 0.01, 0.01], label="Accuracy", linewidth=2)[0].get_color()
  plt.ylim([0,3])
  plt.yticks([])
  plt.xlabel("C/L\n(Linear Scale)")
  plt.xticks([1./101, 1./11, 1./4., 1./2, 3./4, 10./11, 100./101],
             "\n1:100 1:10 1:3 1:1 3:1 10:1 \n100:1".split(" "))
  plt.legend(loc="upper right")
  plt.title("Distribution of $c$")

  plt.sca(axs[1])
  w = x * (1-x)
  one = w
  beta00 = x * 0 + .1
  lower = np.minimum(one, beta00)
  plt.plot(z, one, color=color1, label="Brier Score")
  plt.plot(z, beta00, color=color2, label="Log Loss")
  plt.fill_between(z, one, lower, color=color1, alpha=0.2, zorder=-10)
  plt.fill_between(z, beta00, lower, color=color2, alpha=0.2, zorder=-10)
  plt.fill_between(z, one * 0, lower, color="black", alpha=0.2, zorder=-10)
  plt.plot([-5, 0, 0, 0, 5], [0.001, 0.001, 2, 0.001, 0.001], color=color3, label="Accuracy", linewidth=2)
  plt.ylim([0,.3])
  plt.yticks([])
  plt.xlabel("\nC / L\n(Logistic Scale)")
  plt.xticks(scipy.special.logit([1./101, 1./11, 1./4., 1./2, 3./4, 10./11, 100./101]),
             "1:100 1:10 1:3 1:1 3:1 10:1 100:1".split(" "))
  plt.legend(loc="upper right")
  plt.title(r"Distribution of $\log \frac{c}{1-c}$")
  for ax in axs:
    plt.sca(ax)
    plt.ylabel("Averaging Weight")
  plt.suptitle("Two Ways of Thinking about Cost Distributions")
  plt.tight_layout()
  plt.show()

def weights_hand():
  z = np.linspace(-5, 5, 100)
  x = scipy.special.expit(z)
  one = x * 0 + 1
  beta00 = one/x/(1-x)/40
  beta22 = beta.pdf(x, 2, 2)
  beta28 = beta.pdf(x, 2, 8)
  shifted = 4/(4 * x + 1 * (1-x))**2
  lower = np.minimum(one, beta00)
  fig, axs = plt.subplots(1, 2, figsize=(8, 3))
  plt.sca(axs[0])

  color1 = plt.plot(x, one, label="Beta(1,1)", color="gray")[0].get_color()
  color3 = plt.plot(x, beta22, label="Beta(2,2)")[0].get_color()
  color4 = plt.plot(x, beta28, label="Beta(2,8)")[0].get_color()
  color5 = plt.plot(x, shifted, label="Shifted Brier", alpha=0.5, linewidth=2, linestyle=":", zorder=-10)[0].get_color()
  plt.axvline(x=1/8, color=color4, linestyle="--", lw=0.5, zorder=-10)
  plt.text(1/8, -.8, "1/8", color=color4, fontsize=8, ha="center", va="center", rotation=0)
  plt.fill_between(x, one, one * 0, color=color1, alpha=0.2, zorder=-10)
  plt.fill_between(x, shifted, np.minimum(shifted, one), color=color5, alpha=0.2, zorder=-10)
  plt.ylim([0,3.8])
  plt.xlabel("C/L")
  plt.xticks([1./101, 1./11, 1./4., 1./2, 3./4, 10./11, 100./101],
             "\n1:100 1:10 1:3 1:1 3:1 10:1 \n100:1".split(" "))
  plt.legend(loc="upper right")
  plt.yticks([])
  plt.xlabel("C/L\n(Linear Scale)")
  plt.title("Distribution of $c$")

  plt.sca(axs[1])
  w = x * (1-x)
  one = w
  beta00 = x * 0 + .1
  beta39 = beta.pdf(x, 3, 9) / 10
  lower = np.minimum(one, beta00)
  plt.plot(z, one, color=color1, label="Brier")
  plt.plot(z, beta22 * w, color=color3, label="Hand")
  plt.plot(z, beta39, color=color4, label="Zhu")
  plt.plot(z, shifted * w, color=color5, alpha=0.5, linewidth=2, linestyle=":", label="Shifted Brier", zorder=-10)
  plt.axvline(x=scipy.special.logit(1/8), color=color4, linestyle="--", lw=0.5, zorder=-10)
  plt.text(scipy.special.logit(1/8), -.2, "1/8", color=color4, fontsize=8, ha="center", va="center", rotation=0)
  plt.fill_between(z, one, one * 0, color=color1, alpha=0.2, zorder=-10)
  plt.fill_between(z, shifted*w, np.minimum(shifted*w, one), color=color5, alpha=0.2, zorder=-10)
  plt.ylim([0,.95])
  plt.xlabel("\nC / L\n(Logistic Scale)")
  plt.xticks(scipy.special.logit([1./101, 1./11, 1./4., 1./2, 3./4, 10./11, 100./101]),
             "1:100 1:10 1:3 1:1 3:1 10:1 100:1".split(" "))
  plt.legend(loc="upper right")
  plt.yticks([])
  plt.title(r"Distribution of $\log \frac{c}{1-c}$")

  for ax in axs:
    plt.sca(ax)
    plt.ylabel("Averaging Weight")
  plt.suptitle("Priors over Thresholds")
  plt.tight_layout()
  plt.show()

weights_hand()