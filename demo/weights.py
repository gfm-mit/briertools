from matplotlib import pyplot as plt
import numpy as np
import scipy

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
  color3 = plt.plot([0, 0.5, 0.5, 0.5, 1], [0.01, 0.01, 2, 0.01, 0.01], label="Accuracy")[0].get_color()
  plt.ylim([0,1.5])
  plt.xlabel("C/L")
  plt.xticks([1./101, 1./11, 1./4., 1./2, 3./4, 10./11, 100./101],
             "\n1:100 1:10 1:3 1:1 3:1 10:1 \n100:1".split(" "))
  plt.legend(loc="lower right")

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
  plt.plot([-5, 0, 0, 0, 5], [0.01, 0.01, 2, 0.01, 0.01], color=color3, label="Accuracy")
  plt.ylim([0,.5])
  plt.xlabel("log odds C / L")
  plt.xticks(scipy.special.logit([1./101, 1./11, 1./4., 1./2, 3./4, 10./11, 100./101]),
             "1:100 1:10 1:3 1:1 3:1 10:1 100:1".split(" "))
  plt.legend(loc="upper right")
  for ax in axs:
    plt.sca(ax)
    plt.ylabel("Averaging Weight")
    plt.title("Priors over Thresholds")
  plt.tight_layout()
  plt.show()

weights()