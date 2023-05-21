import scienceplots
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
import matplotlib.pylab as pylab
import random

params = {'legend.fontsize': 'small',
         'axes.labelsize': 'medium',
         'axes.titlesize':'medium',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)
plt.style.use(["science", "ieee", "bright", 'no-latex'])

random.seed(303251)

# Load data
strebelle = np.loadtxt("snesim/data/ti_strebelle.out", skiprows=3).reshape((250, 250))
windows = view_as_windows(strebelle, (150, 150))

# Create plot
fig, axes = plt.subplots(nrows=2, ncols=2)
for ax in axes.flat:
    ax.axis('off')

# Plot training data
plot_windows = windows[np.random.choice(windows.shape[0], size=4, replace=False),
                       np.random.choice(windows.shape[1], size=4, replace=False)]
for ax, im in zip(axes.flat, plot_windows):
    ax.imshow(im, cmap='gray')
fig.suptitle("Training data")
fig.tight_layout()
plt.show()


# Plotting TI sampled images
generated = np.load("generative_model/data/realizations_old.npy").reshape((100, 150, 150))
fig, axes = plt.subplots(nrows=2, ncols=2)
for ax in axes.flat:
    ax.axis('off')
generated_windows = generated[np.random.choice(generated.shape[0], size=4, replace=False)]
for ax, im in zip(axes.flat, generated_windows):
    ax.imshow(im, cmap='gray')

fig.suptitle("Generated GAN")
fig.tight_layout()
plt.show()
