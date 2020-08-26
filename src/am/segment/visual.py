import math
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from am.segment.image_utils import save_rgb_image


def plot_overlay(source, mask, figsize=(10, 10)):
    source, mask = np.array(source), np.array(mask)
    if source.ndim > 2:
        source = source[0]  # plot first channel only
    if mask.ndim > 2:
        mask = mask[0]  # plot first channel only
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(source, cmap='gray', interpolation=None)  # plot first channel only
    ax.imshow(mask, cmap='viridis', interpolation=None, alpha=0.3)
    return fig, ax


def plot_loss_metric(history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    xs = range(1, len(history.train_loss) + 1)

    axes[0].set_title('loss')
    axes[0].set_ylim(0, min(10, max(history.train_loss.max(), history.valid_loss.max()) * 1.1))
    axes[0].plot(xs, history.train_loss, c='g', label='train')
    axes[0].plot(xs, history.valid_loss, c='r', label='valid')
    axes[0].legend()

    axes[1].set_title('metric')
    axes[1].set_ylim(0,  min(10, max(history.train_metric.max(), history.valid_metric.max()) * 1.1))
    axes[1].plot(xs, history.train_metric, c='g', label='train')
    axes[1].plot(xs, history.valid_metric, c='r', label='valid')
    axes[1].legend()


def save_history_overlay(history, path):
    overlay_path = Path(path)
    overlay_path.mkdir(exist_ok=True)
    for i, overlay in enumerate(history.overlay, 1):
        save_rgb_image(np.array(overlay), overlay_path / f'{i}.png')


def plot_images_grid(images, ncols=3):
    n = len(images)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 5))
    axes = axes.flatten()
    for i, image in enumerate(images):
        axes[i].imshow(np.array(image))
