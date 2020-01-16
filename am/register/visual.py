import logging
from math import ceil

from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.patches as patches
import numpy as np

from am.utils import min_max, time_it

logger = logging.getLogger('am-segm')


def cut_patch(image, y_offset=0, x_offset=0, patch=1000):
    return image[y_offset:y_offset+patch, x_offset:x_offset+patch]


def plot_image(image, figsize=(16, 10), dpi=100, **kwargs):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(image, **kwargs)
    return fig, ax


def plot_axis_hist(axis_hist, axis_coords, labels, uniq_labels):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.plot(np.arange(axis_hist.shape[0]), axis_hist, c='black')

    norm = colors.Normalize(vmin=min(uniq_labels), vmax=max(uniq_labels))
    cmap = plt.cm.get_cmap('jet', len(uniq_labels))
    for label in uniq_labels:
        x_min, x_max = min_max(axis_coords[labels == label])
        y_min, y_max = min_max(axis_hist)

        rect = patches.Rectangle(xy=(x_min, y_min),
                                 width=x_max - x_min, height=y_max - y_min,
                                 color=cmap(norm(label)))
        ax.add_patch(rect)


def plot_labels(ax, image, target_axis, axis_coords, labels, uniq_labels):
    norm = colors.Normalize(vmin=min(uniq_labels), vmax=max(uniq_labels))
    cmap = plt.cm.get_cmap('jet', len(uniq_labels))
    for label in uniq_labels:
        if target_axis == 1:
            x_min, x_max = min_max(axis_coords[labels == label])
            y_min, y_max = (0, image.shape[0])
        else:
            x_min, x_max = (0, image.shape[1])
            y_min, y_max = min_max(axis_coords[labels == label])

        rect = patches.Rectangle(xy=(x_min, y_min),
                                 width=x_max - x_min, height=y_max - y_min,
                                 alpha=0.4, color=cmap(norm(label)))
        ax.add_patch(rect)


def plot_image_label_overlay(image, target_axis, axis_coords, labels, uniq_labels):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image, cmap='gray', alpha=0.6)
    plot_labels(image, ax, target_axis, axis_coords, labels, uniq_labels)


def shift_center_coords(centers, indices, row_offset, col_offset, patch_size):
    mask = (row_offset < centers[:, 0]) & (centers[:, 0] < row_offset + patch_size) &\
           (col_offset < centers[:, 1]) & (centers[:, 1] < col_offset + patch_size)
    shifted_centers = centers[mask]
    shifted_centers[:, 0] -= row_offset
    shifted_centers[:, 1] -= col_offset
    return shifted_centers, indices[mask]


def plot_am_labels(
    mask, centers, labels, source=None, row_offset=None, col_offset=None, patch_size=None
):
    if patch_size:
        centers, labels = shift_center_coords(centers, labels, row_offset, col_offset, patch_size)
        mask = mask[row_offset : (row_offset + patch_size), col_offset : (col_offset + patch_size)]
        if source is not None:
            source = source[
                row_offset : (row_offset + patch_size), col_offset : (col_offset + patch_size)
            ]

    dpi = 100
    figsize = (ceil(mask.shape[1] / dpi), ceil(mask.shape[0] / dpi))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if source is None:
        ax.imshow(mask, cmap='viridis', interpolation=None)
    else:
        ax.imshow(source, cmap='gray', interpolation=None)  # plot first channel only
        ax.imshow(mask, cmap='viridis', interpolation=None, alpha=0.3)

    circle_size = (4 * 72 / fig.dpi) ** 2  # circle size (diameter**2) in points
    fontsize = 10 * 72 / fig.dpi  # text size in points
    ys, xs = zip(*centers)
    ax.scatter(xs, ys, color='blue', s=circle_size, linewidth=0)
    for (y, x), label in zip(centers, labels):
        ax.text(x, y, str(label), color='red', fontsize=fontsize)


@time_it
def overlay_image_with_am_labels(source, mask, am_centers, acq_indices, path):
    logger.info(f'Overlaying image with AM labels and saving at {path}')
    plot_am_labels(mask=mask, centers=am_centers, labels=acq_indices, source=source)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_coord_peaks(hist, peaks, height=None):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(hist)
    ax.plot(peaks, hist[peaks], 'x')
    if height:
        ax.plot(np.ones_like(hist) * height, '--')
