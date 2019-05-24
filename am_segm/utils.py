import numpy as np
from matplotlib import pyplot as plt

from concurrent.futures import ProcessPoolExecutor


def convert_to_image(tensor):
    tensor = tensor[0]
    image = tensor.detach().cpu().numpy()
    if image.ndim > 2:
        image = image[0]
    return image


def plot_images_row(images, titles=None):
    n = min(len(images), 4)
    fig, axes = plt.subplots(1, n, figsize=(16, 8))
    for i in range(n):
        axes[i].imshow(images[i])
        if titles:
            axes[i].set_title(titles[i])
    return fig


def plot_overlay(img, mask, figsize=(10, 10)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    img, mask = np.array(img), np.array(mask)
    if img.ndim > 2:
        img = img[0]  # plot first channel only
    if mask.ndim > 2:
        mask = mask[0]  # plot first channel only
    ax.imshow(img, cmap='gray', interpolation=None)  # plot first channel only
    ax.imshow(mask, cmap='jet', interpolation=None, alpha=0.5)
    return fig
