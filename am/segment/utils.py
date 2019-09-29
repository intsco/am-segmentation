import logging
from pathlib import Path
from shutil import rmtree

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

logger = logging.getLogger('am-segm')


def convert_to_image(array):
    if type(array) == torch.Tensor:
        image = array.detach().cpu().numpy()
    else:
        image = array
    if image.ndim == 4:
        image = image[0][0]
    elif image.ndim == 3:
        image = image[0]
    return image


def plot_images_row(images, titles=None):

    def plot(axis, image, title=None):
        image = convert_to_image(image)
        if image.ndim > 2:
            image = image[0]
        axis.imshow(image)
        if title:
            axis.set_title(title)

    n = min(len(images), 4)
    fig, axes = plt.subplots(1, n, figsize=(16, 8))
    if isinstance(axes, np.ndarray):
        for i in range(n):
            plot(axes[i], images[i], titles[i] if titles else None)
    else:
        plot(axes, images[0], titles[0] if titles else None)
    return fig
