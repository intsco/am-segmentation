import logging
from functools import wraps
from pathlib import Path
from shutil import rmtree
from time import time

import cv2
from matplotlib import pyplot as plt
import numpy as np

logger = logging.getLogger('am-segm')


def min_max(a):
    return a.min(), a.max()


def clean_dir(path):
    logger.info(f'Cleaning up {path} directory')
    rmtree(path, ignore_errors=True)
    Path(path).mkdir(parents=True)


def time_it(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        res = func(*args, **kwargs)
        minutes, seconds = divmod(time() - start, 60)
        logger.info(f"Function '{func.__name__}' running time: {minutes:.0f}m {seconds:.0f}s")
        return res

    return wrapper


def read_image(path):
    return cv2.imread(str(path))[:, :, 0]  # because ch0==ch1==ch2


def plot_overlay(image, mask, figsize=(10, 10)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    image, mask = np.array(image), np.array(mask)
    if image.ndim > 2:
        image = image[0]  # plot first channel only
    if mask.ndim > 2:
        mask = mask[0]  # plot first channel only
    ax.imshow(image, cmap='gray', interpolation=None)  # plot first channel only
    ax.imshow(mask, cmap='jet', interpolation=None, alpha=0.5)
    return fig


def overlay_images_with_masks(path, image_ext='png'):
    for group_path in path.iterdir():
        logger.info(f'Overlaying: {group_path}')
        mask = read_image(str(group_path / f'mask.{image_ext}'))
        image = read_image(str(group_path / f'source.{image_ext}'))
        assert image.shape == mask.shape

        fig = plot_overlay(image, mask)
        plt.savefig(group_path / f'overlay.{image_ext}', dpi=600, bbox_inches='tight')
        plt.close()
