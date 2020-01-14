import logging
from functools import wraps
from pathlib import Path
from shutil import rmtree
from time import time

import cv2
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

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


def save_overlay(source, mask, path, dpi=100):
    h, w = source.shape
    plot_size_in = (w / dpi, h / dpi)
    fig, ax = plot_overlay(source, mask, figsize=plot_size_in)
    ax.set_position([0, 0, 1, 1])  # Critical!
    ax.axis('off')
    plt.savefig(path, bbox_inches=Bbox([[0, 0], plot_size_in]))
    plt.close()


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def load_model(model_path):
    logger.info(f'Loading model from "{model_path}"')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = smp.Unet(encoder_name='se_resnext50_32x4d',
                     encoder_weights=None, decoder_use_batchnorm=True)
    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    model.eval()
    return model.to(device)
