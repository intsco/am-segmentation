import math
from pathlib import Path
from typing import Union

import torch
from matplotlib import pyplot as plt
import numpy as np

from am.segment.image_utils import save_rgb_image, normalize


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
        save_rgb_image(np.array(overlay), overlay_path / f'{i:02}.png')


def plot_images_grid(images, titles=None, ncols=3):
    n = len(images)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 5))
    axes = axes.flatten()
    for i, image in enumerate(images):
        array = np.array(image)
        # if array.ndim > 2:
        #     array = array[:, :, 0]
        axes[i].imshow(array)
        if titles:
            axes[i].set_title(titles[i])
    plt.show()


def convert_to_image(array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if type(array) == torch.Tensor:
        array = array.detach().cpu().numpy()

    if array.ndim == 4:
        image = array[0]
    else:
        image = array

    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0]
        else:
            image = np.moveaxis(image, 0, 2)

    return normalize(image)


def predict_plot_batch(model, inputs, targets):
    targets_pred = torch.sigmoid(model(inputs.to('cuda')))
    n = min(len(inputs), len(targets))
    for i in range(n):
        plot_images_grid([
            convert_to_image(targets[i]),
            convert_to_image(inputs[i]),
            # convert_to_image(targets_pred[i]),
            convert_to_image((targets_pred[i] > 0.5).to(torch.float32))
        ], titles=['Labels', 'Inputs', 'Mask'])


def predict_plot(model, dl, n=2):
    iterator = iter(dl)
    while n > 0:
        inputs, targets = next(iterator)
        batch_n = inputs.shape[0]
        if batch_n > n:
            inputs, targets = inputs[:n], targets[:n]

        predict_plot_batch(model, inputs, targets)
        n -= batch_n


def create_uniq_exp_group(dl):
    return set(map(tuple, dl.dataset.image_df[['experiment', 'group']].values))
