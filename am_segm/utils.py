import json
import os
from pathlib import Path
from shutil import rmtree

import cv2
import numpy as np
from albumentations import CenterCrop
from matplotlib import pyplot as plt

from am_segm.image_utils import stitch_tiles


def read_image(path):
    return cv2.imread(str(path))[:,:,0]  # because ch0==ch1==ch2


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


def stitch_crop_tiles(tiles_path, tile_size, meta):
    tile_paths = sorted(tiles_path.glob('*.png'))
    if len(tile_paths) != meta['tile']['rows'] * meta['tile']['cols']:
        print(f'Number of tiles does not match meta: {len(tile_paths)}, {meta}')

    tiles = [None] * len(tile_paths)
    for path in tile_paths:
        i = int(path.stem)
        tiles[i] = cv2.imread(str(path))[:,:,0]  # because ch0==ch1==ch2

    stitched_image = stitch_tiles(tiles, tile_size, meta['tile']['rows'], meta['tile']['cols'])
    stitched_image = CenterCrop(meta['image']['h'], meta['image']['w']).apply(stitched_image)
    return stitched_image


def stitch_tiles_at_path(input_path, overwrite):
    output_path = Path(str(input_path) + '_stitched')
    if overwrite:
        rmtree(output_path, ignore_errors=True)
    output_path.mkdir()

    for root, dirs, files in os.walk(input_path):
        if not dirs:
            tiles_path = Path(root)
            group_path = tiles_path.parent
            group = group_path.name
            print(f'Stitching tiles at {tiles_path}')

            meta = json.load(open(group_path / 'meta.json'))
            stitched_image = stitch_crop_tiles(tiles_path, 512, meta)

            stitched_group_path = output_path / group
            stitched_group_path.mkdir(exist_ok=True)
            stitched_image_path = stitched_group_path / (tiles_path.name + '.png')
            cv2.imwrite(str(stitched_image_path), stitched_image)
            print(f'Saved stitched image to {stitched_image_path}')


def clean_dir(path):
    rmtree(path, ignore_errors=True)
    Path(path).mkdir(parents=True)
