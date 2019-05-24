import cv2
import numpy as np
from albumentations import PadIfNeeded


def get_n_splits(size, tile_size):
    n_splits, rest = np.divmod(size, tile_size)
    if rest > 0:
        n_splits += 1
    return n_splits


def slice_image(image, tile_size):
    n_splits = image.shape[0] // tile_size
    tiles = []
    for i in range(n_splits):
        for j in range(n_splits):
            tile = image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]
            tiles.append(tile)
    return tiles


def pad_image(image, size):
    aug = PadIfNeeded(min_height=size, min_width=size,
                      border_mode=cv2.BORDER_REFLECT_101, p=1.0)
    aug_image = aug(image=image)['image']
    pads = (aug_image.shape[0] - image.shape[0],
            aug_image.shape[1] - image.shape[1])
    return aug_image, pads


def pad_slice_image(image, tile_size):
    n_splits = get_n_splits(max(image.shape[:2]), tile_size)
    full_size = tile_size * n_splits
    # source_image_n_slices[str(image_dir)] = n_splits * n_splits
    source_image, _ = pad_image(image, full_size)
    tiles = slice_image(source_image, tile_size)
    return tiles


def slice_images(image_dirs, tile_size):
    images, masks = [], []
    source_image_padding = {}
    source_image_n_slices = {}
    for image_dir in image_dirs:
        print(image_dir)
        # n_splits = get_n_splits(max(source_image.shape[:2]), tile_size)
        # full_size = tile_size * n_splits
        # source_image_n_slices[str(image_dir)] = n_splits * n_splits
        # source_image, pads = pad_source_image(source_image, full_size)
        # source_image_padding[str(image_dir)] = pads

        images += pad_slice_image(image_dir / 'source.png', tile_size)
        if (image_dir / 'mask.png').exists():
            masks += pad_slice_image(image_dir / 'mask.png', tile_size)

    return images, masks, source_image_padding, source_image_n_slices


def combine_tiles(tiles, tile_size, n_splits):
    image = np.zeros((tile_size * n_splits, tile_size * n_splits))
    for i in range(n_splits):
        for j in range(n_splits):
            tile = tiles[i * n_splits + j]
            image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = tile
    return image


def remove_padding(image, row_pad, col_pad):
    top_pad = row_pad // 2
    bottom_pad = row_pad - top_pad
    left_pad = col_pad // 2
    right_pad = col_pad - left_pad
    full_size = image.shape[:2]
    return image[top_pad:(full_size[0] - bottom_pad), left_pad:(full_size[1] - right_pad)]