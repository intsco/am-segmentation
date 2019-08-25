import cv2
import numpy as np
from albumentations import PadIfNeeded


def compute_tile_row_col_n(shape, tile_size):
    row_n, rest = divmod(max(shape), tile_size)
    if rest > 0:
        row_n += 1
    return row_n, row_n


def slice_image(image, tile_size):
    tile_row_n = tile_col_n = max(image.shape) // tile_size  # image size must be multiple of tile size
    tiles = []
    for i in range(tile_row_n):
        for j in range(tile_col_n):
            tile = image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]
            tiles.append(tile)
    return tiles


def pad_image(image, target_shape):
    h, w = target_shape
    aug = PadIfNeeded(min_height=h, min_width=w,
                      border_mode=cv2.BORDER_REFLECT_101, p=1.0)
    padded_image = aug(image=image)['image']
    return padded_image


def pad_slice_image(image, tile_size, target_size):
    padded_image = pad_image(image, target_size)
    tiles = slice_image(padded_image, tile_size)
    return tiles


def stitch_tiles(tiles, tile_size, tile_row_n, tile_col_n):
    image = np.zeros((tile_size * tile_row_n, tile_size * tile_col_n))
    for i in range(tile_row_n):
        for j in range(tile_col_n):
            tile = tiles[i * tile_col_n + j]
            image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = tile
    return image

