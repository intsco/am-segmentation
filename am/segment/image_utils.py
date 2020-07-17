import os
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from albumentations import PadIfNeeded

logger = logging.getLogger('am-segm')


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
    image = np.zeros(
        (tile_size * tile_row_n, tile_size * tile_col_n), dtype=np.uint8
    )
    for i in range(tile_row_n):
        for j in range(tile_col_n):
            tile = tiles[i * tile_col_n + j]
            image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = tile
    return image


def overlay_source_mask(source: np.ndarray, mask: np.ndarray, alpha: float = 0.3) -> Image:
    mask_3ch = np.array(Image.fromarray(mask).convert('RGB'))
    mask_3ch[mask > 127] = [255, 255, 0]  # make ablation marks yellow
    return Image.blend(
        Image.fromarray(source).convert('RGB'),
        Image.fromarray(mask_3ch),
        alpha
    )


def save_rgb_image(image, path):
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def overlay_tiles(input_path: Path):
    logger.info(f'Overlaying images at {input_path}')

    source_path, mask_path, overlay_path = [
        input_path / img_type for img_type in ['source', 'mask', 'overlay']
    ]

    if source_path.exists() and mask_path.exists():
        overlay_path.mkdir(exist_ok=True)
        for source_tile_path in source_path.iterdir():
            image_fn = source_tile_path.name
            logger.debug(f'Overlaying {image_fn}')

            source_tile = cv2.imread(str(source_tile_path), cv2.IMREAD_GRAYSCALE)

            mask_tile_path = mask_path / image_fn
            mask_tile = cv2.imread(str(mask_tile_path), cv2.IMREAD_GRAYSCALE)

            overlay = overlay_source_mask(source_tile, mask_tile)
            save_rgb_image(overlay, overlay_path / image_fn)
    else:
        for subdir_path in Path(input_path).iterdir():
            overlay_tiles(subdir_path)


def normalize(img):
    return np.uint8((img - img.min()) / (img.max() - img.min()) * 255)


def clip(img, q1=1, q2=99):
    return np.clip(img, a_min=np.percentile(img, q=q1), a_max=np.percentile(img, q=q2))
