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


def overlay_source_mask(source: np.ndarray, mask: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    if max(source.shape) > max(mask.shape):
        new_shape = source.shape
    else:
        new_shape = mask.shape
    new_size = (new_shape[1], new_shape[0])

    mask_3ch = np.array(Image.fromarray(mask).convert('RGB'))
    mask_3ch[mask > 127] = [255, 0, 0]  # make ablation marks red
    overlay = Image.blend(
        Image.fromarray(source).resize(new_size).convert('RGB'),
        Image.fromarray(mask_3ch).resize(new_size),
        alpha
    )
    return np.array(overlay)


def read_image(path: Path, ch_n: int = None) -> np.ndarray:
    if not Path(path).exists():
        raise Exception(f'Image file not found: {path}')

    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if ch_n is None:
        ch1, ch2, ch3 = img.sum(axis=(0, 1))
        ch_n = 1 if ch1 == ch2 == ch3 else 3

    if ch_n == 1:
        return img[:, :, 0]  # because ch0==ch1==ch2
    else:
        return img


def save_rgb_image(array, path):
    """@Deprecated"""
    save_image(array, path)


def save_image(array: np.ndarray, path: Path):
    if array.ndim == 3:
        res = cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    else:
        res = cv2.imwrite(str(path), array)
    assert res, f'Failed to save {path}'


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

            source_tile = read_image(source_tile_path)

            mask_tile_path = mask_path / image_fn
            mask_tile = read_image(mask_tile_path)

            overlay = overlay_source_mask(source_tile, mask_tile)
            save_rgb_image(np.array(overlay), overlay_path / image_fn)
    else:
        for subdir_path in Path(input_path).iterdir():
            overlay_tiles(subdir_path)


def normalize(img):
    """Axes order: [h, w, ch] or [h, w]"""
    return np.uint8(
        (img - img.min(axis=(0, 1))) / (img.max(axis=(0, 1)) - img.min(axis=(0, 1))) * 255
    )


def clip(img, q1=1, q2=99):
    return np.clip(img, a_min=np.percentile(img, q=q1), a_max=np.percentile(img, q=q2))
