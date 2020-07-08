import logging
import os
import json

import cv2
from albumentations import CenterCrop

from am.segment.image_utils import pad_slice_image, compute_tile_row_col_n, stitch_tiles, clip, \
    normalize, overlay_source_mask, save_rgb_image
from am.utils import clean_dir, read_image

logger = logging.getLogger('am-segm')


def rename_image(input_image_path):
    out_image_stem = input_image_path.stem
    if out_image_stem not in ['source', 'mask']:
        out_image_stem = 'source'
    output_image_path = input_image_path.parent / f'{out_image_stem}.tiff'
    if input_image_path != output_image_path:
        logger.info(f'Renaming image: {input_image_path} -> {output_image_path}')
        os.rename(input_image_path, output_image_path)
    return output_image_path


def normalize_source(input_group_path, output_group_path, q1=1, q2=99):
    logger.info(f'Normalizing images at {input_group_path}')
    for image_path in input_group_path.glob('*.tif*'):
        if image_path.name != 'mask':
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            image_norm = normalize(clip(image, q1, q2))

            output_group_path.mkdir(parents=True, exist_ok=True)
            image_norm_path = output_group_path / 'source.tiff'
            logger.debug(f'Saving normalized image to {image_norm_path}')
            cv2.imwrite(str(image_norm_path), image_norm)
            break  # use first non mask tiff image as source


def slice_to_tiles(input_group_path, output_group_path):
    tile_size = 512
    max_size = tile_size * 40
    image_path = input_group_path / 'source.tiff'
    logger.info(f'Slicing {image_path}')

    image = read_image(image_path)

    orig_h, orig_w = map(int, image.shape)
    meta = {'orig_image': {'h': orig_h, 'w': orig_w}}

    if max(image.shape) > max_size:
        factor = max_size / max(image.shape)
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

    image_tiles_path = output_group_path / image_path.stem
    image_tiles_path.mkdir(parents=True, exist_ok=True)

    tile_row_n, tile_col_n = compute_tile_row_col_n(image.shape, tile_size)
    target_size = (tile_row_n * tile_size, tile_col_n * tile_size)
    tiles = pad_slice_image(image, tile_size, target_size)

    h, w = map(int, image.shape)
    meta['image'] = {'h': h, 'w': w}
    meta['tile'] = {'rows': tile_row_n, 'cols': tile_col_n, 'size': tile_size}
    json.dump(meta, open(output_group_path / 'meta.json', 'w'))

    for i, tile in enumerate(tiles):
        tile_path = image_tiles_path / f'{i:04}.png'
        logger.debug(f'Save tile: {tile_path}')
        cv2.imwrite(str(tile_path), tile)


def stitch_and_crop_tiles(tiles_path, tile_size, meta):
    tile_paths = sorted(tiles_path.glob('*.png'))
    if len(tile_paths) != meta['tile']['rows'] * meta['tile']['cols']:
        logger.warning(f'Number of tiles does not match meta: {len(tile_paths)}, {meta}')

    tiles = [None] * len(tile_paths)
    for path in tile_paths:
        i = int(path.stem)
        tiles[i] = cv2.imread(str(path))[:, :, 0]  # because ch0==ch1==ch2

    stitched_image = stitch_tiles(tiles, tile_size, meta['tile']['rows'], meta['tile']['cols'])
    stitched_image = CenterCrop(meta['image']['h'], meta['image']['w']).apply(stitched_image)
    return stitched_image


def stitch_tiles_at_path(input_group_path, output_group_path, image_ext='png'):
    logger.info(f'Stitching tiles at {input_group_path}')

    meta = json.load(open(input_group_path / 'meta.json'))
    for image_type in ['source', 'mask']:
        if (input_group_path / image_type).exists():
            stitched_image = stitch_and_crop_tiles(input_group_path / image_type, 512, meta)
            if stitched_image.max() <= 1:
                stitched_image *= 255

            output_group_path.mkdir(parents=True, exist_ok=True)
            stitched_image_path = output_group_path / f'{image_type}.{image_ext}'
            cv2.imwrite(str(stitched_image_path), stitched_image)
            logger.info(f'Saved stitched image to {stitched_image_path}')


def overlay_images_with_masks(input_group_path, image_ext='png'):
    logger.info(f'Overlaying images at {input_group_path}')
    source = read_image(str(input_group_path / f'source.{image_ext}'))
    mask = read_image(str(input_group_path / f'mask.{image_ext}'))
    assert source.shape == mask.shape
    overlay = overlay_source_mask(source, mask)
    save_rgb_image(overlay, input_group_path / f'overlay.{image_ext}')
