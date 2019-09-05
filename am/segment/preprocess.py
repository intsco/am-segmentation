import logging
import os
from pathlib import Path
from shutil import rmtree
import json

import cv2
from albumentations import CenterCrop

from am.segment.image_utils import pad_slice_image, compute_tile_row_col_n, stitch_tiles
from am.segment.utils import read_image
from am.utils import clean_dir

logger = logging.getLogger('am-segm')


def slice_to_tiles(input_data_path, overwrite=False):
    logger.info('Converting images to tiles')

    tiles_path = input_data_path.parent / (input_data_path.stem + '_tiles')
    if tiles_path.exists():
        if overwrite:
            clean_dir(tiles_path)
        else:
            logger.info('Tiles already exist. Skipping')
            return

    tiles_path.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for root, dirs, files in os.walk(input_data_path):
        if not dirs:
            for f in files:
                image_paths.append(Path(root) / f)

    tile_size = 512
    max_size = tile_size * 15
    for image_path in image_paths:
        logger.info(f'Slicing {image_path}')

        image = read_image(image_path)

        orig_h, orig_w = map(int, image.shape)
        meta = {'orig_image': {'h': orig_h, 'w': orig_w}}

        if max(image.shape) > max_size:
            factor = max_size / max(image.shape)
            image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

        image_tiles_path = tiles_path / image_path.parent.name / image_path.stem
        if image_tiles_path.exists():
            if overwrite:
                clean_dir(image_tiles_path)
            else:
                logger.debug(f'Already exists: {image_tiles_path}')
        else:
            image_tiles_path.mkdir(parents=True)

        tile_row_n, tile_col_n = compute_tile_row_col_n(image.shape, tile_size)
        target_size = (tile_row_n * tile_size, tile_col_n * tile_size)
        tiles = pad_slice_image(image, tile_size, target_size)

        h, w = map(int, image.shape)
        meta['image'] = {'h': h, 'w': w}
        meta['tile'] = {'rows': tile_row_n, 'cols': tile_col_n, 'size': tile_size}
        group_path = image_tiles_path.parent
        json.dump(meta, open(group_path / 'meta.json', 'w'))

        for i, tile in enumerate(tiles):
            tile_path = image_tiles_path / f'{i:03}.png'
            logger.debug(f'Save tile: {tile_path}')
            cv2.imwrite(str(tile_path), tile)


def stitch_and_crop_tiles(tiles_path, tile_size, meta):
    tile_paths = sorted(tiles_path.glob('*.png'))
    if len(tile_paths) != meta['tile']['rows'] * meta['tile']['cols']:
        logger.warning(f'Number of tiles does not match meta: {len(tile_paths)}, {meta}')

    tiles = [None] * len(tile_paths)
    for path in tile_paths:
        i = int(path.stem)
        tiles[i] = cv2.imread(str(path))[:,:,0]  # because ch0==ch1==ch2

    stitched_image = stitch_tiles(tiles, tile_size, meta['tile']['rows'], meta['tile']['cols'])
    stitched_image = CenterCrop(meta['image']['h'], meta['image']['w']).apply(stitched_image)
    return stitched_image


def stitch_tiles_at_path(input_path, meta_path, overwrite=False, image_ext='png'):
    output_path = Path(str(input_path) + '_stitched')
    if overwrite:
        rmtree(output_path, ignore_errors=True)
    output_path.mkdir(exist_ok=True)

    for group_path in input_path.iterdir():
        logger.info(f'Stitching tiles at {group_path}')
        group = group_path.name

        meta = json.load(open(meta_path / group / 'meta.json'))
        for image_type in ['source', 'mask']:
            stitched_image = stitch_and_crop_tiles(group_path / image_type, 512, meta)
            if stitched_image.max() <= 1:
                stitched_image *= 255

            stitched_group_path = output_path / group
            stitched_group_path.mkdir(exist_ok=True)

            stitched_image_path = stitched_group_path / f'{image_type}.{image_ext}'
            cv2.imwrite(str(stitched_image_path), stitched_image)
            logger.info(f'Saved stitched image to {stitched_image_path}')
