import argparse
import os
from pathlib import Path
from shutil import rmtree
import json

import cv2

from am_segm.image_utils import pad_slice_image, compute_tile_row_col_n
from am_segm.utils import read_image, clean_dir


def convert_to_tiles(input_data_path, overwrite=False):
    print('Converting images to tiles')

    tiles_path = input_data_path.parent / (input_data_path.stem + '_tiles')
    tiles_path.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for root, dirs, files in os.walk(input_data_path):
        if not dirs:
            for f in files:
                image_paths.append(Path(root) / f)

    tile_size = 512
    for image_path in image_paths:
        print(f'Splitting {image_path}')

        image = read_image(image_path)

        image_tiles_path = tiles_path / image_path.parent.name / image_path.stem
        if image_tiles_path.exists():
            if overwrite:
                clean_dir(image_tiles_path)
            else:
                print(f'Already exists: {image_tiles_path}')
        else:
            image_tiles_path.mkdir(parents=True)

        tile_row_n, tile_col_n = compute_tile_row_col_n(image.shape, tile_size)
        target_size = (tile_row_n * tile_size, tile_col_n * tile_size)
        tiles = pad_slice_image(image, tile_size, target_size)

        h, w = map(int, image.shape)
        meta = {
            'image': {'h': h, 'w': w},
            'tile': {'rows': tile_row_n, 'cols': tile_col_n, 'size': tile_size}
        }
        group_path = image_tiles_path.parent
        json.dump(meta, open(group_path / 'meta.json', 'w'))

        for i, tile in enumerate(tiles):
            tile_path = image_tiles_path / f'{i:03}.png'
            print(f'Save tile: {tile_path}')
            cv2.imwrite(str(tile_path), tile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str)
    parser.add_argument('--overwrite', dest='overwrite', action='store_true')
    args = parser.parse_args()
    input_path = Path(args.input)

    convert_to_tiles(input_path, args.overwrite)
