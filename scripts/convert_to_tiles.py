import argparse
import os
from pathlib import Path
from shutil import rmtree

import cv2

from am_segm.image_utils import pad_slice_image


def convert_to_tiles(input_data_path, output_data_path):
    print('Converting images to tiles')

    rmtree(output_data_path)
    output_data_path.mkdir(parents=True)

    image_paths = []
    for path, dirs, files in os.walk(input_data_path):
        if not dirs:
            for f in files:
                image_paths.append(Path(path) / f)

    tile_size = 512
    for image_path in image_paths:
        print(f'Splitting {image_path}')
        tiles_path = Path(str(image_path).replace(str(input_data_path), str(output_data_path)))
        tiles_path = tiles_path.parent / tiles_path.stem
        tiles_path.mkdir(parents=True, exist_ok=True)

        source_image = cv2.imread(str(image_path))[:,:,0]  # because ch0==ch1==ch2
        tiles = pad_slice_image(source_image, tile_size)

        for i, tile in enumerate(tiles):
            tile_path = tiles_path / f'{i}.png'
            print(f'Save tile: {tile_path}')
            cv2.imwrite(str(tile_path), tile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    convert_to_tiles(input_path, output_path)
