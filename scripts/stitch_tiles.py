import argparse
from functools import partial
from pathlib import Path

from am.logger import init_logger
from am.segment.preprocess import stitch_tiles_at_path
from am.utils import iterate_groups

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('ds_path', type=str)
    parser.add_argument('groups', nargs='*')
    args = parser.parse_args()
    init_logger()

    iterate_groups(
        Path(args.ds_path) / 'tiles',
        Path(args.ds_path) / 'tiles_stitched',
        groups=args.groups,
        func=partial(stitch_tiles_at_path, image_ext='tiff')
    )
