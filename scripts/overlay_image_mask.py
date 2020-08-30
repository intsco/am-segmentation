import argparse
from functools import partial
from pathlib import Path

from am.segment.preprocess import overlay_images_with_masks
from am.utils import iterate_groups

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('ds_path', type=str)
    parser.add_argument('groups', nargs='*')
    args = parser.parse_args()

    iterate_groups(
        Path(args.ds_path) / 'tiles_stitched',
        groups=args.groups,
        func=partial(overlay_images_with_masks, image_ext='tiff')
    )
