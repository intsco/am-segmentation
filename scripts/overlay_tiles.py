import argparse
from pathlib import Path

from am.logger import init_logger
from am.segment.image_utils import overlay_tiles

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input',
        type=str,
        help='Path to directory that contains source and mask tiles at some level of depth'
    )
    args = parser.parse_args()

    init_logger()
    overlay_tiles(Path(args.input))
