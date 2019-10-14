import argparse
from pathlib import Path

from am.logger import init_logger
from am.segment.image_utils import overlay_tiles

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        'path', type=str, help='Path to directory with source and mask subdirectories'
    )
    args = parser.parse_args()

    init_logger()
    overlay_tiles(Path(args.path))
