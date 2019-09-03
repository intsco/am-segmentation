import argparse
from pathlib import Path

from am.logger import init_logger
from am.segment.preprocess import slice_to_tiles

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str)
    parser.add_argument('--overwrite', dest='overwrite', action='store_true')
    args = parser.parse_args()
    input_path = Path(args.input)

    init_logger()

    slice_to_tiles(input_path, args.overwrite)
