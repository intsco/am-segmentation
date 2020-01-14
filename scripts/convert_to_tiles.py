import argparse
from pathlib import Path

from am.logger import init_logger
from am.segment.preprocess import slice_to_tiles, normalize_source

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str)
    parser.add_argument('--overwrite', dest='overwrite', action='store_true')
    args = parser.parse_args()

    init_logger()

    source_path = Path(args.input)
    source_norm_path = source_path.parent / 'source_norm'
    tiles_path = source_path.parent / 'tiles'
    normalize_source(source_path, source_norm_path, q1=1, q2=99)
    slice_to_tiles(source_norm_path, tiles_path, args.overwrite)
