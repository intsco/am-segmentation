import argparse
from functools import partial
from pathlib import Path

from am.logger import init_logger
from am.segment.preprocess import slice_to_tiles, normalize_source
from am.utils import iterate_groups

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('ds_path', type=str)
    parser.add_argument('groups', nargs='*')
    args = parser.parse_args()
    init_logger()

    source_path = Path(args.ds_path) / 'source'
    source_norm_path = source_path.parent / 'source_norm'
    tiles_path = source_path.parent / 'tiles'

    iterate_groups(source_path, source_norm_path, groups=args.groups, func=normalize_source, )
    iterate_groups(source_norm_path, tiles_path, groups=args.groups, func=slice_to_tiles)
