import argparse
from pathlib import Path
from am.segm.preprocess import stitch_tiles_at_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str)
    parser.add_argument('--overwrite', dest='overwrite', action='store_true')
    args = parser.parse_args()

    input_path = Path(args.input)
    meta_path = input_path.parent / 'source_tiles'
    stitch_tiles_at_path(input_path, meta_path, args.overwrite)
