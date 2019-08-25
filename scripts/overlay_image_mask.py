import argparse
from pathlib import Path

from am.segm.utils import overlay_images_with_masks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str)
    parser.add_argument('--groups', default='', type=str)
    args = parser.parse_args()

    input_path = Path(args.input)
    overlay_images_with_masks(input_path)
