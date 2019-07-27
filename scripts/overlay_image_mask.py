import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from am_segm.utils import read_image, plot_overlay, overlay_group

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str)
    parser.add_argument('--groups', default='', type=str)
    args = parser.parse_args()
    input_path = Path(args.input)
    if args.groups:
        groups = args.groups.split(',')
    else:
        groups = [group_path.name for group_path in input_path.iterdir()]

    for group in groups:
        overlay_group(input_path / group)
