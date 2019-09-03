import argparse
from pathlib import Path

import numpy as np

from am.logger import init_logger
from am.register import load_mask, register_ablation_marks, export_am_coordinates


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('mask_path', type=str)
    parser.add_argument('meta_path', type=str)
    parser.add_argument('am_coord_path', type=str)
    parser.add_argument('--rows', type=int)
    parser.add_argument('--cols', type=int)
    args = parser.parse_args()

    mask_path = Path(args.mask_path)
    meta_path = Path(args.meta_path)
    am_coord_path = Path(args.am_coord_path)
    acq_grid_shape = (args.rows, args.cols)

    init_logger()
    register_ablation_marks(mask_path, meta_path, am_coord_path, acq_grid_shape)
