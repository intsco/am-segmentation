import argparse
from pathlib import Path

from am.logger import init_logger
from am.register import register_ablation_marks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('ds_path', type=str)
    parser.add_argument('groups', nargs='*')
    args = parser.parse_args()

    init_logger()

    groups = args.groups
    if not groups:
        groups = [p.name for p in (Path(args.ds_path) / 'tiles_stitched').iterdir()]

    for group in groups:
        source_path = Path(args.ds_path) / 'source_norm' / group / 'source.tiff'
        mask_path = Path(args.ds_path) / 'tiles_stitched' / group / 'mask.tiff'
        meta_path = Path(args.ds_path) / 'tiles' / group / 'meta.json'
        am_coord_path = Path(args.ds_path) / 'am_coords' / group / 'am_coordinates.npy'
        overlay_path = Path(args.ds_path) / 'am_coords' / group / 'overlay.png'

        register_ablation_marks(
            source_path=source_path,
            mask_path=mask_path,
            meta_path=meta_path,
            am_coord_path=am_coord_path,
            overlay_path=overlay_path,
        )
