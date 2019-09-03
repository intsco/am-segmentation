import argparse
import os
from math import ceil
from pathlib import Path
import logging
from uuid import uuid4

from am.ecs import (
    upload_images_to_s3,
    run_wait_for_inference_task,
    download_images_from_s3,
    time_it,
    INFERENCE_BATCH_SIZE,
    remove_images_from_s3,
    list_images_on_s3,
)
from am.logger import init_logger
from am.register import register_ablation_marks
from am.segment.preprocess import slice_to_tiles, stitch_tiles_at_path
from am.segment.utils import overlay_images_with_masks

logger = logging.getLogger('am-segm')


@time_it
def run_am_pipeline(data_path, acq_grid_shape):
    slice_to_tiles(data_path / 'source')

    for group_path in (data_path / 'source_tiles').iterdir():
        logger.info(f'Inference of group path: {group_path}')

        source_tiles_path = Path(group_path / 'source')

        prefix = str(uuid4())
        # prefix = '29e91cf4-4e38-480b-9d11-48f924126439'
        s3_paths = upload_images_to_s3(
            source_tiles_path, os.environ['INPUT_BUCKET'], prefix, os.environ['QUEUE_URL']
        )

        def stop_callback():
            pred_s3_paths = list_images_on_s3(os.environ['OUTPUT_BUCKET'], prefix)
            logger.debug(f'Predicted {len(pred_s3_paths)}/{len(s3_paths)} images')
            return len(pred_s3_paths) == len(s3_paths)

        task_n = min(ceil(len(s3_paths) / INFERENCE_BATCH_SIZE), 10)
        run_wait_for_inference_task(
            stop_callback, task_n=task_n, sleep_interval=10, timeout=5 * 60
        )

        mask_tiles_path = Path(group_path / 'mask')
        download_images_from_s3(os.environ['OUTPUT_BUCKET'], s3_paths, mask_tiles_path)

        remove_images_from_s3(os.environ['INPUT_BUCKET'], prefix)
        remove_images_from_s3(os.environ['OUTPUT_BUCKET'], prefix)

    stitch_tiles_at_path(
        input_path=data_path / 'source_tiles',
        meta_path=data_path / 'source_tiles',
        overwrite=True,
        image_ext='tiff'
    )

    overlay_images_with_masks(data_path / 'source_tiles_stitched', image_ext='tiff')

    for group_path in (data_path / 'source').iterdir():
        group = group_path.name
        register_ablation_marks(
            mask_path=data_path / 'source_tiles_stitched' / group / 'mask.tiff',
            meta_path=data_path / 'source_tiles' / group / 'meta.json',
            am_coord_path=data_path / 'am_coords' / group / 'am_coordinates.npy',
            acq_grid_shape=acq_grid_shape
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run AM segmentation pipeline')
    parser.add_argument('data_path')
    parser.add_argument('--rows', type=int)
    parser.add_argument('--cols', type=int)
    parser.add_argument('--debug', dest='debug', action='store_true')
    args = parser.parse_args()

    init_logger(logging.DEBUG if args.debug else logging.INFO)

    run_am_pipeline(data_path=Path(args.data_path), acq_grid_shape=(args.rows, args.cols))
