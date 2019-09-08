import argparse
import os
from math import ceil
from pathlib import Path
import logging
from uuid import uuid4

import yaml

from am.logger import init_logger
from am.register import register_ablation_marks
from am.segment.preprocess import slice_to_tiles, stitch_tiles_at_path
from am.segment.utils import overlay_images_with_masks
from am.utils import time_it

logger = logging.getLogger('am-segm')


@time_it
def run_am_pipeline(data_path, acq_grid_shape, config):
    slice_to_tiles(data_path / 'source')

    prefix = str(uuid4())
    local_input_paths, s3_paths = [], []
    for group_path in (data_path / 'source_tiles').iterdir():
        logger.info(f'Inference of group path: {group_path}')
        for local_path in Path(group_path / 'source').iterdir():
            local_input_paths.append(local_path)
            s3_paths.append(f'{prefix}/{group_path.name}/{local_path.name}')

    upload_images_to_s3(
        local_paths=local_input_paths,
        bucket=config['aws']['input_bucket'],
        s3_paths=s3_paths,
        queue_url=config['aws']['queue_url']
    )

    def stop_callback():
        pred_s3_paths = list_images_on_s3(config['aws']['output_bucket'], prefix)
        logger.debug(f'Predicted {len(pred_s3_paths)}/{len(s3_paths)} images')
        return len(pred_s3_paths) == len(s3_paths)

    task_n = min(ceil(len(s3_paths) / INFERENCE_BATCH_SIZE), 10)
    task_config = dict(count=task_n, **config['aws']['ecs'])
    run_wait_for_inference_task(
        task_config, stop_callback, sleep_interval=10, timeout=10 * 60
    )

    local_output_paths = []
    for s3_path in s3_paths:
        _, group, fname = s3_path.split('/')
        path = data_path / 'source_tiles' / group / 'mask' / fname
        local_output_paths.append(path)
    download_images_from_s3(
        bucket=config['aws']['output_bucket'], s3_paths=s3_paths, local_paths=local_output_paths
    )

    remove_images_from_s3(config['aws']['input_bucket'], prefix)
    remove_images_from_s3(config['aws']['output_bucket'], prefix)

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

    config = yaml.load(open('config/config.yml'))
    os.environ['AWS_ACCESS_KEY_ID'] = config['aws']['aws_access_key_id']
    os.environ['AWS_SECRET_ACCESS_KEY'] = config['aws']['aws_secret_access_key']
    os.environ['AWS_DEFAULT_REGION'] = config['aws']['aws_default_region']

    from am.ecs import (
        upload_images_to_s3,
        run_wait_for_inference_task,
        download_images_from_s3,
        INFERENCE_BATCH_SIZE,
        remove_images_from_s3,
        list_images_on_s3,
    )

    run_am_pipeline(
        data_path=Path(args.data_path),
        acq_grid_shape=(args.rows, args.cols),
        config=config
    )