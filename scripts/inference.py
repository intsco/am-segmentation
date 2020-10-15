import argparse
from functools import partial
from math import ceil
from pathlib import Path
import logging
from uuid import uuid4
from time import time, sleep
from collections import Counter

import boto3

from am.logger import init_logger
from am.register import register_ablation_marks
from am.segment.preprocess import slice_to_tiles, stitch_tiles_at_path, overlay_images_with_masks, \
    normalize_source
from am.utils import time_it, iterate_groups, find_all_groups
from am.ecs import (
    upload_images_to_s3,
    download_images_from_s3,
    INFERENCE_BATCH_SIZE,
    remove_images_from_s3,
    list_images_on_s3,
)
from am.config import Config

logger = logging.getLogger('am-segm')


def upload_to_s3(input_path, groups, prefix):
    logger.info(f'Uploading {input_path} to s3')
    local_input_paths, s3_paths = [], []
    for group_path in input_path.iterdir():
        if group_path.name in groups:
            for local_path in Path(group_path / 'source').iterdir():
                local_input_paths.append(local_path)
                s3_paths.append(f'{prefix}/{group_path.name}/{local_path.name}')

    upload_images_to_s3(
        local_paths=local_input_paths,
        bucket=config['input_bucket'],
        s3_paths=s3_paths,
        queue_url=config['queue_url'],
    )
    return s3_paths


@time_it
def run_inference(s3_paths, prefix):
    def stop_callback():
        pred_s3_paths = list_images_on_s3(config['output_bucket'], prefix)
        logger.debug(f'Predicted {len(pred_s3_paths)}/{len(s3_paths)} images')
        return len(pred_s3_paths) == len(s3_paths)

    ecs = boto3.client('ecs')
    ecs_max_task_n = 10

    task_arns = []
    task_n = min(ceil(len(s3_paths) / INFERENCE_BATCH_SIZE), 20)
    while task_n > 0:
        ecs_task_n = min(task_n, ecs_max_task_n)

        logger.info(f'Running {ecs_task_n} tasks in ECS')
        resp = ecs.run_task(count=ecs_task_n, **config.task_config())
        task_arns += [t['taskArn'] for t in resp['tasks']]
        task_n -= ecs_task_n
        if task_n > 0:
            sleep(5)

    sleep_interval = 10
    timeout = 5 * 60
    finish = time() + timeout
    while time() < finish:
        logger.debug(f'Waiting for {sleep_interval}s')
        sleep(sleep_interval)
        resp = ecs.describe_tasks(cluster='am-segm', tasks=task_arns)
        task_statuses = [t['lastStatus'] for t in resp['tasks']]
        logger.debug(f'Task statuses: {Counter(task_statuses)}')

        if stop_callback():
            break
    else:
        raise Exception(f'Timeout: {timeout}s')


def download_from_s3(s3_paths, data_path):
    local_output_paths = []
    for s3_path in s3_paths:
        _, group, fname = s3_path.split('/')
        path = data_path / group / 'mask' / fname
        local_output_paths.append(path)
    download_images_from_s3(
        bucket=config['output_bucket'], s3_paths=s3_paths, local_paths=local_output_paths
    )


def register_ablation_marks_at_path(data_path, groups, acq_grid_shape):
    for group_path in (data_path / 'source').iterdir():
        try:
            group = group_path.name
            if group in groups:
                register_ablation_marks(
                    source_path=data_path / 'source_norm' / group / 'source.tiff',
                    mask_path=data_path / 'tiles_stitched' / group / 'mask.tiff',
                    meta_path=data_path / 'tiles' / group / 'meta.json',
                    am_coord_path=data_path / 'am_coords' / group / 'am_coordinates.png',
                    overlay_path=data_path / 'am_coords' / group / 'overlay.png',
                    acq_grid_shape=acq_grid_shape,
                )
        except Exception as e:
            logger.error(f'Failed to register AM marks at path {group_path}', exc_info=True)


@time_it
def run_am_pipeline(data_path, groups, acq_grid_shape, tile_size, register):
    if not groups:
        groups = find_all_groups(data_path)

    iterate_groups(
        data_path / 'source', data_path / 'source_norm', groups=groups, func=normalize_source
    )
    iterate_groups(
        data_path / 'source_norm',
        data_path / 'tiles',
        groups=groups,
        func=partial(slice_to_tiles, tile_size=tile_size)
    )

    prefix = str(uuid4())
    s3_paths = upload_to_s3(data_path / 'tiles', groups, prefix)
    run_inference(s3_paths, prefix)
    download_from_s3(s3_paths, data_path / 'tiles')

    remove_images_from_s3(config['input_bucket'], prefix)
    remove_images_from_s3(config['output_bucket'], prefix)

    iterate_groups(
        data_path / 'tiles',
        data_path / 'tiles_stitched',
        groups=groups,
        func=partial(stitch_tiles_at_path, tile_size=tile_size, image_ext='tiff')
    )
    iterate_groups(
        data_path / 'tiles_stitched',
        groups=groups,
        func=partial(overlay_images_with_masks, image_ext='tiff')
    )

    if register:
        register_ablation_marks_at_path(data_path, groups, acq_grid_shape)


def parse_args():
    parser = argparse.ArgumentParser(description='Run AM segmentation pipeline')
    parser.add_argument('ds_path', type=str, help='Dataset directory path')
    parser.add_argument('groups', nargs='*')
    parser.add_argument('--tile-size', type=int, default=512)
    parser.add_argument('--rows', type=int)
    parser.add_argument('--cols', type=int)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-register', dest='register', action='store_false')

    args = parser.parse_args()
    if args.register:
        assert args.rows and args.cols

    return args


if __name__ == '__main__':
    args = parse_args()

    init_logger(logging.DEBUG if args.debug else logging.INFO)
    config = Config('config/config.yml')

    boto3.client('sqs').create_queue(
        QueueName=config['queue_name'], Attributes={'MessageRetentionPeriod': '3600'}
    )

    run_am_pipeline(
        data_path=Path(args.ds_path),
        groups=args.groups,
        acq_grid_shape=(args.rows, args.cols),
        tile_size=args.tile_size,
        register=args.register
    )
