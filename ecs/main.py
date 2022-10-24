#!/usr/bin/env python

import logging
import os
import tarfile
from pathlib import Path

import boto3

from am.ecs import (
    consume_messages,
    download_images_from_s3,
    predict,
    save_predictions,
    upload_images_to_s3,
    delete_messages,
    INFERENCE_BATCH_SIZE,
)
from am.logger import init_logger
from am.utils import clean_dir, load_model, time_it

local_inputs_dir = Path('/tmp/inputs')
local_outputs_dir = Path('/tmp/outputs')

logger = logging.getLogger('am-segm')


def create_model():
    logger.info(f'Downloading model from {os.environ["MODEL_PATH"]}')
    bucket, key = os.environ['MODEL_PATH'].replace('s3://', '').split('/', 1)
    fname = Path(key).name
    boto3.client('s3').download_file(bucket, key, fname)
    if fname.endswith('.tar.gz'):
        with tarfile.open(fname, 'r:gz') as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f)
    return load_model('model.pt')


@time_it
def run_inference():
    try:
        logger.info('Batch inference of AM images')
        model = create_model()

        total_n = 0
        while True:
            s3_paths, receipt_handles = consume_messages(
                queue_url=os.environ['QUEUE_URL'], n=INFERENCE_BATCH_SIZE
            )
            if not s3_paths:
                logger.info('No more messages in the queue. Exiting')
                break
            total_n += len(s3_paths)

            local_input_paths = [local_inputs_dir / s3_path for s3_path in s3_paths]
            download_images_from_s3(
                bucket=os.environ['INPUT_BUCKET'], s3_paths=s3_paths, local_paths=local_input_paths
            )

            predictions = predict(model, local_input_paths)
            local_output_paths = [local_outputs_dir / s3_path for s3_path in s3_paths]
            save_predictions(predictions, local_output_paths)

            upload_images_to_s3(
                local_paths=local_output_paths, bucket=os.environ['OUTPUT_BUCKET'], s3_paths=s3_paths
            )

            delete_messages(os.environ['QUEUE_URL'], receipt_handles)
            clean_dir(local_inputs_dir)
            clean_dir(local_outputs_dir)

        logger.info(f'Processed {total_n} images')
    except Exception as e:
        logger.error(e, exc_info=True)


if __name__ == '__main__':
    init_logger(logging.DEBUG if int(os.environ.get('DEBUG', 0)) else logging.INFO)
    run_inference()
