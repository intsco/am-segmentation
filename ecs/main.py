#!/usr/bin/env python

import logging
import os
from pathlib import Path

import boto3

import am.ecs.utils
from am.ecs.utils import (
    download_images_from_s3, load_model, predict, save_predictions,
    upload_images_to_s3, delete_messages)
from am.logger import init_logger

local_inputs_dir = Path('/tmp/inputs')
local_outputs_dir = Path('/tmp/outputs')

if __name__ == '__main__':
    # Init AWS clients
    init_logger()
    logger = logging.getLogger('am-segm')
    am.ecs.utils.s3 = boto3.client('s3')
    am.ecs.utils.sqs = boto3.client('sqs')

    logger.info('Batch inference of AM images')
    logger.debug(f"AWS_ACCESS_KEY_ID: {os.environ['AWS_ACCESS_KEY_ID']}")
    logger.debug(f"AWS_SECRET_ACCESS_KEY: {os.environ['AWS_SECRET_ACCESS_KEY']}")

    # Download data from S3
    input_paths, receipt_handles = download_images_from_s3(os.environ['INPUT_BUCKET'],
                                                           os.environ['QUEUE_URL'],
                                                           local_inputs_dir)

    # Load model and predict, save predictions
    model = load_model('model')
    image_dir = local_inputs_dir / input_paths[0].parent
    predictions = predict(model, image_dir)
    save_predictions(input_paths, predictions, local_outputs_dir)

    # Upload predictions to S3
    prefix = str(input_paths[0].parent)
    upload_images_to_s3(local_outputs_dir / prefix,
                        os.environ['OUTPUT_BUCKET'], prefix)
    delete_messages(os.environ['QUEUE_URL'], receipt_handles)
