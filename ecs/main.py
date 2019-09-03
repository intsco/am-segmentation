#!/usr/bin/env python

import logging
import os
from pathlib import Path

from am.ecs import (
    consume_messages,
    download_images_from_s3,
    load_model, predict,
    save_predictions,
    upload_images_to_s3,
    delete_messages,
    INFERENCE_BATCH_SIZE,
    time_it,
)
from am.logger import init_logger
from am.utils import clean_dir

local_inputs_dir = Path('/tmp/inputs')
local_outputs_dir = Path('/tmp/outputs')


@time_it
def run_inference():
    init_logger(logging.INFO)
    logger = logging.getLogger('am-segm')

    logger.info('Batch inference of AM images')

    model = load_model('model')

    total_n = 0
    while True:
        input_paths, receipt_handles = consume_messages(
            queue_url=os.environ['QUEUE_URL'], n=INFERENCE_BATCH_SIZE
        )
        if not input_paths:
            logger.info('No more messages in the queue. Exiting')
            break
        total_n += len(input_paths)

        local_input_paths = download_images_from_s3(
            os.environ['INPUT_BUCKET'], input_paths, local_inputs_dir
        )

        predictions = predict(model, local_input_paths)
        save_predictions(input_paths, predictions, local_outputs_dir)

        prefix = str(input_paths[0].parent)
        upload_images_to_s3(
            local_outputs_dir, os.environ['OUTPUT_BUCKET'], prefix
        )

        delete_messages(os.environ['QUEUE_URL'], receipt_handles)
        clean_dir(local_inputs_dir)
        clean_dir(local_outputs_dir)

    logger.info(f'Processed {total_n} images')


if __name__ == '__main__':
    run_inference()
