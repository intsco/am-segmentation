#!/usr/bin/env python

import logging
import os
from pathlib import Path

from am.ecs import (
    consume_messages,
    download_images_from_s3,
    predict,
    save_predictions,
    upload_images_to_s3,
    delete_messages,
    INFERENCE_BATCH_SIZE,
    time_it,
)
from am.logger import init_logger
from am.utils import clean_dir, load_model

local_inputs_dir = Path('/tmp/inputs')
local_outputs_dir = Path('/tmp/outputs')

logger = logging.getLogger('am-segm')


@time_it
def run_inference():
    try:
        logger.info('Batch inference of AM images')

        model = load_model(os.environ['MODEL_PATH'])

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
