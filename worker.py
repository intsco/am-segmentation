import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from time import sleep
import redis

from segmentation.inference import SegmentationModel
from utils import logger, save_status


def save_mask(mask, path):
    logger.info(f'Saving mask to {path}')
    mask = (mask * 255).astype(np.uint8)
    image = Image.fromarray(mask, mode='L')
    image.save(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentation model worker')
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--data-path', type=str)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    model = SegmentationModel(args.model_path)
    redis_client = redis.Redis()

    interval = 1
    while True:
        task_path = None
        try:
            task_id = redis_client.lpop('task-queue')
            if task_id:
                task_path = data_path / task_id.decode('utf-8')
                save_status(task_path / 'status.txt', 'PROCESSING')
                pred_mask = model.predict_mask(task_path)
                save_mask(pred_mask, task_path / 'mask.png')
                save_status(task_path / 'status.txt', 'FINISHED')
        except Exception as e:
            logger.error(e, exc_info=True)
            if task_path:
                save_status(task_path / 'status.txt', 'FAILED')
        except KeyboardInterrupt:
            break
        else:
            logger.debug(f'Sleep for {interval} sec...')
            sleep(interval)
