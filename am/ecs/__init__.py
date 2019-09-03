import logging
import os
from pathlib import Path
from time import time, sleep
from collections import Counter

import boto3
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import cv2
from albumentations.pytorch.functional import img_to_tensor
from albumentations import Compose, Normalize
from torch.utils.data import Dataset

from am.utils import time_it

s3 = boto3.client('s3')
sqs = boto3.client('sqs')
ecs = boto3.client('ecs')
logger = logging.getLogger('am-segm')

INFERENCE_BATCH_SIZE = 8


def upload_images_to_s3(local_inputs_dir, bucket, prefix, queue_url=None):
    logger.info(
        f'Uploading files from {local_inputs_dir} to s3://{bucket}/{prefix}'
    )

    uploaded_paths = []
    for file_path in local_inputs_dir.iterdir():
        s3_file_path = f'{prefix}/{file_path.name}'
        uploaded_paths.append(s3_file_path)

        logger.debug(f'Uploading {file_path} to s3://{bucket}/{s3_file_path}')
        s3.upload_file(str(file_path), bucket, s3_file_path)

        if queue_url:
            logger.debug(f'Sending message to queue: {s3_file_path}')
            sqs.send_message(QueueUrl=queue_url, MessageBody=s3_file_path)

    logger.info(f'Uploaded {len(uploaded_paths)} files')
    return uploaded_paths


def consume_messages(queue_url, n=8):
    receipt_handles = []
    input_paths = []
    for i in range(n):
        response = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1)
        logger.debug(f"Round: {i}, messages: {len(response.get('Messages', []))}")
        for message in response.get('Messages', []):
            input_paths.append(Path(message['Body']))
            receipt_handles.append(message['ReceiptHandle'])

    return input_paths, receipt_handles


def download_images_from_s3(bucket, input_paths, local_dir):
    logger.info(
        (f'Downloading {len(input_paths)} files '
         f'from s3://{bucket} to {local_dir}')
    )
    local_dir.mkdir(parents=True, exist_ok=True)

    local_paths = []
    for input_path in input_paths:
        local_path = local_dir / Path(input_path).name
        local_paths.append(local_path)
        logger.debug(f'Downloading {input_path} to {local_path}')
        s3.download_file(bucket, str(input_path), str(local_path))
    return local_paths


def remove_images_from_s3(bucket, prefix):
    logger.info(f'Deleting objects from s3://{bucket}/{prefix}')
    boto3.resource('s3').Bucket(bucket).objects.filter(Prefix=prefix).delete()


def load_model(model_dir):
    logger.info(f'Loading model from "{model_dir}"')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = smp.Unet(encoder_name='se_resnext50_32x4d',
                     encoder_weights=None, decoder_use_batchnorm=True)
    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with open(os.path.join(model_dir, 'unet.pt'), 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    model.eval()
    return model.to(device)


class AMDataset(Dataset):

    def __init__(self, image_paths):
        self._image_paths = image_paths
        self._transform = Compose([Normalize(p=1), ], p=1)

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        image_path = self._image_paths[idx]
        image = cv2.imread(str(image_path))
        image = self._transform(image=image)['image']
        return img_to_tensor(image)


def predict(model, image_paths):
    logger.info(f'Predicting {len(image_paths)} paths')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        ds = AMDataset(image_paths)
        inputs = torch.stack([t for t in ds])
        inputs = inputs.to(device)
        probs = torch.sigmoid(model(inputs))
        probs = probs.squeeze(dim=1).detach().cpu().numpy()
        masks = (probs > 0.5).astype(int)
        return masks


def save_predictions(input_paths, predictions, local_dir):
    for input_path, pred in zip(input_paths, predictions):
        file_path = local_dir / input_path.name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        image = pred * 255
        logger.debug(f'Saving prediction: {image.shape} to {file_path}')
        res = cv2.imwrite(str(file_path), image)
        assert res, f'Failed to save {file_path}'


def delete_messages(queue_url, receipt_handles):
    logger.info(f'Deleting {len(receipt_handles)} messages from {queue_url}')
    for handle in receipt_handles:
        sqs.delete_message(QueueUrl=queue_url,
                           ReceiptHandle=handle)


def list_images_on_s3(bucket, prefix):
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [doc['Key'] for doc in resp.get('Contents', [])]


@time_it
def run_wait_for_inference_task(stop_callback, task_n=2, sleep_interval=10, timeout=300):
    assert 0 < task_n <= 20
    ecs_max_task_n = 10

    task_arns = []
    while task_n > 0:
        ecs_task_n = min(task_n, ecs_max_task_n)

        logger.info(f'Running {ecs_task_n} tasks in ECS')
        resp = ecs.run_task(
            cluster='am-segm',
            taskDefinition='am-segm-batch',
            count=ecs_task_n,
            launchType='FARGATE',
            networkConfiguration={
                'awsvpcConfiguration': {
                    'subnets': [
                        'subnet-2619c87f',  # eu-west-1a availability zone in SM VPC
                    ],
                    'securityGroups': [
                        'sg-73462d16',  # default in SM VPC
                    ],
                    'assignPublicIp': 'ENABLED'
                }
            }
        )
        task_arns += [t['taskArn'] for t in resp['tasks']]
        task_n -= ecs_task_n
        if task_n > 0:
            sleep(5)

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
