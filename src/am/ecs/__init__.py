import logging
from concurrent.futures import ThreadPoolExecutor

import boto3
import torch
from albumentations.pytorch.functional import img_to_tensor
from albumentations import Compose, Normalize, Resize
from torch.utils.data import Dataset

from am.segment.image_utils import read_image, save_image

logger = logging.getLogger('am-segm')

INFERENCE_BATCH_SIZE = 8


def upload_images_to_s3(local_paths, bucket, s3_paths, queue_url=None):
    logger.info(f'Uploading {len(local_paths)} files to s3://{bucket}')
    s3 = boto3.client('s3')
    sqs = boto3.client('sqs')

    def upload(args):
        local_path, s3_path = args
        logger.debug(f'Uploading {local_path} to s3://{bucket}/{s3_path}')
        s3.upload_file(str(local_path), bucket, s3_path)

        if queue_url:
            logger.debug(f'Sending message to queue: {s3_path}')
            sqs.send_message(QueueUrl=queue_url, MessageBody=s3_path)

    with ThreadPoolExecutor() as executor:
        list(executor.map(upload, zip(local_paths, s3_paths)))


def consume_messages(queue_url, n=8):
    sqs = boto3.client('sqs')
    receipt_handles = []
    input_paths = []
    for i in range(n):
        resp = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1)
        logger.debug(f"Round: {i}, messages: {len(resp.get('Messages', []))}")
        for message in resp.get('Messages', []):
            input_paths.append(message['Body'])
            receipt_handles.append(message['ReceiptHandle'])

    return input_paths, receipt_handles


def download_images_from_s3(bucket, s3_paths, local_paths):
    logger.info(f'Downloading {len(s3_paths)} files from s3://{bucket}')
    s3 = boto3.client('s3')

    def download(args):
        s3_path, local_path = args
        if not local_path.parent.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f'Downloading {s3_path} to {local_path}')
        s3.download_file(bucket, str(s3_path), str(local_path))

    with ThreadPoolExecutor() as executor:
        list(executor.map(download, zip(s3_paths, local_paths)))


def remove_images_from_s3(bucket, prefix):
    logger.info(f'Deleting objects from s3://{bucket}/{prefix}')
    boto3.resource('s3').Bucket(bucket).objects.filter(Prefix=prefix).delete()


class AMDataset(Dataset):

    def __init__(self, image_paths):
        self._image_paths = image_paths
        self._transform = Compose([Normalize(), Resize(512, 512)])

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        image_path = self._image_paths[idx]
        image = read_image(image_path, ch_n=3)
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


def save_predictions(predictions, output_paths):
    logger.info(f'Saving {len(predictions)} predictions')
    for pred, output_path in zip(predictions, output_paths):
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        image = pred * 255
        logger.debug(f'Saving prediction: {image.shape} to {output_path}')
        save_image(image, output_path)


def delete_messages(queue_url, receipt_handles):
    logger.info(f'Deleting {len(receipt_handles)} messages from {queue_url}')
    sqs = boto3.client('sqs')
    for handle in receipt_handles:
        sqs.delete_message(QueueUrl=queue_url,
                           ReceiptHandle=handle)


def list_images_on_s3(bucket, prefix):
    s3 = boto3.client('s3')
    keys = []
    kwargs = dict(Bucket=bucket, Prefix=prefix)
    while True:
        resp = s3.list_objects_v2(**kwargs)
        for doc in resp.get('Contents', []):
            keys.append(doc['Key'])

        if resp.get('IsTruncated', False):
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        else:
            break
    return keys
