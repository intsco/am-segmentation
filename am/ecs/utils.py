import logging
import os
from pathlib import Path

import boto3
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import cv2
from albumentations.pytorch.functional import img_to_tensor
from albumentations import Compose, Normalize
from torch.utils.data import Dataset

s3 = None
sqs = None
logger = logging.getLogger('am-segm')


def upload_images_to_s3(local_inputs_dir, bucket, prefix, queue_url=None):
    for file_path in local_inputs_dir.iterdir():
        s3_file_path = f'{prefix}/{file_path.name}'
        logger.info(f'Uploading {file_path} to {s3_file_path}')
        s3.upload_file(str(file_path), bucket, s3_file_path)
        if queue_url:
            logger.info(f'Sending message to queue: {s3_file_path}')
            sqs.send_message(QueueUrl=queue_url,
                             MessageBody=s3_file_path)


def download_images_from_s3(bucket, queue_url, local_dir):
    receive_message_rounds = 4
    receipt_handles = []
    input_paths = []
    for i in range(receive_message_rounds):
        response = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=8)
        logger.debug(f"Round: {i}, messages: {len(response.get('Messages', []))}")
        for message in response.get('Messages', []):
            input_paths.append(Path(message['Body']))
            receipt_handles.append(message['ReceiptHandle'])

    input_local_paths = []
    for input_path in input_paths:
        input_local_path = local_dir / input_path
        input_local_paths.append(input_local_path)
        if not input_local_path.parent.exists():
            input_local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f'Downloading {input_path} to {input_local_path}')
        s3.download_file(bucket, str(input_path), str(input_local_path))

    return input_paths, receipt_handles


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

    def __init__(self, image_dir):
        self._image_paths = list(Path(image_dir).iterdir())
        self._transform = Compose([Normalize(p=1), ], p=1)

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        image_path = self._image_paths[idx]
        image = cv2.imread(str(image_path))
        image = self._transform(image=image)['image']
        return img_to_tensor(image)


def predict(model, image_dir):
    logger.info(f'Predicting: {image_dir}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        ds = AMDataset(image_dir)
        inputs = torch.stack([t for t in ds])
        inputs = inputs.to(device)
        probs = torch.sigmoid(model(inputs))
        probs = probs.squeeze(dim=1).detach().cpu().numpy()
        masks = (probs > 0.5).astype(int)
        return masks


def save_predictions(input_paths, predictions, local_dir):
    for input_path, pred in zip(input_paths, predictions):
        file_path = local_dir / input_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        image = pred * 255
        logger.info(f'Saving prediction: {image.shape} to {file_path}')
        res = cv2.imwrite(str(file_path), image)
        assert res, f'Failed to save {file_path}'


def delete_messages(queue_url, receipt_handles):
    logger.info(f'Deleting {len(receipt_handles)} messages from {queue_url}')
    for handle in receipt_handles:
        sqs.delete_message(QueueUrl=queue_url,
                           ReceiptHandle=handle)
