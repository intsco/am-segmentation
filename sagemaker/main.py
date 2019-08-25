import io
import logging
import os
from io import BytesIO

import cv2
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from albumentations.pytorch.functional import img_to_tensor

from am.segm.dataset import valid_transform

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def model_fn(model_dir):
    logger.info(f'Loading model from {model_dir}')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = smp.Unet(encoder_name='se_resnext50_32x4d', encoder_weights=None, decoder_use_batchnorm=True)
    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with open(os.path.join(model_dir, 'unet.pt'), 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    model.eval()
    return model.to(device)


def input_fn(input_data, content_type):
    logger.info(f'Deserializing input data')

    if content_type == 'application/x-image':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        buf = np.array(bytearray(input_data), dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        transform = valid_transform()
        image = transform(image=image)['image']
        tensor = img_to_tensor(image)
        tensor = tensor.unsqueeze(dim=0)
        return tensor.to(device)

    raise Exception(f'Wrong content_type: {content_type}')


def predict_fn(inputs, model):
    logger.info(f'Predicting')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        outputs = model(inputs.to(device))
        probs = torch.sigmoid(outputs).squeeze(dim=1).detach().cpu().numpy()
        masks = (probs > 0.5).astype(int)
        return masks


def output_fn(prediction, content_type):
    logger.info(f'Serializing output data')
    if content_type == 'application/x-image':
        image_array = prediction.squeeze() * 255
        is_success, buf = cv2.imencode('.png', image_array)
        assert is_success, f'Failed to encode image'
        return buf.tobytes()

    raise Exception(f'Wrong content_type: {content_type}')
