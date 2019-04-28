import torch
from albumentations.pytorch.functional import img_to_tensor

from .dataset import (
    combine_tiles,
    remove_padding,
    slice_images_masks,
    default_transform
)
from .model import UNet11
from utils import logger

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SegmentationModel(object):

    def __init__(self, model_path, full_size=4096, n_splits=8):
        self.full_size = full_size
        self.n_splits = n_splits
        self.transform = default_transform()

        logger.info('Loading model...')
        model = UNet11(pretrained=False)
        state = torch.load(open(model_path, 'rb'), map_location=device)
        model.load_state_dict(state)
        self.model = model.to(device)

    def predict_mask(self, image_dir, threshold=0.3):
        images, masks, source_image_padding = slice_images_masks([image_dir],
                                                                 full_size=self.full_size,
                                                                 n_splits=self.n_splits)
        image_tensors = [img_to_tensor(self.transform(image=img)['image'])
                         for img in images]

        logger.info(f'Predicting mask for {image_dir}...')
        pred_outputs = []
        with torch.no_grad():
            self.model.eval()
            for inputs in image_tensors:
                inputs = torch.unsqueeze(inputs, dim=0).to(device)
                outputs = torch.sigmoid(self.model(inputs)) > threshold
                pred_outputs.append(outputs)
        pred_outputs = torch.squeeze(torch.cat(pred_outputs))
        pred_outputs = pred_outputs.detach().cpu().numpy()

        pred_mask = combine_tiles(pred_outputs, self.full_size, self.n_splits)
        row_pad, col_pad = source_image_padding[str(image_dir)]
        pred_mask = remove_padding(pred_mask, self.full_size, row_pad, col_pad)
        logger.info('Done')
        return pred_mask




