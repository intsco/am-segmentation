import cv2
import torch
from albumentations.pytorch.functional import img_to_tensor
from torch.utils.data import Dataset

from .dataset import (
    combine_tiles,
    remove_padding,
    default_transform,
    slice_image, pad_source_image,
    get_n_splits
)
from .model import UNet11
from utils import logger

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AMDataset(Dataset):
    def __init__(self, image_dirs, tile_size=512, transform=None):
        self.transform = transform or default_transform()
        self.source_image_padding = {}
        self.images, self.masks, self.source_image_padding, self.source_image_n_slices = \
            slice_images_masks(image_dirs, tile_size=tile_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx][:, :, :1]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = img_to_tensor(augmented['image'])
            mask = img_to_tensor(augmented['mask'])

        return image, mask



class SegmentationModel(object):

    def __init__(self, model_path, tile_size=512):
        self.tile_size = tile_size
        self.transform = default_transform()

        logger.info('Loading model...')
        model = UNet11(pretrained=False)
        state = torch.load(open(model_path, 'rb'), map_location=device)
        model.load_state_dict(state)
        self.model = model.to(device)

    def predict_mask(self, image_path, threshold=None):
        image = cv2.imread(str(image_path))

        n_splits = get_n_splits(max(image.shape[:2]), self.tile_size)
        full_size = n_splits * self.tile_size
        image, (row_pad, col_pad) = pad_source_image(image, full_size)
        tiles = slice_image(image, self.tile_size)

        image_tensors = [img_to_tensor(self.transform(image=img)['image'])
                         for img in tiles]

        logger.info(f'Predicting mask for {image_path}...')
        pred_outputs = []
        with torch.no_grad():
            self.model.eval()
            for inputs in image_tensors:
                inputs = torch.unsqueeze(inputs, dim=0).to(device)
                outputs = torch.sigmoid(self.model(inputs))
                if threshold:
                    outputs = outputs > threshold
                pred_outputs.append(outputs)
        pred_outputs = torch.squeeze(torch.cat(pred_outputs))
        pred_outputs = pred_outputs.detach().cpu().numpy()

        pred_mask = combine_tiles(pred_outputs, self.tile_size, n_splits)
        pred_mask = remove_padding(pred_mask, row_pad, col_pad)
        logger.info('Done')
        return pred_mask
