import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import img_to_tensor
from albumentations import (
    PadIfNeeded,
    Compose,
    Normalize,
)
import cv2

full_size = 4096
n_splits = 8


def slice_image(image, n_splits):
    tile_size = image.shape[0] // n_splits
    tiles = []
    for i in range(n_splits):
        for j in range(n_splits):
            tile = image[i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size]
            tiles.append(tile)
    return tiles


def slice_images_masks(image_dirs, full_size, n_splits):
    """ N.B. All dirs should contains or should not contain masks
    """
    aug = PadIfNeeded(min_height=full_size, min_width=full_size,
                      border_mode=cv2.BORDER_REFLECT_101, p=1.0)

    images, masks = [], []
    source_image_padding = {}
    for image_dir in image_dirs:
        image_dir = Path(image_dir)

        source_image = cv2.imread(str(image_dir / 'image.png'))
        source_image_padding[str(image_dir)] = [full_size - x for x in source_image.shape[:2]]
        images += slice_image(aug(image=source_image)['image'], n_splits)

        if (image_dir / 'mask.png').exists():
            source_mask = cv2.imread(str(image_dir / 'mask.png'))
            masks += slice_image(aug(image=source_mask)['image'], n_splits)

    return images, masks, source_image_padding


def combine_tiles(tiles, full_size, n_splits):
    tile_size = full_size // n_splits
    image = np.zeros((full_size, full_size))
    for i in range(n_splits):
        for j in range(n_splits):
            tile = tiles[i * n_splits + j]
            image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = tile
    return image


def remove_padding(image, full_size, row_pad, col_pad):
    top_pad = row_pad // 2
    bottom_pad = row_pad - top_pad
    left_pad = col_pad // 2
    right_pad = col_pad - left_pad
    return image[top_pad:(full_size - bottom_pad), left_pad:(full_size - right_pad)]


def default_transform(p=1):
    return Compose([
        Normalize(p=1)
    ], p=p)


class AMDataset(Dataset):
    def __init__(self, image_dirs, full_size, n_splits, transform=None):
        self.transform = transform or default_transform()
        self.source_image_padding = {}
        self.images, self.masks, self.source_image_padding = \
            slice_images_masks(image_dirs, full_size, n_splits)

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
