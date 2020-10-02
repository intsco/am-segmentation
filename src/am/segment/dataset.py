from collections import defaultdict
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as albu
from albumentations.pytorch.transforms import img_to_tensor
import cv2


def default_transform(p=1):
    return albu.Compose([
        albu.Normalize(p=1)
    ], p=p)


def train_transform(p=1):
    return albu.Compose([
        albu.VerticalFlip(p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.Transpose(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.ShiftScaleRotate(p=1),
        albu.IAAAdditiveGaussianNoise(p=0.5, scale=(0, 0.02 * 255)),
        albu.OneOf([
            albu.CLAHE(p=1, clip_limit=3),
            albu.RandomBrightnessContrast(p=1, brightness_limit=0.2, contrast_limit=0.2),
            albu.RandomGamma(p=1, gamma_limit=(80, 120)),
        ], p=1),
        albu.Normalize(),
        albu.Resize(512, 512),
    ], p=p)


def valid_transform(p=1):
    return albu.Compose([
        albu.Normalize(),
        albu.Resize(512, 512),
    ], p=p)


class AMDataset(Dataset):
    def __init__(self, image_df, mask_df, transform=None):
        self.image_df = image_df
        self.mask_df = mask_df
        self.transform = transform

    def __len__(self):
        return len(self.mask_df)

    def _read_image(self, path, one_channel):
        image = cv2.imread(str(path))
        if one_channel:
            image = image[:, :, :1]  # because ch0==ch1==ch2
        return image

    def __getitem__(self, idx):
        image_path = self.image_df.iloc[idx].path
        image = self._read_image(image_path, one_channel=False)
        mask_path = self.mask_df.iloc[idx].path
        if mask_path.exists():
            mask = self._read_image(mask_path, one_channel=True)
        else:
            mask = np.zeros_like(image)[:, :, :1]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = img_to_tensor(image)
        mask = img_to_tensor(mask)
        return image, mask

    def __add__(self, other):
        comb_image_df = pd.concat([self.image_df, other.image_df])
        comb_mask_df = pd.concat([self.mask_df, other.mask_df])
        return AMDataset(comb_image_df, comb_mask_df, self.transform)


def create_image_mask_dfs(data_path):
    experiment = data_path.parent.name
    image_paths = defaultdict(list)
    for group_path in data_path.iterdir():
        for image_path in sorted((group_path / 'source').glob('*.png')):
            mask_path = group_path / 'mask' / image_path.name

            image_paths['source'].append((experiment, group_path.name, image_path))
            image_paths['mask'].append((experiment, group_path.name, mask_path))

    image_df = pd.DataFrame(image_paths['source'], columns=['experiment', 'group', 'path'])
    mask_df = pd.DataFrame(image_paths['mask'], columns=['experiment', 'group', 'path'])
    return image_df, mask_df


def create_ds(data_path, transform=None, groups=None, size=None):
    image_df, mask_df = create_image_mask_dfs(Path(data_path))
    image_n, mask_n = len(image_df), len(mask_df)
    assert image_n > 0, f'No image files found at {data_path}'
    assert image_n == mask_n, f'Different number of source and mask files: {image_n} != {mask_n}'

    if groups:
        image_df = image_df[image_df.group.isin(groups)]
        mask_df = mask_df[mask_df.group.isin(groups)]

    if size:
        n = image_df.shape[0]
        if n < size:
            mult = ceil(size / n)
            image_df = pd.concat([image_df] * mult).head(size)
            mask_df = pd.concat([mask_df] * mult).head(size)
        else:
            inds = np.random.choice(image_df.shape[0], size, replace=False)
            image_df = image_df.iloc[inds]
            mask_df = mask_df.iloc[inds]

    return AMDataset(image_df, mask_df, transform=transform)


def create_dl(
    paths, transform=None, path_image_n=None, shuffle=True, batch_size=4, num_workers=4
):
    assert paths

    print(f'Loading data from {paths} paths')

    ds = None
    while paths:
        path = paths.pop(0)
        path_ds = create_ds(
            Path(path), transform=transform, size=path_image_n
        )
        if not ds:
            ds = path_ds
        else:
            ds += path_ds

    dl = DataLoader(
        dataset=ds,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )
    return dl
