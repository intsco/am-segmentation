from collections import defaultdict
import pandas as pd
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch.transforms import img_to_tensor
import cv2


def make_image_mask_dfs(data_path):
    image_types = {'source', 'mask'}

    image_paths = defaultdict(list)
    for group_path in data_path.iterdir():
        mis_types = image_types - set(d.name for d in group_path.iterdir())
        if mis_types:
            print(f'{group_path.name} group misses image type(s) {mis_types}')
        else:
            print(f'{group_path.name} group images collecting')

            for image_path, mask_path in zip(sorted((group_path / 'source').glob('*.png')),
                                             sorted((group_path / 'mask').glob('*.png'))):
                assert image_path.name == mask_path.name

                image_paths['source'].append((group_path.name, image_path))
                image_paths['mask'].append((group_path.name, mask_path))

    image_df = pd.DataFrame(image_paths['source'], columns=['group', 'path'])
    mask_df = pd.DataFrame(image_paths['mask'], columns=['group', 'path'])

    return image_df, mask_df


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
        albu.IAAAdditiveGaussianNoise(p=0.5, scale=(0, 0.01 * 255)),
        albu.OneOf([
            albu.ElasticTransform(p=1, alpha=240, sigma=240 * 0.05, alpha_affine=240 * 0.03),
            albu.GridDistortion(p=1),
            albu.OpticalDistortion(p=1, distort_limit=0.25, shift_limit=0.2)
        ], p=1),
        albu.OneOf([
            albu.CLAHE(p=1),
            albu.RandomBrightnessContrast(p=1),
            albu.RandomGamma(p=1),
        ], p=1),
        albu.Normalize(p=1),
    ], p=p)


def valid_transform(p=1):
    return albu.Compose([
        albu.Normalize(p=1),
    ], p=p)


class AMDataset(Dataset):
    def __init__(self, image_df, mask_df, transform=None):
        self.transform = transform
        self.image_df = image_df
        self.mask_df = mask_df

    def __len__(self):
        return len(self.mask_df)

    def __getitem__(self, idx):
        image_path = self.image_df.iloc[idx].path
        image = cv2.imread(str(image_path))
        mask_path = self.mask_df.iloc[idx].path
        mask = cv2.imread(str(mask_path))[:, :, :1]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = img_to_tensor(image)
        mask = img_to_tensor(mask)
        return image, mask
