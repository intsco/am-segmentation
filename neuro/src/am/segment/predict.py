import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.functional import img_to_tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def predict_ds(model, ds):
    batch_size = 32
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dl = DataLoader(
        dataset=ds,
        shuffle=False,
        num_workers=4,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )
    batch_n = len(dl)
    pred_list = []
    with torch.no_grad():
        model.eval()
        for batch_i, (inputs, _) in enumerate(dl, 1):
            print(f'{batch_i}/{batch_n} batches processed', end='\r', flush=True)
            inputs = inputs.to(device)
            probs = torch.sigmoid(model(inputs))
            probs = probs.squeeze(dim=1).detach().cpu().numpy()
            masks = (probs > 0.5).astype(np.uint8) * 255
            pred_list.extend([m for m in masks])
    return pred_list


def load_ds_images(ds):
    def f(i):
        image_path = ds.image_df.iloc[i].path
        return cv2.imread(str(image_path))[:, :, 0]

    return [f(i) for i in range(len(ds))]


def predict_save(model, ds, pred_path, groups=None):
    pred_list = predict_ds(model, ds)
    image_list = load_ds_images(ds)
    group_list = ds.image_df.group.values

    if not groups:
        groups = set(group_list)

    for group in groups:
        print(group)
        (pred_path / group / 'source').mkdir(parents=True, exist_ok=True)
        (pred_path / group / 'mask').mkdir(parents=True, exist_ok=True)

    for group in groups:
        print(group)
        inds = np.arange(group_list.shape[0])[group_list == group]

        for i, idx in enumerate(inds):
            group = group_list[idx]
            image = image_list[idx]
            pred_mask = pred_list[idx]

            image_path = pred_path / group / 'source' / f'{i:03}.png'
            print(image_path)
            cv2.imwrite(str(image_path), image)
            # mask_uint8 = np.round(pred_mask * (2 ** 8 - 1)).astype(np.uint8)
            mask_path = pred_path / group / 'mask' / f'{i:03}.png'
            print(mask_path)
            cv2.imwrite(str(mask_path), pred_mask)
