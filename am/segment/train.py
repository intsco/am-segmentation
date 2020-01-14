import time
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet, FPN

from am.segment.dataset import AMDataset, create_image_mask_dfs, train_transform, valid_transform
from am.segment.loss import jaccard, CombinedLoss
from am.segment.model import UNet11

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_loop(model, train_dl, valid_dl=None,
               optimizer=None, criterion=None,
               n_epochs=1, writer=None):
    start = time.time()
    for epoch in range(0, n_epochs):
        print(f'Epoch {epoch + 1}/{n_epochs}')
        print('-' * 10)

        model.to(device)
        model.train()

        running_loss, running_metric = 0.0, 0.0
        for i, (inputs, targets) in enumerate(train_dl, 1):
            print('.', end='')
            inputs = inputs.to(device)

            with torch.no_grad():
                targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_metric += jaccard((outputs > 0.5).float(), targets).sum().item()

        running_loss /= len(train_dl.dataset)
        running_metric /= len(train_dl.dataset)
        print(f'\nTrain loss: {running_loss:.5f}, train metric: {running_metric:.5f}')
        if writer:
            writer.add_scalar('loss/train', running_loss, epoch)
            writer.add_scalar('jaccard/train', running_metric, epoch)

        if valid_dl:
            with torch.no_grad():
                model.eval()
                losses, metrics = [], []
                for inputs, targets in valid_dl:
                    print('.', end='')
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    losses.append(loss.item() * inputs.size(0))
                    metric = jaccard((outputs > 0.5).float(), targets).sum().item()
                    metrics.append(metric)

                valid_loss = sum(losses) / len(valid_dl.dataset)
                valid_metric = sum(metrics) / len(valid_dl.dataset)
                print(f'\nValid loss: {valid_loss:.5f}, valid metric: {valid_metric:.5f}')
                if writer:
                    writer.add_scalar('loss/valid', valid_loss, epoch)
                    writer.add_scalar('jaccard/valid', valid_metric, epoch)
        elapsed = int(time.time() - start)
        print(f'{elapsed // 60} min {elapsed % 60} sec')


if __name__ == '__main__':
    from sklearn.model_selection import GroupKFold

    image_df, mask_df = create_image_mask_dfs(Path('data/tiles'))
    cv = GroupKFold(n_splits=4)
    train_inds, valid_inds = next(cv.split(image_df, groups=image_df.group))

    train_ds = AMDataset(image_df.iloc[train_inds], mask_df.iloc[train_inds],
                         transform=train_transform())
    valid_ds = AMDataset(image_df.iloc[valid_inds], mask_df.iloc[valid_inds],
                         transform=valid_transform())

    batch_size = 4
    train_dl = DataLoader(
        dataset=train_ds,
        shuffle=True,
        num_workers=4,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )
    valid_dl = DataLoader(
        dataset=valid_ds,
        shuffle=False,
        num_workers=4,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )
    len(train_dl), len(valid_dl)

    lr = 1e-3
    n_epochs = 5

    model = UNet11(pretrained=True)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CombinedLoss()

    train_loop(model, train_dl, valid_dl,
               optimizer, criterion,
               n_epochs=n_epochs)
