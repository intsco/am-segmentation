import time
from pathlib import Path

import numpy as np
import torch

from am.segment.image_utils import overlay_source_mask
from am.segment.loss import jaccard, CombinedLoss
from am.segment.utils import convert_to_image
from am.utils import dict_to_namedtuple

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def convert_history_to_tuple(history):
    history_dic = {}
    for name in history[0].keys():
        values = [m[name] for m in history]
        if name.endswith('loss') or name.endswith('metric'):
            values = np.array(values)
        history_dic[name] = values
    return dict_to_namedtuple(history_dic)


def train_loop(model, train_dl, valid_dl=None,
               optimizer=None, criterion=None,
               n_epochs=1, writer=None):
    print(f'Starting training loop. Using {device} device')

    start = time.time()
    best_valid_metric = 0
    best_model_path = Path('./best_model.pt')
    history = []
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

        track_overlay = None
        if valid_dl:
            with torch.no_grad():
                model.eval()
                losses, metrics = [], []
                max_pixels, track_inputs, track_outputs = 0, None, None
                for inputs, targets in valid_dl:
                    print('.', end='')
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    losses.append(loss.item() * inputs.size(0))
                    metric = jaccard((outputs > 0.5).float(), targets).sum().item()
                    metrics.append(metric)

                    if targets.sum() > max_pixels:
                        max_pixels = targets.sum()
                        track_inputs = inputs
                        track_outputs = outputs

                valid_loss = sum(losses) / len(valid_dl.dataset)
                valid_metric = sum(metrics) / len(valid_dl.dataset)
                print(f'\nValid loss: {valid_loss:.5f}, valid metric: {valid_metric:.5f}')
                if writer:
                    writer.add_scalar('loss/valid', valid_loss, epoch)
                    writer.add_scalar('jaccard/valid', valid_metric, epoch)
                if valid_metric > best_valid_metric:
                    best_valid_metric = valid_metric
                    print(f'Saving best model as "{best_model_path}"')
                    torch.save(model.state_dict(), best_model_path)

                track_overlay = overlay_source_mask(
                    convert_to_image(track_inputs), convert_to_image(torch.sigmoid(track_outputs))
                )

        history.append({
            'train_loss': running_loss,
            'train_metric': running_metric,
            'valid_loss': valid_loss,
            'valid_metric': valid_metric,
            'overlay': track_overlay
        })

        elapsed = int(time.time() - start)
        print(f'{elapsed // 60} min {elapsed % 60} sec')

    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint)

    return history


if __name__ == '__main__':
    pass
