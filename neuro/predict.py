import argparse
import math
import os
from pathlib import Path

import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from am.ecs import AMDataset
from am.segment.dataset import create_dl, train_transform, valid_transform
from am.segment.loss import CombinedLoss
from am.segment.train import train_loop, convert_history_to_tuple
from am.segment.visual import plot_loss_metric, save_history_overlay
from am.utils import save_model, clean_dir, load_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)

    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default='results')
    parser.add_argument('--data-dir', type=str)

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    model = load_model(Path(args.model_dir) / 'model.pt')

    # Load data
    # dl = create_dl(
    #     paths=[Path(args.data_dir)],
    #     transform=valid_transform(),
    #     shuffle=False,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    # )

    image_paths = Path(args.data_dir).iterdir()
    batch_size = args.batch_size
    batch_n = math.ceil(len(a) / batch_size)

    AMDataset()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for inputs, _ in dl:
            print('.', end='')
            inputs = inputs.to(device)
            probs = torch.sigmoid(model(inputs))
            probs = probs.squeeze(dim=1).detach().cpu().numpy()
            masks = (probs > 0.5).astype(int)

