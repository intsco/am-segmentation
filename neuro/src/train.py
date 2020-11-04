import argparse
import os
from pathlib import Path

import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from am.segment.dataset import create_dl, train_transform, valid_transform
from am.segment.loss import CombinedLoss
from am.segment.train import train_loop, convert_history_to_tuple
from am.segment.visual import plot_loss_metric, save_history_overlay
from am.utils import save_model, clean_dir


def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr-dec-1', type=float, default=3e-2)
    parser.add_argument('--lr-enc-2', type=float, default=3e-4)

    parser.add_argument('--num-workers', type=int, default=4)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default='results')
    parser.add_argument('--model-dir', type=str, default='results')
    parser.add_argument('--train', type=str, default='data/training-data/train')
    parser.add_argument('--valid', type=str, default='data/training-data/valid')

    args, _ = parser.parse_known_args()
    return args


def save_output_data(history, path):
    print(f'Saving training output to {path}')

    clean_dir(path)

    plot_loss_metric(history)
    plt.savefig(path / 'loss_metric_plots.png', bbox_inches='tight')
    plt.close()

    save_history_overlay(history, path / 'overlay')


if __name__ == '__main__':
    args = parse_args()

    # Load data
    train_dl = create_dl(
        paths=list(Path(args.train).iterdir()),
        transform=train_transform(),
        path_image_n=16,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    valid_dl = create_dl(
        paths=list(Path(args.valid).iterdir()),
        transform=valid_transform(),
        path_image_n=16,
        shuffle=False,
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers,
    )

    # First round: decoder only
    model = smp.Unet(encoder_name='se_resnext50_32x4d', decoder_use_batchnorm=True)
    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': args.lr_dec_1},
    ])
    criterion = CombinedLoss(bce_weight=0.5, jaccard=True, smooth=1e-15)
    history = train_loop(model, train_dl, valid_dl, optimizer, criterion, args.epochs)

    # Second round: encoder + decoder
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': args.lr_enc_2},
        {'params': model.decoder.parameters(), 'lr': args.lr_dec_1 / 10},
    ])
    history += train_loop(model, train_dl, valid_dl, optimizer, criterion, args.epochs)

    # Third round: encoder + decoder, lower lrs
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': args.lr_enc_2 / 10},
        {'params': model.decoder.parameters(), 'lr': args.lr_dec_1 / 100},
    ])
    history += train_loop(model, train_dl, valid_dl, optimizer, criterion, args.epochs)

    # Save model and outputs
    save_output_data(convert_history_to_tuple(history), Path(args.output_data_dir))
    save_model(model, model_path=Path(args.model_dir) / 'model.pt')
