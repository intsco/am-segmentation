import argparse
import logging
import math
from pathlib import Path
from typing import List

from albumentations.pytorch.functional import img_to_tensor
from albumentations import Compose, Normalize, Resize
import torch
from torch.utils.data import Dataset

from am.logger import init_logger
from am.utils import load_model
from am.segment.image_utils import read_image, save_image

init_logger(level=logging.INFO)
logger = logging.getLogger('am-segm')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=8)
    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default='results')
    parser.add_argument('--pred-dir', type=str, default='results/predictions')
    parser.add_argument('--data-dir', type=str)

    args, _ = parser.parse_known_args()
    return args


class AMDataset(Dataset):

    def __init__(self, image_paths: List[Path], pred_base_path: Path):
        self._image_paths = image_paths
        self._pred_base_path = pred_base_path
        self._transform = Compose([Normalize(), Resize(512, 512)])

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        image_path = self._image_paths[idx]
        image = read_image(image_path, ch_n=3)
        image = self._transform(image=image)['image']
        mask_path = Path(image_path.parent.parent.name) / 'mask' / image_path.name
        return img_to_tensor(image), self._pred_base_path / mask_path

    def batches(self, batch_size=8):
        batch_n = math.ceil(len(self) / batch_size)
        idx = 0
        for _ in range(batch_n):
            items = []
            for _ in range(batch_size):
                if idx == len(self):
                    break
                items.append(self[idx])
                idx += 1
            yield zip(*items)


def save_predictions(predictions, output_paths):
    logger.debug(f'Saving {len(predictions)} predictions')
    for pred, output_path in zip(predictions, output_paths):
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        image = pred * 255
        logger.debug(f'Saving prediction: {image.shape} to {output_path}')
        save_image(image, output_path)


if __name__ == '__main__':
    args = parse_args()
    logger.info(f'Predicting image masks at "{args.data_dir}"')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(Path(args.model_dir) / 'model.pt')

    image_paths = []
    for group_path in Path(args.data_dir).iterdir():
        image_paths.extend(list((group_path / 'source').iterdir()))
    dataset = AMDataset(image_paths, pred_base_path=Path(args.pred_dir))

    with torch.no_grad():
        for images, mask_paths in dataset.batches(batch_size=args.batch_size):
            print('.', end='')

            inputs = torch.stack(images).to(device)
            probs = torch.sigmoid(model(inputs))
            probs = probs.squeeze(dim=1).detach().cpu().numpy()
            masks = (probs > 0.5).astype(int)

            save_predictions(masks, mask_paths)
