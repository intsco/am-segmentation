import argparse
import shutil
from pathlib import Path


def copy_images(selected_exp_path, input_path, output_path):
    with open(selected_exp_path) as f:
        selected_exp = f.read().strip('\n').split('\n')

    for exp in selected_exp:
        (output_path / exp).mkdir(exist_ok=True)
        for fn in ['mask.png', 'source.tiff']:
            inp = input_path / exp / fn
            out = output_path / exp / fn
            print(f'Copying {inp} to {out}')
            shutil.copy(inp, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str)
    args = parser.parse_args()
    input_path = Path(args.input)

    #input_path = Path('/tmp/intsco/20180514_Coculture_PAPER')
    exp_set = input_path.name
    microscopy_path = Path('data/microscopy') / exp_set
    microscopy_path.mkdir(parents=True, exist_ok=True)
    selected_exp_path = f'data/{exp_set}_selected.txt'

    copy_images(selected_exp_path, input_path, microscopy_path)
