import argparse
import os
from pathlib import Path
import shutil
import numpy as np
import re
import matplotlib.pyplot as plt
import os
import skimage.io as io


def exp_output_path(input_path, output_path, p):
    p = str(p)
    stop_words = [str(input_path), 'Analyzed', 'Analysis', 'gridFit']
    for w in stop_words:
        p = re.sub(w, '', p)
    suffix = '_'.join(p.strip('/').split('/'))
    return output_path / suffix


fn_map = {
    'marks_check/PHASE_crop_bin1x1_window100.tiff': 'source.tiff',
    'marksMask.npy': 'mask.npy',
    'xye_clean2.npy': 'curated.npy',
}


def copy_paths(input_path, output_path):
    found_paths = []
    for root, dirs, files in os.walk(input_path):
        print(root)
        if root.endswith('gridFit'):
            root = Path(root)

            if np.all([(root / fn).exists() for fn in fn_map.keys()]):
                found_paths.append(root)

                exp_path = exp_output_path(input_path, output_path, root)
                exp_path.mkdir(parents=True, exist_ok=True)

                print(f'Copying {root} to {exp_path} ...')
                for inp_fn, out_fn in fn_map.items():
                    shutil.copy(root / inp_fn, exp_path / out_fn)
    print(f'Found {len(found_paths)} paths')


def scale(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def convert_npy_to_png(p):
    print(f'Converting {p}')

    window = 100
    source = plt.imread(f'{p}/source.tiff')
    XY_mask = np.load(f'{p}/mask.npy')
    XY_CM_curated = np.load(f'{p}/curated.npy')

    X0, Y0 = np.min(XY_CM_curated, axis=1) - window
    label = np.zeros(source.shape)

    for i, (X, Y) in enumerate(XY_mask):
        label[[int(x) for x in X-X0], [int(y) for y in Y-Y0]] = i+1

    io.imsave(fname=f'{p}/label.png', arr=scale(label))
    io.imsave(fname=f'{p}/mask.png', arr=np.array(label > 0, dtype=np.float))


def convert_paths_to_png(input_path):
    for p in input_path.iterdir():
        convert_npy_to_png(p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # copy_paths(input_path, output_path)
    convert_paths_to_png(output_path)
