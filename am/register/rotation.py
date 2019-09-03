from pathlib import Path
import logging

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2

from am.utils import time_it

logger = logging.getLogger('am-segm')


def plot_image(image, figsize=(7, 7)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image)
    return ax


def plot_hist(ys, figsize=(10, 5)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(np.arange(ys.shape[0]), ys)


def rotate_image(image, angle):
    rows, cols = image.shape
    center = ((cols - 1)/2.0, (rows - 1)/2.0)
    rot_M = cv2.getRotationMatrix2D(center, angle, scale=1)
    image = cv2.warpAffine(image.copy() / 255, rot_M, (cols, rows)).round()
    image = (image * 255).astype(np.uint8)
    return image


def rotate_am_centers(centers, angle, shape):
    logger.info(f'Rotating AM centers on {angle:.3f} angle')
    row_n, col_n = shape
    c_x, c_y = ((col_n - 1) / 2.0, (row_n - 1) / 2.0)
    rot_M = cv2.getRotationMatrix2D((c_x, c_y), angle, scale=1)

    n = centers.shape[0]
    points = np.ones((n, 3))
    points[:, :2] = centers[:, [1, 0]]  # swap columns (row, col) -> (x, y)
    points_rot = np.matmul(points, rot_M.T)
    centers_rot = points_rot[:, [1, 0]]  # swap columns back

    return centers_rot.round().astype(np.uint16)


def axis_proj(m, axis=0, thr_q=0.1):
    sums = m.sum(axis)
    return (sums > np.quantile(sums, thr_q)).sum() / sums.shape[0]


@time_it
def optimal_mask_rotation(image, target_axis, angle_range=2, angle_step=0.1):
    logger.info(f'Optimizing mask rotation, angle range={angle_range} with step={angle_step}')
    angle_proj_mapping = []
    for angle in np.arange(-angle_range, angle_range, angle_step):
        rot_image = rotate_image(image, angle)
        proj = axis_proj(rot_image, target_axis, thr_q=0.1)
        angle_proj_mapping.append((angle, proj))

    angle_proj_mapping = sorted(angle_proj_mapping, key=lambda t: t[1])
    best_angle, best_proj = angle_proj_mapping[0]
    # worst_angle, worst_proj = angle_proj_mapping[-1]
    logger.info(
        f'Target axis: {target_axis}, best angle: {best_angle:.3f}, '
        f'best proj {best_proj:.3f}'
    )
    return best_angle

