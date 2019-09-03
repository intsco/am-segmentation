import json
import logging
import operator
from pathlib import Path

import cv2
import numpy as np

from am.register.clustering import (
    cluster_coords, convert_labels_to_grid, convert_grid_to_indices
)
from am.register.image_processing import erode_dilate, find_am_centers, create_acq_index_mask
from am.register.rotation import optimal_mask_rotation, rotate_image, rotate_am_centers
from am.utils import time_it

logger = logging.getLogger('am-segm')


def load_mask(mask_path, meta_path):
    logger.info(f'Loading mask from {mask_path}')

    meta = json.load(open(meta_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    h, w = meta['orig_image']['h'], meta['orig_image']['w']
    if (h, w) != mask.shape:
        logger.info(f'Resizing mask: {mask.shape} -> {(h, w)}')
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


@time_it
def export_am_coordinates(acq_index_mask_coo, path, acq_grid_shape):
    logger.info(f'Exporting acquisition index mask as AM coordinates at {path}')

    data = acq_index_mask_coo.data
    row = acq_index_mask_coo.row
    col = acq_index_mask_coo.col

    am_x_y_coords = []
    for acq_idx in range(1, operator.mul(*acq_grid_shape) + 1):
        pixel_inds = (data == acq_idx).nonzero()[0]
        ys = row[pixel_inds]
        xs = col[pixel_inds]
        am_x_y_coords.append([xs, ys])

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, am_x_y_coords)


@time_it
def register_ablation_marks(mask_path, meta_path, am_coord_path, acq_grid_shape):
    image = load_mask(mask_path, meta_path).astype(np.uint8)

    target_axis = 1  # target axis: (1 = columns = X-axis, 0 = rows = Y-axis)
    best_angle = optimal_mask_rotation(image, target_axis, angle_range=2, angle_step=0.1)
    image = rotate_image(image, best_angle)

    image = erode_dilate(image)
    am_centers = find_am_centers(image)

    target_axis = 0  # target axis: (1 = columns = X-axis, 0 = rows = Y-axis)
    row_labels = cluster_coords(
        axis_coords=am_centers[:, target_axis],
        n_clusters=acq_grid_shape[target_axis],
        sample_ratio=1
    )
    row_coords = am_centers[:, target_axis]

    target_axis = 1  # target axis: (1 = columns = X-axis, 0 = rows = Y-axis)
    col_labels = cluster_coords(
        axis_coords=am_centers[:, target_axis],
        n_clusters=acq_grid_shape[target_axis],
        sample_ratio=1
    )
    col_coords = am_centers[:, target_axis]

    acq_y_grid = convert_labels_to_grid(row_coords, row_labels)
    acq_x_grid = convert_labels_to_grid(col_coords, col_labels)
    acq_indices = convert_grid_to_indices(acq_y_grid, acq_x_grid, acq_grid_shape[1])

    image = rotate_image(image, -best_angle)
    am_centers = rotate_am_centers(am_centers, -best_angle, image.shape)

    acq_index_mask_coo = create_acq_index_mask(image, am_centers, acq_indices)

    export_am_coordinates(acq_index_mask_coo, am_coord_path, acq_grid_shape)
