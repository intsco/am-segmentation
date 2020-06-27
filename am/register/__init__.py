import json
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from am.register.acq_grid_estimation import estimate_acq_grid_shape
from am.register.clustering import (
    cluster_coords, convert_labels_to_grid, convert_grid_to_indices
)
from am.register.image_processing import erode_dilate, find_am_centers, create_acq_index_mask, \
    remove_noisy_marks
from am.register.rotation import optimal_mask_rotation, rotate_image, rotate_am_centers
from am.register.visual import overlay_image_with_am_labels
from am.utils import time_it

logger = logging.getLogger('am-segm')


def load_source_mask(source_path, mask_path, meta_path):
    logger.info(f'Loading source from {source_path}')
    logger.info(f'Loading mask from {mask_path}')

    meta = json.load(open(meta_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    source = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)

    h, w = meta['orig_image']['h'], meta['orig_image']['w']
    if (h, w) != mask.shape:
        logger.info(f'Resizing mask: {mask.shape} -> {(h, w)}')
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return source, mask.astype(np.uint8)


@time_it
def export_am_coordinates(acq_index_mask_coo, path, acq_grid_shape):
    logger.info(f'Exporting acquisition index mask as AM coordinates at {path}')

    array = acq_index_mask_coo.todense().astype(np.uint16)
    image = Image.fromarray(array, "I;16")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


@time_it
def register_ablation_marks(
    source_path, mask_path, meta_path, am_coord_path, overlay_path, acq_grid_shape
):
    logger.info(f'Registering ablation marks for {mask_path}')
    source, mask = load_source_mask(source_path, mask_path, meta_path)

    target_axis = 1  # target axis: (1 = columns = X-axis, 0 = rows = Y-axis)
    best_angle = optimal_mask_rotation(mask, target_axis, angle_range=2, angle_step=0.1)
    mask = rotate_image(mask, best_angle)

    est_acq_grid_shape = estimate_acq_grid_shape(mask)
    if est_acq_grid_shape != acq_grid_shape:
        logger.warning(f'Estimated acquisition grid shape {est_acq_grid_shape} '
                       f'is different from provided {acq_grid_shape}')

    mask = erode_dilate(mask)
    mask = remove_noisy_marks(mask, est_acq_grid_shape)
    am_centers = find_am_centers(mask)

    target_axis = 0  # target axis: (1 = columns = X-axis, 0 = rows = Y-axis)
    row_labels = cluster_coords(
        axis_coords=am_centers[:, target_axis],
        n_clusters=est_acq_grid_shape[target_axis],
        sample_ratio=1
    )
    row_coords = am_centers[:, target_axis]

    target_axis = 1  # target axis: (1 = columns = X-axis, 0 = rows = Y-axis)
    col_labels = cluster_coords(
        axis_coords=am_centers[:, target_axis],
        n_clusters=est_acq_grid_shape[target_axis],
        sample_ratio=1
    )
    col_coords = am_centers[:, target_axis]

    mask = rotate_image(mask, -best_angle)
    am_centers = rotate_am_centers(am_centers, -best_angle, mask.shape)

    acq_y_grid = convert_labels_to_grid(row_coords, row_labels)
    acq_x_grid = convert_labels_to_grid(col_coords, col_labels)

    if est_acq_grid_shape != acq_grid_shape:
        acq_indices = convert_grid_to_indices(acq_y_grid, acq_x_grid + 1, cols=acq_grid_shape[1])
    else:
        acq_indices = convert_grid_to_indices(acq_y_grid, acq_x_grid, cols=acq_grid_shape[1])

    acq_index_mask_coo = create_acq_index_mask(mask, am_centers, acq_indices)
    export_am_coordinates(acq_index_mask_coo, am_coord_path, acq_grid_shape)
    overlay_image_with_am_labels(source, mask, am_centers, acq_indices, overlay_path)
