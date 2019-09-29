import logging

import cv2
import numpy as np
from scipy.sparse import coo_matrix

from am.utils import min_max

logger = logging.getLogger('am-segm')


def erode_dilate(image, kernel=5):
    logger.info(f'Applying erosion and dilation')
    img = image.copy()
    img = cv2.erode(img, np.ones((kernel, kernel), np.uint8), iterations=1)
    img = cv2.dilate(img, np.ones((kernel, kernel), np.uint8), iterations=1)
    return img


def remove_noisy_marks(image):
    logger.info('Removing noisy ablation marks')

    def min_max_thr(image, target_axis):
        axis_sum = image.sum(axis=int(not target_axis))
        axis_sum_mean = np.mean(axis_sum)
        axis_min_thr, axis_max_thr = min_max((axis_sum > axis_sum_mean).nonzero()[0])
        return axis_min_thr, axis_max_thr

    target_axis = 0
    row_min_thr, row_max_thr = min_max_thr(image, target_axis)
    logger.info(f'axis={target_axis}, min={row_min_thr}, max={row_max_thr}')

    img = image.copy()
    img[:row_min_thr - 10, :] = 0
    img[row_max_thr + 10:, :] = 0

    target_axis = 1
    col_min_thr, col_max_thr = min_max_thr(image, target_axis)
    logger.info(f'axis={target_axis}, min={col_min_thr}, max={col_max_thr}')

    img[:, :col_min_thr - 10] = 0
    img[:, col_max_thr + 10:] = 0

    return img


def find_am_centers(image):
    logger.info(f'Finding AM centers')
    contours, hierarchy = cv2.findContours(image.copy().astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    am_centers = []
    for c in contours:
        M = cv2.moments(c)
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
        am_centers.append((y, x))

    return np.array(am_centers, ndmin=2)


def create_acq_index_mask(image, am_centers, acq_indices):
    logger.info(f'Creating acquisition index mask')
    _, markers = cv2.connectedComponents(image.copy().astype(np.uint8))
    acq_index_mask_coo = coo_matrix(markers)

    marker_acq_index_mapping = np.zeros_like(acq_index_mask_coo.data)
    for (c_row, c_col), acq_idx in zip(am_centers, acq_indices):
        marker = markers[c_row, c_col]
        marker_acq_index_mapping[marker] = acq_idx

    acq_index_mask_coo.data = marker_acq_index_mapping[acq_index_mask_coo.data]
    return acq_index_mask_coo
