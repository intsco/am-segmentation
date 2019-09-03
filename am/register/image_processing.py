import logging

import cv2
import numpy as np
from scipy.sparse import coo_matrix

logger = logging.getLogger('am-segm')


def erode_dilate(image, kernel=5):
    logger.info(f'Applying erosion and dilation')
    img = image.copy()
    img = cv2.erode(img, np.ones((kernel, kernel), np.uint8), iterations=2)
    img = cv2.dilate(img, np.ones((kernel, kernel), np.uint8), iterations=1)
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
