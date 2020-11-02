import logging

import numpy as np
from scipy import sparse
from scipy.signal import find_peaks, savgol_filter

logger = logging.getLogger('am-segm')


def find_coord_peaks(coords):
    hist, _ = np.histogram(coords, bins=2000)
    hist = savgol_filter(hist, window_length=21, polyorder=3)
    peaks, _ = find_peaks(hist)

    percentile_height = np.percentile(hist[peaks], q=90)
    peaks, _ = find_peaks(hist, prominence=percentile_height * 0.1, height=percentile_height / 2)
    return hist, peaks, percentile_height / 2


def estimate_acq_grid_shape(mask):
    logger.info(f'Estimating acquisition grid shape')
    mask_sparse = sparse.coo_matrix(mask)
    _, x_peaks, _ = find_coord_peaks(mask_sparse.col)
    _, y_peaks, _ = find_coord_peaks(mask_sparse.row)
    return y_peaks.shape[0], x_peaks.shape[0]
