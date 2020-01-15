import numpy as np
from scipy import sparse
from scipy.signal import find_peaks


def find_coord_peaks(coords):
    hist, _ = np.histogram(coords, bins=500)
    peaks, _ = find_peaks(hist)
    half_avg_height = hist[peaks].mean() / 2
    peaks, _ = find_peaks(hist, height=half_avg_height)
    return hist, peaks, half_avg_height


def estimate_acq_grid_shape(mask):
    mask_sparse = sparse.coo_matrix(mask)
    _, x_peaks, _ = find_coord_peaks(mask_sparse.col)
    _, y_peaks, _ = find_coord_peaks(mask_sparse.row)
    return y_peaks.shape[0], x_peaks.shape[0]
