import logging

from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

logger = logging.getLogger('am-reg')


def get_axis_coords(image, axis):
    coo_m = coo_matrix(image)
    return coo_m.col if axis == 1 else coo_m.row


def cluster_coords(axis_coords, n_clusters, sample_ratio=0.1):
    logger.info(f'Clustering {axis_coords.shape} array into {n_clusters} clusters')

    sample_size = int(axis_coords.shape[0] * sample_ratio)
    axis_coords_sample = np.random.choice(axis_coords, sample_size, replace=False)

    model = KMeans(n_clusters)
    model.fit(axis_coords_sample[:, None])
    labels = model.predict(axis_coords[:, None])
    return labels


def convert_labels_to_grid(axis_coords, labels, uniq_labels):
    rows = []
    for label in uniq_labels:
        rows.append([label, axis_coords[labels == label].max()])

    label_max_coords = np.array(rows, ndmin=2)
    label_max_coords = label_max_coords[label_max_coords[:,1].argsort()]

    grid = np.zeros_like(labels)
    for idx, label in enumerate(label_max_coords[:,0]):
        grid[labels == label] = idx
    return grid


def convert_grid_to_indices(y_grid, x_grid, cols, offset=1):
    return (y_grid * cols + x_grid) + offset


def find_acquisition_indices(image, acq_grid_shape):
    target_axis = 0
    # target axis: (1 = columns = X-axis, 0 = rows = Y-axis)
    # acq_grid_shape[target_axis]
    y_uniq_labels = np.arange(acq_grid_shape[target_axis])
    y_axis_coords = get_axis_coords(image, target_axis)
    y_labels = cluster_coords(y_axis_coords, n_clusters=acq_grid_shape[target_axis], sample_ratio=0.05)
    # y_axis_hist = image.sum(axis=int(not target_axis))

    target_axis = 1  # target axis: (1 = columns = X-axis, 0 = rows = Y-axis)
    # acq_grid_shape[target_axis]
    x_axis_coords = get_axis_coords(image, target_axis)
    x_labels = cluster_coords(x_axis_coords, n_clusters=acq_grid_shape[target_axis], sample_ratio=0.05)
    x_uniq_labels = np.arange(acq_grid_shape[target_axis])
    # x_axis_hist = image.sum(axis=int(not target_axis))

    y_grid = convert_labels_to_grid(y_axis_coords, y_labels, y_uniq_labels)
    x_grid = convert_labels_to_grid(x_axis_coords, x_labels, x_uniq_labels)

    acq_indices = convert_grid_to_indices(y_grid, x_grid, acq_grid_shape[1])
    return acq_indices
