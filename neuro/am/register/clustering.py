import logging

from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

logger = logging.getLogger('am-segm')


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


def convert_labels_to_grid(axis_coords, labels):
    logger.info(f'Converting labels to grid values')
    rows = []
    for label in set(labels):
        rows.append([label, axis_coords[labels == label].max()])

    label_max_coords = np.array(rows, ndmin=2)
    label_max_coords = label_max_coords[label_max_coords[:, 1].argsort()]

    grid = np.zeros_like(labels)
    for idx, label in enumerate(label_max_coords[:, 0]):
        grid[labels == label] = idx
    return grid


def convert_grid_to_indices(y_grid, x_grid, cols, offset=1):
    logger.info(f'Converting grid values to indices')
    return (y_grid * cols + x_grid) + offset
