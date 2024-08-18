import numpy as np


def minkowski_dist(x: np.ndarray, y: np.ndarray, p: float):
    if p <= 0:
        return ValueError("p must be > 0")
    
    _dif = np.abs(x - y)

    return np.power(np.power(_dif, p).sum(axis=(1, 2)), 1 / p)

def euclidean_dist(x: np.ndarray, y: np.ndarray):
    return minkowski_dist(x, y, 2)

def manhattan_dist(x: np.ndarray, y: np.ndarray):
    return minkowski_dist(x, y, 1)
