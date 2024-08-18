import numpy as np
from higherdimknn.exceptions import NotFittedError
from higherdimknn.distances import *
from higherdimknn.constants import TOLERANCE
from typing import Callable

class KNRegressor():
    def __init__(
            self,
            n_neighbors: int,
            distance: str,
            p: float = None,
            custom_distance: Callable[[np.ndarray, np.ndarray], float] = None,
            weighted_predict: bool = False
        ):
        
        _distances = {
            'euclidean': euclidean_dist,
            'manhattan': manhattan_dist,
            'minkowski': lambda x, y: minkowski_dist(x, y, p),
            'custom': custom_distance
        }

        if distance not in _distances:
            raise ValueError("Provide a valid distance")

        if (distance == 'minkowski') and (p is None):
            raise Exception("Minkowski distance needs a value for p.")
        
        self.distance = _distances[distance]
        self.n_neighbors = n_neighbors

        self.weighted_predict = weighted_predict

        self.X_train = None
        self.y_train = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        if (X is None) or (y is None):
            raise ValueError("Data can't be of `None` type")
        
        self.X_train = X
        self.y_train = y
        
    def get_neighbors(self, p):

        distances = self.distance(self.X_train, p)

        n_indices = np.argsort(distances)[:self.n_neighbors]

        return n_indices, distances[n_indices]
        
    def _uniform_predict(self, X):
        if self.X_train is None:
            raise NotFittedError("The object wasn't trained. Use the `fit` method first. ")

        output = np.array([
            np.array([self.y_train[i] for i in self.get_neighbors(p)[0]])
            for p in X
        ])
        
        return np.array(output).mean(axis=1)
    
    def get_weights(self, distances):
        weights = np.where(distances <= TOLERANCE, 1.0, 1.0 / np.maximum(distances, TOLERANCE))
        return weights / weights.sum()
    
    def _weight_predict(self, X):

        _output = list()
        for p in X:
            _neighbors_indices, _distances = self.get_neighbors(p)

            _weights = self.get_weights(_distances)
            print(_weights)

            _output.append(
                np.matmul(_weights, self.y_train[_neighbors_indices])
            )
        return np.array(_output)
    
    def predict(self, X):
        if self.weighted_predict:
            return self._weight_predict(X)
        else:
            return self._uniform_predict(X)
        
