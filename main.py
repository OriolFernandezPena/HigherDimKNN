from higherdimknn.data import get_stock_data_history, get_stock_array, window_data
from higherdimknn.model import KNRegressor
import numpy as np

if __name__ == '__main__':

    google = get_stock_data_history('GOOGL')
    google_array = get_stock_array(google)
    X, y = window_data(google_array, lag=3)

    # print(X.shape, y.shape)

    knn = KNRegressor(n_neighbors=2, distance='euclidean')

    knn.fit(X, y)

    print(knn.predict(X[0: 5]))
