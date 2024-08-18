from higherdimknn.data import get_stock_data_history, get_stock_array, window_data
from higherdimknn.model import KNRegressor
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    google = get_stock_data_history('GOOGL')

    # google.reset_index(inplace=True)

    # print(google.dtypes)
    # google['prueba'] = google['Datetime'].apply(lambda x: x[:20])

    # print(google)
    # raise SystemExit(0)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(google['Close'], label='GOOG Close')
    ax.set_title('Alphabet Inc. (GOOG) Stock Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price (USD)')
    ax.legend()
    ax.grid(True)
    fig.savefig('images/GOOG_close.jpg', format='jpg', dpi=300)

    training_example = google.iloc[21:49, :]

    prices = training_example['Close'].values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(prices))[:21], prices[:21], label='Independent variable', color='blue')
    ax.plot(range(len(prices))[20:], prices[20:], label='Dependent variable', color='green')
    ax.set_xticks(range(len(prices)))
    ax.set_xticklabels(training_example.index.strftime('%Y-%m-%d %H:%M'), rotation=60)

    ax.set_title('Alphabet Inc. (GOOG) Stock Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price (USD)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    fig.savefig('images/training_example.jpg', format='jpg', dpi=300)

    google_array = get_stock_array(google)

    # print(google_array)

    X, y = window_data(google_array, lag=3)

    np.random.seed(41)
    indices = np.random.rand(X.shape[0]) > 0.8

    X_train, y_train = X[indices], y[indices]
    X_test, y_test = X[~indices], y[~indices]


    # print(X.shape, y.shape)

    knn = KNRegressor(n_neighbors=2, distance='euclidean', weighted_predict=True)

    knn.fit(X_train, y_train)

    obs_to_pred = 0
    neighbours, distances = knn.get_neighbors(X_test[obs_to_pred: obs_to_pred + 1])
    y_pred = knn.predict(X_test[obs_to_pred: obs_to_pred + 1])
    y_true = y_test[0]

    y_min, y_max = 105, 130

    fig, ax = plt.subplots(figsize=(10, 5))
    xs = range(28)
    ax.plot(xs[:21], X_test[obs_to_pred: obs_to_pred + 1].flatten(), label='Data to predict', color='blue')
    ax.plot(xs[:21], X_train[neighbours[0]].flatten(),
            label=f'Closest neighbour at {distances[0]:.3f} units',
            linestyle='dashed',
            color='green')
    ax.plot(xs[:21], X_train[neighbours[1]].flatten(),
            label=f'2nd closest neighbour at {distances[1]:.3f} units',
            linestyle='dashed',
            color='grey')
    ax.set_ylabel('Close Price (USD)')
    ax.legend()
    ax.grid(True)
    ax.set_ylim([y_min, y_max])
    plt.tight_layout()
    fig.savefig('images/predicting_step1.jpg', format='jpg', dpi=300)


    fig, ax = plt.subplots(figsize=(10, 5))
    xs = range(28)
    ax.plot(xs[:21], X_test[obs_to_pred: obs_to_pred + 1].flatten(), label='Data to predict', color='blue')
    ax.plot(xs[:21], X_train[neighbours[0]].flatten(),
            # label=f'Closest neighbour at {distances[0]:.3f} units',
            linestyle='dashed',
            color='green')
    ax.plot(xs[:21], X_train[neighbours[1]].flatten(),
            # label=f'2nd closest neighbour at {distances[1]:.3f} units',
            linestyle='dashed',
            color='grey')
    ax.plot(xs[20:], np.concatenate((np.array([X_train[neighbours[0]].flatten()[-1]]), y_train[neighbours[0]].flatten())),
            label=f'Dependent variable of closest neighbour',
            linestyle='dotted',
            color='green')
    ax.plot(xs[20:], np.concatenate((np.array([X_train[neighbours[1]].flatten()[-1]]), y_train[neighbours[1]].flatten())),
            label=f'Dependent variable of 2nd closest neighbour',
            linestyle='dotted',
            color='grey')
    ax.set_ylabel('Close Price (USD)')
    ax.legend()
    ax.grid(True)
    ax.set_ylim([y_min, y_max])
    plt.tight_layout()
    fig.savefig('images/predicting_step2.jpg', format='jpg', dpi=300)


    fig, ax = plt.subplots(figsize=(10, 5))
    xs = range(28)
    ax.plot(xs[:21], X_test[obs_to_pred: obs_to_pred + 1].flatten(), label='Data to predict', color='blue')
    ax.plot(xs[:21], X_train[neighbours[0]].flatten(),
            # label=f'Closest neighbour at {distances[0]:.3f} units',
            linestyle='dashed',
            color='green')
    ax.plot(xs[:21], X_train[neighbours[1]].flatten(),
            # label=f'2nd closest neighbour at {distances[1]:.3f} units',
            linestyle='dashed',
            color='grey')
    ax.plot(xs[20:], np.concatenate((np.array([X_train[neighbours[0]].flatten()[-1]]), y_train[neighbours[0]].flatten())),
            label=f'Dependent variable of closest neighbour',
            linestyle='dotted',
            color='green')
    ax.plot(xs[20:], np.concatenate((np.array([X_train[neighbours[1]].flatten()[-1]]), y_train[neighbours[1]].flatten())),
            label=f'Dependent variable of 2nd closest neighbour',
            linestyle='dotted',
            color='grey')
    
    ax.plot(xs[21:], y_pred.flatten(),
            label=f'Prediction (weigthed by distance)',
            linestyle='dotted',
            color='red')
    ax.plot(xs[20:], np.concatenate((np.array([X_test[0].flatten()[-1]]), y_true)), label='Real value', color='blue', linestyle='dotted')

    ax.set_ylabel('Close Price (USD)')
    ax.legend()
    ax.grid(True)
    ax.set_ylim([y_min, y_max])
    plt.tight_layout()
    fig.savefig('images/predicting_step3.jpg', format='jpg', dpi=300)

    # print(knn.predict(X[0: 5]))
