import numpy as np
import pandas as pd
import yfinance as yf


def get_stock_data_history(stock: str):

    _stocks = yf.Ticker(stock)

    prices = _stocks.history(period="2y", interval='1h')

    return prices.loc[:, ['Close']]


def get_stock_array(data: pd.DataFrame):
    _data = data.reset_index()
    _data['date'] = _data['Datetime'].dt.date
    _data = _data.sort_values(by='Datetime')

    _grouped = _data.groupby('date')['Close']

    return np.array([group.values for _, group in _grouped if len(group) == 7])


def window_data(data: np.array, lag: int):
    X = list()
    y = list()
    for i, _ in enumerate(data[:-lag]):
        X.append(data[i: i + lag])
        y.append(data[i + lag])

    return np.array(X), np.array(y)