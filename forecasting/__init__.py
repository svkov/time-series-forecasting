from datetime import datetime

import numpy as np
from numpy import fft
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def fourierExtrapolation(x, n_predict, n_harm=10):
    n = x.size
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)  # find linear trend in x
    x_notrend = x - p[0] * t  # detrended x
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
    f = fft.fftfreq(n)  # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key=lambda i: np.absolute(f[i]))

    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n  # amplitude
        phase = np.angle(x_freqdom[i])  # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t


def forecaft_df_to_future(df, n_pred, n_harm=10000, start_from=60000, last_n_values=None):
    val = df['avg'].values[start_from:]
    y1 = val
    y2 = fourierExtrapolation(val, n_predict=n_pred, n_harm=n_harm)
    y2_hat = y2[-n_pred:] - (y2[-n_pred] - y1[-1])
    return y2_hat


def get_indexes(df, start_from=60000):
    return df.iloc[start_from:].index


def get_indexes_for_prediction(df, n_pred, freq='H'):
    ts = int(df.iloc[-1].time)
    date = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    return pd.date_range(date, periods=n_pred, freq=freq)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_forecast_test(df, n_harm=10000, start_from=0, train_size=0.8):
    val = df['avg'].values[start_from:]
    train, test = train_test_split(val, shuffle=False, train_size=train_size)
    n_pred = len(test)
    y_forecast = fourierExtrapolation(train, n_predict=n_pred, n_harm=n_harm)[-n_pred:]
    y_forecast = y_forecast - (y_forecast[0] - train[-1])
    return train, test, y_forecast


def calculate_metrics(y_true, y_pred):
    return {'MAPE': mean_absolute_percentage_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred)}


def forecast_train_test_df(df, n_harm=10000, start_from=0, train_size=0.8):
    train, test, y_forecast = get_forecast_test(df, n_harm, start_from, train_size)
    return calculate_metrics(test, y_forecast)
