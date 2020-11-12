import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
import pmdarima as pm
from numpy import fft
from sklearn.metrics import mean_absolute_error, r2_score
from itertools import product

from forecasting.fourier import get_predict
from utils import mean_absolute_percentage_error, train_test_split, get_grid_from_dict, transform_date_start
from forecasting.model import Model


def generate_shapes(len_old, horizont, wt, level):
    """
    Генерирует shape массивов коэффициентов для заданных параметров
    len_old - общая длина исходного сигнала
    horizont - горизонт прогнозирования
    wt - название вейвлет-функции
    level - уровень декомпозиции dwt

    output - shapes в том же формате, в котором генерируются коэффициенты функцией wavedecn
    """
    pseudo_signal = np.arange(len_old + horizont)
    coeffs = pywt.wavedecn(pseudo_signal, wt, level=level)
    _, _, coeff_shapes = pywt.ravel_coeffs(coeffs)
    return coeff_shapes


def predict_wavelet(X, n, forecast, wt='coif1', level=None, plot=False, **forecast_params_list):
    """
    Предсказывает X на n точек вперед.
    forecast - функция, которая принимает на вход 1d вектор и n, дает предсказание на n точек вперед
    wt - имя вейвлет-функции, по умолчанию coif1, так как показало лучшую метрику
    level - уровень разложения (если None, то берется максимальный)
    plot - если True, то выведет графики предсказания всех компонент
    forecast_params_list - словарь
    """
    X = np.array(X).reshape(-1)

    # Ищем сколько прогнозировать на каждую компоненту
    shapes = generate_shapes(len(X), n, wt, level=level)
    coeff = pywt.wavedecn(X, wt, level=level)
    new_coeffs = []

    param_grid = get_grid_from_dict(forecast_params_list)

    for i in range(len(coeff)):
        if i == 0:
            # main component
            signal = coeff[i].reshape(-1)
            # сколько точек надо предсказать по этой компоненте
            n_to_predict = shapes[i][0] - len(coeff[i])
        else:
            # detail component
            signal = coeff[i]['d'].reshape(-1)
            n_to_predict = shapes[i]['d'][0] - len(coeff[i]['d'])

        if n_to_predict > 0:
            train, test = train_test_split(signal, train_size=0.9)
            mapes = []
            for param in param_grid:
                pred = forecast(train, len(test), **param)
                mapes.append(mean_absolute_error(test, pred))
            mapes = np.array(mapes)
            optimal_params = param_grid[np.argmin(mapes)]

            pred = forecast(signal, n_to_predict, **optimal_params)
            if n_to_predict > 1 and plot:
                plt.figure()
                plt.plot(signal)
                plt.plot(range(len(signal), len(signal) + n_to_predict), pred)
                plt.title(f'Level: {i + 1}')
                plt.show()
        else:
            pred = np.array([])
        res_signal = np.array(signal.tolist() + pred.tolist())
        if i == 0:
            new_coeffs.append(res_signal)
        else:
            new_coeffs.append({'d': res_signal})
    return pywt.waverecn(new_coeffs, wt)[-n:]


class Wavelet(Model):

    def __init__(self, X, n=14, level=None, wt='coif1', forecast=None, **params):
        """
        Init model and fit best params.
        Wavelet model fitted by series itself.

        X - df with column `price`
        n - horizon of forecasting
        level - how many times to decompose (if None will be set to max level)
        wt - which wavelet function to use from pywt library (if None will be fitted best)
        forecast - method which takes 1d vector and n and return prediction of len n (if None fourier model will be used)
        """
        super().__init__()

        self.level = level
        self.wt = wt
        self.n = n
        self.params = params

        if forecast is None:
            forecast = get_predict
        self.forecast = forecast

    def fit(self, X, verbose=False):
        pass

    def predict(self, X):
        X = X['price'].values
        return predict_wavelet(X, self.n, forecast=self.forecast, wt=self.wt, level=self.level)

    def predict_for_report(self, X, date_start, date_end):
        n = self.n
        # dates = X['price'].loc[date_start:date_end].index
        dates = pd.date_range(date_start, date_end)
        result = []
        for start_pivot in dates:
            full_signal = X.loc[:start_pivot, 'price'].interpolate().dropna().values
            train = full_signal[:-n]
            # print('for report', train.shape)
            pred = predict_wavelet(train, n, self.forecast, self.wt, self.level, **self.params)
            result.append(pred)

        columns = [f'n{i + 1}' for i in range(n)]

        date_start = transform_date_start(date_start, n)
        date_end = transform_date_start(date_end, n)
        dates = pd.date_range(date_start, date_end)
        return pd.DataFrame(result, columns=columns, index=dates)


if __name__ == '__main__':
    X = np.cos(np.linspace(1, 200, 200)) + np.cos(20 * np.linspace(10, 100, 200)) + np.cos(
        5 * np.linspace(1, 100, 200)) * np.sin(np.linspace(50, 100, 200)) + 4
    X = pd.DataFrame(X, columns=['price'])
    n_harm = list(range(1, 10))
    n = 140
    wt = 'coif1'
    level = None

    # model = Wavelet(X, n=n, wt=wt, level=level, n_harm=n_harm, trend_deg=list(range(5)))
    model = Wavelet(X, n=10)
    pred = model.predict(X)

    print(pred)
    # plt.plot(range(len(X)), X.values)
    # plt.plot(range(len(X), len(X) + 140), pred)
    # plt.show()
