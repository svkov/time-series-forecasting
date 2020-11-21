import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from sklearn.metrics import mean_absolute_error

from src.models.fourier import get_predict
from src.utils import train_test_split, get_grid_from_dict, transform_date_start
from src.models.model import Model


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
    Предсказывает df на n точек вперед.
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

    def __init__(self, df, n=14, level=None, wt='coif1', forecast=None, column_name='price', **params):
        """
        Init model and fit best params.
        Wavelet model fitted by series itself.

        df - df with column `price`
        n - horizon of models
        level - how many times to decompose (if None will be set to max level)
        wt - which wavelet function to use from pywt library (if None will be fitted best)
        forecast - method which takes 1d vector and n and return prediction of len n (if None fourier model will be used)
        """
        super().__init__(df, n, column_name=column_name)

        self.column_name = column_name
        self.level = level
        self.wt = wt
        self.n = n
        self.params = params

        if forecast is None:
            forecast = get_predict
        self.forecast = forecast

    def predict(self, X):
        X = X[self.column_name].values
        return predict_wavelet(X, self.n, forecast=self.forecast, wt=self.wt, level=self.level)


if __name__ == '__main__':
    X = np.cos(np.linspace(1, 200, 200)) + np.cos(20 * np.linspace(10, 100, 200)) + np.cos(
        5 * np.linspace(1, 100, 200)) * np.sin(np.linspace(50, 100, 200)) + 4
    X = pd.DataFrame(X, columns=['price'])
    n_harm = list(range(1, 10))
    n = 140
    wt = 'coif1'
    level = None

    # model = Wavelet(df, n=n, wt=wt, level=level, n_harm=n_harm, trend_deg=list(range(5)))
    model = Wavelet(X, n=10)
    pred = model.predict(X)

    print(pred)
    # plt.plot(range(len(df)), df.values)
    # plt.plot(range(len(df), len(df) + 140), pred)
    # plt.show()
