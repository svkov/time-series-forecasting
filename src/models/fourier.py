import os
import pandas as pd
import numpy as np

from numpy import fft

from src.utils import get_grid_from_dict, get_nmape, transform_date_start

from src.models.model import Model


def get_predict(x, n=14, n_harm=1000, trend_deg=1):
    pred = FourierPredictor.fourier_extrapolation(x, n, n_harm, trend_deg=trend_deg)[len(x):]
    bias = abs(x[-1] - pred[0])
    pred += bias  # Сдвигаем на текущий уровень
    return pred


def get_trend(x, y, trend_deg):
    poly_coef = np.polyfit(x, y, trend_deg)
    new_y = np.zeros_like(x)
    for i in range(trend_deg):  # Последний коэфициент не прибавляем умышленно
        new_y += poly_coef[i] * np.power(x, trend_deg - i)
    return new_y, poly_coef


def set_trend(x, poly_coef):
    trend_deg = len(poly_coef) - 1
    trend = np.zeros_like(x)
    for i in range(trend_deg):  # Последний коэфициент не прибавляем умышленно
        trend += poly_coef[i] * np.power(x, trend_deg - i)
    return trend


class Fourier(Model):
    default_n_harm = list(range(1, 10))
    default_trend_deg = list(range(4))

    def __init__(self, df_train: pd.DataFrame, n_harm=None, trend_deg=None, n=14, train_len=None, verbose=False,
                 models_dir=None, column_name='price', **kwargs):
        """
        Инициализирует модель и подбирает оптимальные параметры

        :param df_train: DataFrame с колонкой price
        :param n_harm: Количество гармоник.
         Если указывается None, то подбирается оптимальное n_harm.
         Если указывается список, то подбирается оптимальное значение из списка.
         Если указывается число, то выбирается это число в качестве оптимального значения параметра
        :param trend_deg: Степень тренда.
         Если указывается None, то подбирается оптимальное trend_deg.
         Если указывается список, то подбирается оптимальное значение из списка.
         Если указывается число, то выбирается это число в качестве оптимального значения параметра
        :param n: Горизонт прогнозирования
        :param train_len: Длина обучающей выборки.
         Если None, то будет взята половина выборки df_train
        :param verbose: Вывод работы модели
        :param kwargs:
        """
        super().__init__(n, verbose, **kwargs)

        if models_dir:
            if not os.path.isdir(models_dir):
                os.makedirs(models_dir, exist_ok=True)
            self.path_to_params = f'{models_dir}params.csv'

        self.column_name = column_name
        self.check_price_in_df(df_train, self.column_name)
        self.check_trend_deg(trend_deg)
        self.n = n
        self.df_train = df_train[[self.column_name]].interpolate().dropna()

        self.predictor = FourierPredictor(self.df_train, self.n, column_name=self.column_name)

        self.train_len = self.preproc_train_len(train_len)
        self.params = self.select_best_params(n_harm, trend_deg)

    def predict(self, df):
        self.check_price_in_df(df, self.column_name)
        return self.predictor.predict(df[self.column_name].values, **self.params)

    @property
    def start_date_train(self):
        return self.df_train.index[-self.train_len]

    @property
    def end_date_train(self):
        return self.df_train.index[-self.n]

    def predict_for_report(self, df, date_start, date_end):
        self.check_price_in_df(df, self.column_name)
        predictor = FourierPredictor(df, self.n, column_name=self.column_name)
        preds = predictor.get_predictions(pd.date_range(date_start, date_end), **self.params)
        columns = self.get_columns_for_report()
        index = self.get_index_for_report(df, date_start, date_end)
        return pd.DataFrame(preds, columns=columns, index=index)

    def get_index_for_report(self, df, date_start, date_end):
        date_start_report = transform_date_start(date_start, self.n)
        date_end_report = transform_date_start(date_end, self.n)
        return df[date_start_report:date_end_report].index

    def get_columns_for_report(self):
        return [f'n{i + 1}' for i in range(self.n)]

    def select_best_params(self, n_harm, trend_deg):
        try:
            return self.load_params()
        except:
            self._print(f'Не нашлось сохраненных параметров, считаю новые...')

        params = self.make_params_dict_from_values(n_harm=self.get_n_harm_list(n_harm),
                                                   trend_deg=self.get_trend_deg_list(trend_deg))
        self._print(f'Все параметры: {params}')
        params = self.get_optimal_params(params)
        self._print(f'Оптимальные параметры: {params}')
        self.save_params(params)
        return params

    def save_params(self, params: dict):
        """
         Сохраняет параметры в csv формате.
         Индекс - последняя дата в параметрах
         Колонки - ключи params
        :param params: словарь с ключами n_harm и trend_deg.
         Значение каждого ключа - целое чило
        """
        if hasattr(self, 'path_to_params'):
            index = self.df_train.index[-1]
            pd.DataFrame(params, index=[index]).to_csv(self.path_to_params)

    def load_params(self):
        params = pd.read_csv(self.path_to_params, index_col=0, parse_dates=True)
        trend_deg = params.trend_deg[0]
        n_harm = params.n_harm[0]
        return self.make_params_dict_from_values(n_harm=n_harm, trend_deg=trend_deg)

    @staticmethod
    def make_params_dict_from_values(**kwargs):
        return kwargs

    def preproc_train_len(self, train_len):
        if train_len is None:
            return len(self.df_train) // 2
        if isinstance(train_len, int):
            return train_len
        raise ValueError(f'train_len должен быть int или None, вмест этого {type(train_len)}')

    @staticmethod
    def check_price_in_df(df, column_name):
        if column_name not in df.columns:
            raise KeyError(f'В данных нет столбца price, вместо него: {df.columns}')

    @staticmethod
    def check_trend_deg(trend_deg):
        if isinstance(trend_deg, list):
            return
        if trend_deg and trend_deg < 0:
            raise ValueError(f'Степень тренда должна быть положительна или 0')

    def get_n_harm_list(self, n_harm):
        if n_harm is None:
            return self.default_n_harm
        return self.preprocess_param(n_harm)

    def get_trend_deg_list(self, trend_deg):
        if trend_deg is None:
            return self.default_trend_deg
        return self.preprocess_param(trend_deg)

    @staticmethod
    def preprocess_param(n_harm):
        if isinstance(n_harm, int):
            n_harm = [n_harm]
        if not isinstance(n_harm, list):
            raise ValueError(f'Параметр может быть числом, списком или None, вместо этого {n_harm}')
        return n_harm

    def get_optimal_params(self, params):
        metrics_history = []
        params_grid = get_grid_from_dict(params)
        for params in params_grid:
            mape = self.score_params(params)
            metrics_history.append(mape)
            self._print(f'Для параметров: {params} MAPE: {mape}')
        optimal_param = params_grid[np.array(metrics_history).argmin()]
        return optimal_param

    def score_params(self, params):
        tests, preds = self.predict_all_dates_on_params(**params)
        return self.get_worst_mape(tests, preds)

    @staticmethod
    def get_worst_mape(tests, preds):
        return max(get_nmape(tests, preds))

    def predict_all_dates_on_params(self, **params):
        return self.get_tests(), self.predictor.get_predictions(self.train_date_range, **params)

    def get_tests(self):
        tests_list = []
        for pivot in self.train_date_range:
            test = self.df_train.loc[:pivot, self.column_name].values[-self.n:]  # последние n значений
            tests_list.append(test)
        return np.array(tests_list)

    @property
    def train_date_range(self):
        return self.df_train.loc[self.start_date_train:self.end_date_train].index

    def __repr__(self):
        return f'Fourier: {self.params}'


class FourierPredictor:

    def __init__(self, df_train, n, column_name='price'):
        self.df_train = df_train
        self.n = n
        self.column_name = column_name

    def get_predictions(self, pivot_list, **params):
        predictions_list = []
        for pivot in pivot_list:
            prediction = self.get_prediction_by_pivot(pivot, **params)
            predictions_list.append(prediction)
        return np.array(predictions_list)

    def get_prediction_by_pivot(self, pivot, **params):
        train = self.df_train.loc[:pivot, self.column_name].values[:-self.n]  # все, кроме последних n значений
        if len(train) == 0:
            raise ValueError('Пустая обучающая выборка!')
        return self.predict(train, **params)

    def predict(self, x, n_harm=1000, trend_deg=1):
        pred = self.fourier_extrapolation(x, self.n, n_harm, trend_deg=trend_deg)[len(x):]
        bias = abs(x[-1] - pred[0])
        pred += bias  # Сдвигаем на текущий уровень
        return pred

    @staticmethod
    def fourier_extrapolation(x, n_predict, n_harm=10, trend_deg=1):
        n = x.size
        t = np.arange(0, n).astype(float)
        trend, poly_coef = get_trend(t, x, trend_deg)
        x_notrend = x - trend

        t = np.arange(0, n + n_predict).astype(float)
        new_trend = set_trend(t, poly_coef)

        x_freqdom = fft.fft(x_notrend)
        f = fft.fftfreq(n)
        indexes = list(range(n))
        indexes.sort(key=lambda i: np.absolute(x_freqdom[i]))
        indexes.reverse()
        restored_sig = np.zeros(t.size)
        for i in indexes[2:1 + n_harm * 2]:
            ampli = np.absolute(x_freqdom[i]) / n
            phase = np.angle(x_freqdom[i])
            restored_sig += ampli * 2 * np.cos(2 * np.pi * f[i] * t + phase)
        return restored_sig + new_trend


if __name__ == "__main__":
    a = 100
    b = 110
    step = 2000
    X = np.power(np.linspace(a, b, step), 2) + np.cos(np.linspace(a, b, step)) + 5 + np.sin(
        2 * np.linspace(a, b, step)) + np.cos(5 * np.linspace(a, b, step))
    index = pd.date_range('2016-01-01', periods=step)
    X = pd.DataFrame(X, columns=['price'], index=index)
    model = Fourier(X, n=14, n_harm=[1, 2, 4], trend_deg=1, verbose=True, train_len=400)
    pred = model.predict(X)
    # plt.plot(range(len(X)), X.values)
    # plt.plot(range(len(X), len(X) + 140), pred)
    print(model)

    res = model.predict_for_report(X, '2020-03-10', '2020-05-20')
    print(res)
    # plt.show()
