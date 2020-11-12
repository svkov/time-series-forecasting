import click
import pandas as pd
import numpy as np
from utils import read_pred_csv, mean_absolute_percentage_error, prepare_test_dataframe, transform_date_start
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from forecasting.model import Model
import matplotlib.pyplot as plt
from typing import List, Iterable, Dict, Tuple


class Stacking(Model):

    def __init__(self, test: pd.DataFrame, *predictions: pd.DataFrame, model=None, verbose=False, **kwargs):
        """

        :param test: исходный датафрейм с колонкой price и датой в индексе
        :param predictions: датафреймы с колонками n1, n2, ... и датами в индексах
        :param model: sklearn-модель (необходимо передать класс, не создавая объект)
        :param verbose:
        :param kwargs:
        """
        self.validator = StackingInputValidator(test, *predictions)
        super().__init__(self.validator.get_n(), verbose)
        self.stacking_predictor = StackingPredictor(self.predictions_array, self.test_array, model=model, **kwargs)

    @property
    def predictions_array(self) -> np.ndarray:
        return self.validator.predictions_array

    @property
    def test_array(self) -> np.ndarray:
        return self.validator.test_array

    def predict(self, predictions_array: Iterable) -> np.ndarray:
        """
        Предсказывает значение ряда по заданным предсказаниям других моделей.
        :param predictions_array: Вектор предсказаний других моделей размерности (n, n_models).
         Столбец - предсказание одной модели
         Необходимо передавать предсказания в том же порядке, в котором они были переданы при создании модели
        :return: Вектор из n чисел
        """
        predictions_array = np.array(predictions_array)
        return self.stacking_predictor.predict(predictions_array)

    @staticmethod
    def predict_for_report(test: pd.DataFrame, *predictions: pd.DataFrame, model=None, **model_params) -> pd.DataFrame:
        validator = StackingInputValidator(test, *predictions)
        stacking_predictor = StackingPredictor(validator.predictions_array, validator.test_array, model=model, **model_params)
        result = stacking_predictor.predict_for_report()
        columns = [f'n{i + 1}' for i in range(validator.get_n())]
        index = validator.test.loc[validator.date_start:validator.date_end].index
        return pd.DataFrame(result, columns=columns, index=index)


class StackingInputValidator:

    def __init__(self, test: pd.DataFrame, *predictions: pd.DataFrame):
        self.predictions = list(predictions)
        self.test = self.process_test(test)
        self.check_input()

        self.predictions_array = self.models_pred_to_array()
        self.test_array = self.get_test_array()

    def get_test_array(self) -> np.ndarray:
        return self.test_to_array(self.test, self.get_n())

    def check_input(self) -> None:
        self.check_predictions_length()
        self.check_index_consistency()
        self.check_test_and_prediction_consistency()

    def check_predictions_length(self) -> None:
        m = len(self.any_prediction)
        for i in self.predictions:
            if len(i) != m:
                raise ValueError(f'Все предсказания должны быть одной длины. Сейчас {m} и {len(i)}')

    def get_n(self) -> int:
        return self.any_prediction.shape[1]

    def models_pred_to_array(self) -> np.ndarray:
        return self.predictions_to_array(self.predictions)

    @property
    def date_start(self) -> pd.Timestamp:
        return self.preprocess_date(self.test.index[0])

    @property
    def date_end(self) -> pd.Timestamp:
        return self.preprocess_date(self.test.index[-1])

    def preprocess_date(self, date: pd.Timestamp) -> pd.Timestamp:
        return pd.Timestamp(transform_date_start(date.strftime('%Y-%m-%d'), self.get_n()))

    @property
    def any_prediction(self) -> pd.DataFrame:
        return self.predictions[0]

    def check_index_consistency(self) -> None:
        m = self.any_prediction.index
        for pred in self.predictions:
            m = m.intersection(pred.index)
        if (len(m) != len(self.any_prediction)) or (m != self.any_prediction.index).any():
            raise ValueError(f'У предсказаний разные индексы')

    def check_test_and_prediction_consistency(self) -> None:
        test_and_prediction_index = self.test.index.intersection(self.any_prediction.index)
        if len(test_and_prediction_index) != len(self.any_prediction):
            prediction_date = self.any_prediction.index[-1].strftime('%Y-%m-%d')
            raise ValueError(f'''У исходных данных и предсказаний разные индексы. 
            Индексы должны быть одинаковыми везде, кроме последних n значений. 
            Если исходные данные кончаются на дате m, то предсказания должны кончаться на дате m-n. 
            Тест кончается на {self.date_end.strftime('%Y-%m-%d')}, предсказания на {prediction_date}, n={self.get_n()}''')

    def process_test(self, test: pd.DataFrame) -> pd.DataFrame:
        try:
            start_date = self.any_prediction.index[0]
            end_date = self._shift_date(self.any_prediction.index[-1], n=self.get_n())
            return test.loc[start_date:end_date]
        except ValueError as e:
            print(e)
            raise ValueError('Ошибка в датах предсказаний или теста')

    @staticmethod
    def test_to_array(test, n, date_start=None, date_end=None) -> np.ndarray:
        if date_start is None:
            date_start = StackingInputValidator.date_start_from_test(test, n)
        if date_end is None:
            date_end = StackingInputValidator.date_end_from_test(test)
        return prepare_test_dataframe(test, date_start, date_end, n).values

    @staticmethod
    def date_start_from_test(test, n) -> str:
        return test.index[n].strftime('%Y-%m-%d')

    @staticmethod
    def date_end_from_test(test) -> str:
        return test.index[-1].strftime('%Y-%m-%d')

    @staticmethod
    def _shift_date(date, n: int) -> pd.Timestamp:
        """
        Сдвигает дату на n дней вперед
        :param date: либо Timestamp, либо строка в формате %Y-%m-%d
        :param n: на сколько дней сдвигать
        :return: Сдвинутая дата
        """
        if not isinstance(date, str):
            date = date.strftime('%Y-%m-%d')
        return pd.Timestamp(transform_date_start(date, -n))

    @staticmethod
    def predictions_to_array(predictions) -> np.ndarray:
        arr = []
        for pred in predictions:
            arr.append(pred.values)
        return np.array(arr)


class StackingPredictor:
    models = []  # Модели для каждого дня

    def __init__(self, predictions_array, test_array, model=None, **kwargs):
        self.predictions_array = predictions_array
        self.test_array = test_array
        self.n = self.predictions_array.shape[2]
        self.check_predictions()
        self.check_test()

        self.model = self.process_model(model)
        self.model_params = self.process_model_params(model, kwargs)

        self.models = [self.model(**self.model_params) for _ in range(self.n)]
        self.model_for_report = self.model(**self.model_params)
        self.fit_models()

    @staticmethod
    def process_model(model):
        if model is None:
            return RandomForestRegressor
        return model

    @staticmethod
    def process_model_params(model, params) -> Dict:
        if model is None:
            params = {'oob_score': True}
        return params

    def check_predictions(self) -> None:
        if len(self.predictions_array.shape) != 3:
            raise ValueError(f'Размерность predictions_array должна быть (k, m, n), вместо этого {self.predictions_array.shape}')

    def check_test(self) -> None:
        if self.test_array.shape != self.predictions_array[0, :, :].shape:
            raise ValueError(f'Размерность каждого предсказания должна совпадать с test')

    def fit_models(self) -> None:
        for i in range(self.n):
            self.models[i].fit(self.predictions_array[:, :, i].T, self.test_array[:, i])

    def predict(self, predictions_array: np.ndarray) -> np.ndarray:
        """
        Предсказывает значение ряда по заданным предсказаниям других моделей.
        :param predictions_array: Вектор предсказаний других моделей размерности (n, n_models).
         Столбец - предсказание одной модели
         i-ая строка - предсказание модели на i-ый день
         Необходимо передавать предсказания в том же порядке, в котором они были переданы при создании модели
        :return: Вектор из n чисел
        """
        prediction = []
        for day in range(predictions_array.shape[0]):
            prediction.append(self._predict(predictions_array[day], day))
        return np.array(prediction)

    def _predict(self, x: Iterable, n: int) -> int:
        """
        x - вектор размерности k (предсказания других моделей)
        n - на какой день надо сделать предсказание
        """
        x = np.array(x).reshape(1, -1)
        return self.models[n].predict(x)

    def predict_for_report(self) -> np.ndarray:
        predictions = []
        for pivot in range(self.n, self.predictions_array.shape[1]):
            pred = self.predict_all_day_with_pivot(pivot).tolist()
            predictions.append(pred)
        predictions = self._append_zero_rows(predictions)
        return np.array(predictions)

    def predict_all_day_with_pivot(self, pivot: int) -> np.ndarray:
        prediction = []
        for day in range(self.n):
            prediction.append(self.predict_i_day_with_pivot(pivot, day))
        return np.array(prediction)

    def predict_i_day_with_pivot(self, pivot: int, day: int) -> np.ndarray:
        train, test = self.select_train_test_data_by_day(day)
        x_train = train[:pivot, :]
        y_train = test[:pivot]
        self.model_for_report.fit(x_train, y_train)
        x_test = train[pivot:pivot + 1, :]
        return self.model_for_report.predict(x_test)

    def select_train_test_data_by_day(self, day: int) -> (np.ndarray, np.ndarray):
        train = self.predictions_array[:, :, day].T  # (m, 3)
        test = self.test_array[:, day]  # (m,)
        return train, test

    def _append_zero_rows(self, predictions):
        """
        Первые n строчек предсказания пустые, так как нужны данные для обучения в начале
        :return: Добавляет матрицу нулей в начало предсказания
        """
        zero_rows = np.zeros((self.n, self.n, 1)).tolist()
        return np.array(zero_rows + predictions)[:, :, 0]


if __name__ == "__main__":
    n = 3  # Горизонт прогнозирования
    m = 20  # Количество точек данных

    # Смоделируем предсказания моделей на n дней вперед от каждой даты
    x = np.arange(m - n)
    index = pd.date_range('2020-01-01', periods=m - n)  # Кончается 17-ого числа, последний прогноз на 20-ое

    y1 = np.array([5 * x, 2 + x, x]).T
    y2 = np.array([1 + 2 * x * x, x + 5, x]).T
    y3 = np.array([x * x * np.ones_like(x), x, x * x]).T

    columns = ['n1', 'n2', 'n3']
    y1 = pd.DataFrame(y1, index=index, columns=columns)
    y2 = pd.DataFrame(y2, index=index, columns=columns)
    y3 = pd.DataFrame(y3, index=index, columns=columns)

    # Сформируем фактические значения
    index = pd.date_range('2020-01-01', periods=m)
    x = np.arange(m)
    test = pd.DataFrame(x, index=index, columns=['price'])  # Кончается 20-го

    # Для простоты возьмем линейную регрессию
    # По умолчанию RandomForest, в проде модель не надо указывать
    model = LinearRegression

    # Обучаем модель
    stacking = Stacking(test, y1, y2, y3, model=model)

    # В проде не используется:
    # Предсказываем на исторических значениях
    # Первые n значений нулевые, так как на них модель обучается
    pred_stacking = stacking.predict_for_report(test, y1, y2, y3)  # (m, n)
    assert (pred_stacking.index == y1.index).all(), f'Индексы должны совпадать! {pred_stacking.index} и {y1.index}'
    pred_stacking = pred_stacking.values

    plt.figure(figsize=(10, 5))
    legend = []
    preds = [y1.values, y2.values, y3.values]
    for i in range(len(preds)):
        plt.plot(preds[i][:, -1])
        legend.append(f'pred {i + 1}')
    plt.plot(test.values)
    legend.append('test')

    plt.plot(pred_stacking[:, -1])
    legend.append('stacking')
    plt.legend(legend)
    plt.show()

    # В проде используется:
    # Предсказываем в будущее
    pred1 = np.array([30, 40, 50]).reshape(1, -1)  # (1, 3) - предсказания моделей
    pred2 = np.array([40, 50, 60]).reshape(1, -1)

    preds = [pred1, pred2]
    prediction = stacking.predict(preds)

    print(f'Pred for {pred1[0]}: {prediction[0]}')
    print(f'Pred for {pred2[0]}: {prediction[1]}')
