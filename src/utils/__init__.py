import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime, timedelta
import telegram_send
import yaml


def validate_df(needed_columns, df_columns):
    needed_columns = set(needed_columns)
    df_columns = set(df_columns)

    intersected_columns = needed_columns.intersection(df_columns)
    if len(intersected_columns) != len(needed_columns):
        need_to_append = needed_columns.difference(intersected_columns)
        raise KeyError(f'В DataFrame не хватает столбцов: {" ".join(need_to_append)}')


def train_test_split(x, train_size=0.8):
    n = x.shape[0]
    pivot = int(train_size * n)
    return x[:pivot], x[pivot:]


def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Считает MAPE между y_true и y_pred
    """
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100

def exponential_smoothing(df, column, alpha=0.7):
    series = df[column]
    result = [series.iloc[0]]
    for n in range(1, len(series)):
        result.append(alpha * series.iloc[n] + (1 - alpha) * result[n - 1])
    df_new = pd.DataFrame(df)
    df_new[column] = result
    return df_new


def substract_n_days(date_start: str, n) -> datetime:
    """
    Вычитает n дней из date_start

    Нужно использовать для построении предсказаний
    """
    return datetime.strptime(date_start, '%Y-%m-%d') - timedelta(days=n)


def read_pred_csv(path, n):
    """
    Возвращает DataFrame, начиная с n-ой строки
    """
    return pd.read_csv(path, index_col=0, parse_dates=True).iloc[n:]


def get_nmape(ntest, npred):
    assert ntest.shape == npred.shape, f'Pred and test must have equal shapes, given {ntest.shape}, {npred.shape}'
    mape = []
    for i in range(len(npred)):
        pred_i = npred[i]
        test_i = ntest[i]
        mape.append(mean_absolute_percentage_error(test_i, pred_i))
    return mape


def get_grid_from_dict(kwargs):
    """
    Генерирует сетку параметров из словаря
    Пример:
    Вход:
    {'a': [1, 2], 'b': [3, 4]}
    Выход:
    [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
    """
    names = list(kwargs.keys())
    params = [v for k, v in kwargs.items()]
    params = list(product(*params))
    params = [{names[j]: i[j] for j in range(len(i))} for i in params]
    return params


def fill_dates_monthly(X, date_column='date', n_months_after_last=0):
    """ Add missing months in time series

    Missing date is filled with last day of missing month
    Adds to df column year_month with string with format '%Y-%m'

    :param X: pd.DataFrame with column `date_column`
    :param date_column: columns with dates
    :param n_months_after_last: how many month dates add to the end with NAs
    :return: pd.DataFrame
    """
    X = pd.DataFrame(X)

    X[date_column] = pd.to_datetime(X[date_column])
    X.insert(1, 'year_month', pd.to_datetime(pd.to_datetime(X[date_column]).dt.strftime('%Y-%m')))

    X = X.set_index('year_month')
    idx = pd.date_range(start=X[date_column].iloc[0],
                        end=X[date_column].iloc[-1] + pd.Timedelta(f'{10 + 31 * n_months_after_last}d'), freq='M')
    idx = pd.to_datetime(pd.to_datetime(idx).strftime('%Y-%m'))
    df_new = pd.DataFrame(idx, columns=['year_month']).sort_values(by="year_month").reset_index(drop=True)
    X = pd.merge_asof(df_new, X, on='year_month', tolerance=pd.Timedelta('30d'))
    X.year_month = pd.to_datetime(X.year_month).dt.strftime('%Y-%m')
    return X


def fill_dates_daily(X, date_column='date'):
    """ Add missing days in time series

    :param X: pd.DataFrame with column `date_column`
    :param date_column: columns with dates
    :return: pd.DataFrame
    """
    X = pd.DataFrame(X)
    X[date_column] = pd.to_datetime(X[date_column])
    idx = pd.date_range(start=X[date_column].iloc[0], end=X[date_column].iloc[-1] + pd.Timedelta('0.5d'), freq='d')
    df_new = pd.DataFrame(idx, columns=[date_column]).sort_values(by=date_column).reset_index(drop=True)
    X = pd.merge_asof(df_new, X, on=date_column, tolerance=pd.Timedelta('0.5d'))
    return X


def prepare_test_dataframe(df, date_start, date_end, n):
    """
    date_start - дата на которую надо получить первое n-ое предсказание
    date_end - дата от которой последней надо предсказывать
    n - горизонт прогнозирования
    """
    index = df.loc[date_start:date_end].index
    tests = []
    for date in index:
        test = df.loc[:date, 'price'].interpolate().dropna().values[-n:]
        tests.append(test)
    tests = np.array(tests)

    date_start = substract_n_days(date_start, n)
    date_end = substract_n_days(date_end, n)
    index = df.loc[date_start:date_end].index

    columns = [f'n{i + 1}' for i in range(n)]

    return pd.DataFrame(tests, columns=columns, index=index)


def send_to_telegram_if_fails(func, *args, **kwargs):
    def wrapper():
        try:
            res = func(*args, **kwargs)
            # telegram_send.send(messages=[f'Функция {func.__name__} посчиталась'])
            return res
        except Exception as e:
            telegram_send.send(messages=[f'Что-то пошло не так: {e}', f'Функция {func}'])
            raise e

    return wrapper
