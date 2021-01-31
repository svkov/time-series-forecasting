import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score

from src.trade import Simulation
from src.trade.model import fit_model, get_x_y_train_test

warnings.simplefilter("ignore")


def get_label_by_diff(d, thresh):
    if abs(d) < thresh:
        return 'hold'
    if d < -thresh:
        return 'sell'
    if d > thresh:
        return 'buy'


def diff_percent(df, n=10):
    diff_percent = []
    for i in range(len(df['target']) - n):
        d_p = (df['target'][i + n] - df['target'][i]) / df['target'][i] * 100
        diff_percent.append(d_p)
    diff_percent += [np.nan for i in range(n)]
    df['diff_percent'] = diff_percent
    return df


def make_labels(df, column, n=10, thresh=2):
    df.loc[:, 'target'] = df[column]
    df[df['target'] <= 0] = np.nan
    df.dropna(inplace=True)
    df = df[['target']]
    df.loc[:, 'diff'] = df.target.diff(n)
    df = diff_percent(df, n)
    df.loc[:, 'label'] = df['diff_percent'].apply(lambda x: get_label_by_diff(x, thresh))
    return df


def generate_window(df, window_len=5):
    for i in range(window_len):
        df[f'n{i + 1}'] = df['target'].shift(i + 1)
    return df


def calculate_imbalance(df, column='label', num_of_classes=3):
    N = len(df)
    ideal_param = 1 / num_of_classes
    error = 0
    for name, g in df.groupby(column):
        n_i = len(g)
        ratio = n_i / N
        error += (ideal_param - ratio) ** 2
    return error / N


def get_imbalance_by_thresh(thresh, df, instrument='CL=F', n=3):
    df = make_labels(df, f'{instrument} Close', n=n, thresh=thresh)
    return calculate_imbalance(df)


def read_data(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)


def prepare_data(path='../data/processed/all.csv', instrument='CL=F', n=3, window_len=10, bounds=[0.1, 50]):
    df = prepare_data_without_window(path, instrument, n, bounds=bounds)
    df = generate_window(df, window_len=window_len)
    return df


def prepare_data_without_window(path='../data/processed/all.csv', instrument='CL=F', n=3, bounds=None):
    if bounds is None:
        bounds = [0.1, 50]
    df = read_data(path)
    res = minimize_scalar(get_imbalance_by_thresh, method='bounded', args=(df, instrument, n), bounds=bounds)
    thresh = res.x
    print(thresh)
    return make_labels(df, f'{instrument} Close', n=n, thresh=thresh)


def train_test_split(df, train_size=0.9):
    pivot = int(len(df) * train_size)
    train = df[:pivot].dropna()
    test = df[pivot:].dropna()
    return train, test


def add_results_to_df(df_window, window_len, model_type='logistic'):
    train, test = train_test_split(df_window, train_size=0.9)
    x_train, x_test, y_train, y_test = get_x_y_train_test(train, test, window_len)
    model = fit_model(x_train, y_train, model_type)
    y_pred = model.predict(x_test)
    print('Accuracy:', accuracy_score(y_pred, y_test))
    df_window['predicted_label'] = pd.Series(y_pred, index=df_window.index[-y_pred.shape[0]:])
    df_window.dropna(inplace=True)
    return df_window


def get_simulation_results(df_window, cap=1000):
    res, log = Simulation(cap).play_simulation(df_window, label_column='predicted_label')
    best_res, log = Simulation(cap).play_simulation(df_window, label_column='label')
    return res, best_res
