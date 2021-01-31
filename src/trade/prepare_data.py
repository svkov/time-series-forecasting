import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

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


def prepare_data(path='../data/processed/all.csv', instrument='CL=F', n=3, window_len=10):
    df = prepare_data_without_window(path, instrument, n)
    df = generate_window(df, window_len=window_len)
    return df


def prepare_data_without_window(path='../data/processed/all.csv', instrument='CL=F', n=3):
    df = read_data(path)
    res = minimize_scalar(get_imbalance_by_thresh, method='bounded', args=(df, instrument, n), bounds=[0.3, 50])
    thresh = res.x
    return make_labels(df, f'{instrument} Close', n=n, thresh=thresh)

