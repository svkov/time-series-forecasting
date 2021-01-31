from sklearn.metrics import accuracy_score

from src.trade import Simulation
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings

warnings.simplefilter("ignore")


def get_model(model_type='logistic'):
    models = {
        'logistic': LogisticRegression(),
        'random_forest': RandomForestClassifier(),
        'ridge': RidgeClassifier(),
    }
    return models[model_type]


def get_x(train, test, x_columns):
    x_train = train[x_columns].to_numpy().reshape(-1, len(x_columns))
    x_test = test[x_columns].to_numpy().reshape(-1, len(x_columns))
    return x_train, x_test


def get_y(train, test):
    return train.label, test.label


def get_columns(window_len):
    return ['target', 'diff'] + [f'n{i + 1}' for i in range(window_len)]


def get_cv_train_test(df, train_size=0.9):
    df = df.dropna()
    start_pivot = int(len(df) * train_size)
    for pivot in range(start_pivot, len(df)):
        train = df[:pivot]
        test = df[pivot:]
        yield train, test


def fit_model(x_train, y_train, model_type='logistic'):
    model = get_model(model_type)
    model.fit(x_train, y_train)
    return model


def get_x_y_train_test(train, test, window_len):
    x_columns = get_columns(window_len)
    x_train, x_test = get_x(train, test, x_columns)
    y_train, y_test = get_y(train, test)
    return x_train, x_test, y_train, y_test


def get_accuracy(train, test, window_len, model_type='logistic'):
    x_train, x_test, y_train, y_test = get_x_y_train_test(train, test, window_len)
    model = fit_model(x_train, y_train, model_type)
    y_pred = model.predict(x_test)
    return accuracy_score(y_pred, y_test)


def best_window_len_cv(df, window_len, model_type='logistic'):
    #     train, test = train_test_split(df)
    accuracies = []
    for train, test in get_cv_train_test(df, train_size=0.95):
        accuracy = get_accuracy(train, test, window_len, model_type)
        accuracies.append(accuracy)
    return sum(accuracies) / len(accuracies)
