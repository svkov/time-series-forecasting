import pmdarima as pmd
import pandas as pd
from src.utils import transform_date_start

import warnings
warnings.filterwarnings("ignore")


def fit_model(pivoted_df, n):
    train = pivoted_df.iloc[:-n]
    train_ex = train.drop(['price'], axis=1)
    model = pmd.auto_arima(train['price'], train_ex)
    return model


def predict_model(model, pivoted_df, n):
    train = pivoted_df.iloc[:-n]
    last_val = train.drop(['price'], axis=1).values[-1].reshape(1, -1).tolist()
    last_ex = last_val * n
    return model.predict(n, last_ex)


def update_model(model, pivoted_df, n):
    train = pivoted_df.iloc[:-n]
    new_price = train['price'].values[-1]
    new_val = train.drop(['price'], axis=1).values[-1].reshape(1, -1).tolist()
    model.update(new_price, new_val)
    return model


def arima_predict_for_report(df, start_date, end_date, n):
    pivoted_df = df[:start_date]

    model = fit_model(pivoted_df, n)
    predictions = []

    for pivot in pd.date_range(start_date, end_date):
        pivoted_df = df[:pivot]
        model = update_model(model, pivoted_df, n)
        prediction = predict_model(model, pivoted_df, n)
        predictions.append(prediction)

    columns = [f'n{i + 1}' for i in range(n)]

    date_start = transform_date_start(start_date, n)
    date_end = transform_date_start(end_date, n)
    index = pd.date_range(date_start, date_end)

    return pd.DataFrame(predictions, columns=columns, index=index)
