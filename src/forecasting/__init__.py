import click
import pandas as pd
from src.models.model import Model


def forecast(model: Model, df: pd.DataFrame, n_pred: int, date_start: str) -> pd.DataFrame:
    df = df[:date_start]
    pred = model.predict(df)
    dates = generate_date_range(date_start, n_pred)
    return generate_df(pred, dates)


def generate_df(pred, dates):
    pred_df = pd.DataFrame({
        'Prediction': pred,
        'Date': dates
    })
    pred_df = pred_df.set_index('Date')
    return pred_df


def generate_date_range(date_start, n_pred):
    return pd.date_range(date_start, periods=n_pred)
