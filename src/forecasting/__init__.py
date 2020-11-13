import click
import pandas as pd
from src.models.model import Model


def forecast(model: Model, df: pd.DataFrame, n_pred: int, date_start: str) -> pd.DataFrame:
    df = df[:date_start]
    pred = model.predict(df)
    dates = pd.date_range(date_start, periods=n_pred)
    pred_df = pd.DataFrame({
        'Prediction': pred,
        'Date': dates
    })
    pred_df = pred_df.set_index('Date')
    return pred_df
