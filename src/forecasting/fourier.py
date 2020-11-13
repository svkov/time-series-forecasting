import click
import pandas as pd
from src.models.fourier import Fourier


def forecast(df: pd.DataFrame, n_pred: int, date_start: str) -> pd.DataFrame:
    df = df[:date_start]

    model = Fourier(df, n_harm=1, trend_deg=0, n=n_pred, column_name='Close')
    pred = model.predict(df)
    dates = pd.date_range(date_start, periods=n_pred)
    pred_df = pd.DataFrame({
        'Prediction': pred,
        'Date': dates
    })
    pred_df = pred_df.set_index('Date')
    return pred_df


@click.command()
@click.option('--input')
@click.option('--output')
@click.option('--n_pred')
@click.option('--date_start')
def forecast_click(input, output, n_pred, date_start):
    n_pred = int(n_pred)
    df = pd.read_csv(input, index_col='Date', parse_dates=True)
    pred = forecast(df, n_pred, date_start)
    pred.to_csv(output)


if __name__ == '__main__':
    forecast_click()  # noqa
