import click
import pandas as pd

from src.models.arima import ARIMA


def forecast_arima(df: pd.DataFrame, n_pred: int, date_start: str, date_end: str) -> pd.DataFrame:
    model = ARIMA(df, n=n_pred, column_name='Close')
    return model.predict_for_report(df, date_start, date_end)


@click.command()
@click.option('--input')
@click.option('--output')
@click.option('--n_pred')
@click.option('--date_start')
@click.option('--date_end')
def forecast_click(input, output, n_pred, date_start, date_end):
    n_pred = int(n_pred)
    df = pd.read_csv(input, index_col='Date', parse_dates=True)
    pred = forecast_arima(df, n_pred, date_start, date_end)
    pred.to_csv(output)


if __name__ == '__main__':
    forecast_click()  # noqa
