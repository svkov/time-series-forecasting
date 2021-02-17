import click
import pandas as pd

from src.forecasting.forecasting_methods import *
from src.utils import send_to_telegram_if_fails
from src.utils.click_commands import InputCommand, ModelCommand


@send_to_telegram_if_fails
@click.command(cls=ModelCommand)
def forecast_click(input, output, n_pred, date_start, date_end, model, ticker, **kwargs):
    n_pred = int(n_pred)
    df = pd.read_csv(input, index_col='Date', parse_dates=True)
    df = df.filter(like=ticker, axis=1)
    # if model != 'var':
    #     df['Close'] = df.filter(like='Close', axis=1)
    pred = models[model].forecast(df, n_pred, f'{ticker} Close', date_start, date_end)
    pred.to_csv(output)


if __name__ == '__main__':
    forecast_click() # noqa