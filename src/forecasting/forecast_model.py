import click
import pandas as pd

from src.forecasting.forecasting_methods import *

models = {
    'arima': forecast_arima,
    'baseline': forecast_baseline,
    'test': forecast_test,
    'fourier': forecast_fourier,
    'wavelet': forecast_wavelet
}


@click.command()
@click.option('--input')
@click.option('--output')
@click.option('--n_pred')
@click.option('--date_start')
@click.option('--date_end')
@click.option('--model')
def forecast_click(input, output, n_pred, date_start, date_end, model):
    n_pred = int(n_pred)
    df = pd.read_csv(input, index_col='Date', parse_dates=True)
    pred = models[model](df, n_pred, date_start, date_end)
    pred.to_csv(output)


if __name__ == '__main__':
    forecast_click() # noqa