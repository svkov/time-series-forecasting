import numpy as np
import pandas as pd
import os
import sys

import click

from src.forecasting.forecasting_methods import *
from src.models.stacking import Stacking
from src.utils import send_to_telegram_if_fails
from src.utils.click_commands import InputCommand


@send_to_telegram_if_fails
@click.command(cls=InputCommand)
@click.option('--input_all')
@click.option('--models')
@click.option('--ticker')
def forecast_stacking(input, output, logs, input_all, models, ticker):
    models = models.split()

    data = []
    for model in models:
        path = os.path.join(input, model, f'{ticker}.csv')
        print(path)
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if model != 'test':
            data.append(df)

    df = pd.read_csv(input_all, parse_dates=True, index_col=0)
    test_df = pd.DataFrame()
    test_df['price'] = df[f'{ticker} Close']
    print(data)
    model = Stacking(test_df, *data)

    pred = model.predict_for_report(test_df, *data)
    mapper = {column: f'{ticker} Close {column}' for column in pred.columns}
    pred = pred.rename(columns=mapper)
    pred.to_csv(output)


if __name__ == '__main__':
    forecast_stacking() # noqa