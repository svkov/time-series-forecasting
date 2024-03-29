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
def aggregate_forecast(input, output, **kwargs):
    forecasts = input.split()

    final_df = pd.DataFrame()
    for path in forecasts:
        df = pd.read_csv(path, index_col=0, parse_dates=True)

        new_path, ticker = os.path.split(path)
        ticker = ticker.replace('.csv', '')
        model_name = os.path.split(new_path)[-1]
        mapper = {column: f'{model_name}{column.replace(ticker, "")}' for column in df.columns}
        df = df.rename(columns=mapper)
        final_df = pd.concat([final_df, df], axis=1)
    final_df.to_csv(output)


if __name__ == '__main__':
    aggregate_forecast() # noqa
