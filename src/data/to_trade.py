import click
import os
import pandas as pd

from src.trade.prepare_data import prepare_data_without_window
from src.utils import send_to_telegram_if_fails


@send_to_telegram_if_fails
@click.command()
@click.option('--input')
@click.option('--output')
@click.option('--n')
def to_trade(input, output, n):
    n = int(n)
    out_list = os.path.split(output)
    if not os.path.isdir(out_list[0]):
        os.makedirs(out_list[0])
    ticker = out_list[-1].replace('.csv', '')
    prepare_data_without_window(input, instrument=ticker, n=n, bounds=[0.1, 20]).to_csv(output)


if __name__ == '__main__':
    to_trade() # noqa
