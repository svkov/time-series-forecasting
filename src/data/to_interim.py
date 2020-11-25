import click
import pandas as pd

from src.utils import send_to_telegram_if_fails


@send_to_telegram_if_fails
@click.command()
@click.option('--input')
@click.option('--output')
def interim_data(input, output):
    df = pd.read_csv(input, index_col='Date', parse_dates=True)
    date_start, date_end = df.index[0], df.index[-1]
    dates = pd.date_range(date_start, date_end)
    df = df.reindex(dates).ffill().bfill()
    df['Date'] = df.index
    df.set_index('Date', inplace=True)
    df.to_csv(output)


if __name__ == '__main__':
    interim_data() # noqa