import click
import os
import pandas as pd

from src.utils import send_to_telegram_if_fails
from src.utils.click_commands import InputCommand


def filename_without_extension(path_to_file):
    filename = os.path.split(path_to_file)[-1]
    return os.path.splitext(filename)[0]


def read_dir_csv(files, index_col):
    data = {}
    for file in files:
        df = pd.read_csv(file, index_col=index_col, parse_dates=True)
        key = filename_without_extension(file)
        data[key] = df
    return data


def transform_interim_to_processed(data):
    res = pd.DataFrame()
    needed_columns = ['Open', 'Close', 'High', 'Low', 'Volume']
    for key, df in data.items():
        for column in needed_columns:
            res[f'{key} {column}'] = df[column]
    return res


def aggregate_to_processed(files) -> pd.DataFrame:
    data = read_dir_csv(files, index_col='Date')
    return transform_interim_to_processed(data)


@send_to_telegram_if_fails
@click.command(cls=InputCommand)
def process(input, output, **kwargs):
    files = input.split()
    df = aggregate_to_processed(files)
    df.to_csv(output)


if __name__ == '__main__':
    process()  # noqa
