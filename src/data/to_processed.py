import click
import os
import pandas as pd


def path_to_file_to_key(path_to_file):
    filename = os.path.split(path_to_file)[-1]
    return os.path.splitext(filename)[0]


def read_data(files):
    data = {}
    for file in files:
        df = pd.read_csv(file, index_col='Date', parse_dates=True)
        key = path_to_file_to_key(file)
        data[key] = df
    return data


def aggregate(files) -> pd.DataFrame:
    data = read_data(files)
    res = pd.DataFrame()
    for key, df in data.items():
        res[f'{key} Open'] = df['Open']
        res[f'{key} Close'] = df['Close']
    return res


@click.command()
@click.option('--input')
@click.option('--output')
def process(input, output):
    files = input.split()
    df = aggregate(files)
    df.to_csv(output)


if __name__ == '__main__':
    process()  # noqa
