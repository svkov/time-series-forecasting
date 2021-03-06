import click
import json
import pandas as pd

from src.trade.model import choose_best_window_size


def best_window_size(df, n, model_types):
    values = {}
    for model_type in model_types:
        acc, window = choose_best_window_size(df, n, model_type)
        values[model_type] = {'accuracy': acc, 'window': window}
    return values


def save_json(dict_, output):
    with open(output, 'w') as file:
        content = json.dumps(dict_)
        file.write(content)


@click.command()
@click.option('--input')
@click.option('--output')
@click.option('--n')
@click.option('--model_types')
def find_best_window_size_trade(input, output, n, model_types):
    n = int(n)
    model_types = model_types.split()
    df = pd.read_csv(input, index_col=0, parse_dates=True)
    values = best_window_size(df, n, model_types)
    save_json(values, output)


if __name__ == '__main__':
    find_best_window_size_trade()  # noqa
