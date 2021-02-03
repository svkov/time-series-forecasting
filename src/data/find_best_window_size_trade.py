import click
import json
import pandas as pd

from src.trade.model import choose_best_window_size


@click.command()
@click.option('--input')
@click.option('--output')
@click.option('--n')
@click.option('--model_types')
def best_window(input, output, n, model_types):
    n = int(n)
    model_types = model_types.split()
    df = pd.read_csv(input, index_col=0, parse_dates=True)

    values = {}
    for model_type in model_types:
        acc, window = choose_best_window_size(df, n, model_type)
        values[model_type] = {'accuracy': acc, 'window': window}
    with open(output, 'w') as file:
        content = json.dumps(values)
        file.write(content)


if __name__ == '__main__':
    best_window()  # noqa
