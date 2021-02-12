import click
import numpy as np
import pandas as pd

from src.trade.prepare_data import read_data, get_imbalance_by_thresh
from src.utils import save_plotly_fig
from src.utils.click_commands import InputCommand
import plotly.express as px


@click.command(cls=InputCommand)
@click.option('--n')
@click.option('--instrument')
@click.option('--freq')
def plot_trade_function_to_optimize(input, output, n, instrument, freq):
    n = int(n)
    freq = int(freq)

    df = read_data(input)

    x = np.linspace(0.1, 100, freq)
    y = [get_imbalance_by_thresh(i, df, instrument, n) for i in x]

    df = pd.DataFrame({'Threshold': x, 'Balance measure': y})
    fig = px.line(df, x='Threshold', y='Balance measure', title=f'Мера сбалансированности классов в {instrument}')
    save_plotly_fig(fig, output)


if __name__ == '__main__':
    plot_trade_function_to_optimize()  # noqa
