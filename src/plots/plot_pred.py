import os

import click
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error
from typing import Callable

from src.plots import get_results
from src.utils import send_to_telegram_if_fails
from src.utils.click_commands import InputCommand


def get_fig(df: pd.DataFrame, name: str, metric: Callable, n_pred) -> go.Figure:
    plots = []
    for col in df.columns:
        if col == 'test':
            scatter_name = col
        else:
            scatter_name = f'{col} {metric(df.test, df[col]):.2f}'
        plots.append(go.Scatter(x=df.index, y=df[col], name=scatter_name))
    return go.Figure(data=plots, layout={'title': f'{name}, {n_pred}'})


def save_fig(fig: go.Figure, path):
    fig.write_image(path)


@send_to_telegram_if_fails
@click.command(cls=InputCommand)
@click.option('--name')
@click.option('--models')
@click.option('--n_pred')
def plot_pred(input, output, logs, name, models, n_pred):
    """

    :param input: Список путей до предсказания одной модели (без стекинга)
    :param output: Путь куда должен сохраниться график предсказания по этому тикеру
    :param name:
    :param models:
    :param n_pred:
    :return:
    """
    n_pred = int(n_pred)
    df = pd.read_csv(input, index_col=0, parse_dates=True)
    results = get_results(df, models, n_pred)
    fig = get_fig(results, name, metric=mean_absolute_error, n_pred=n_pred)
    save_fig(fig, output)


if __name__ == '__main__':
    plot_pred()  # noqa
