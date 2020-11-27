import os

import click
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error
from typing import Callable

from src.plots import get_results
from src.utils import send_to_telegram_if_fails


def get_fig(df: pd.DataFrame, name: str, metric: Callable, n_pred: str) -> go.Figure:
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
@click.command()
@click.option('--input')
@click.option('--output')
@click.option('--name')
@click.option('--models')
@click.option('--n_pred')
def plot_pred(input, output, name, models, n_pred):
    n_pred = int(n_pred)
    # results = get_results(models, path_to_pred, name)
    df = pd.read_csv(input, index_col=0, parse_dates=True)
    models = models.split()
    results = pd.DataFrame()
    for model in models:
        # for i in range(n_pred):
        results[model] = df[f'{model} Close n{n_pred}'].iloc[n_pred:]
    fig = get_fig(results, name, metric=mean_absolute_error, n_pred=n_pred)
    save_fig(fig, output)


if __name__ == '__main__':
    plot_pred()  # noqa
