import os

import click
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error
from typing import Callable


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


@click.command()
@click.option('--input')
@click.option('--output')
@click.option('--name')
@click.option('--models')
@click.option('--path_to_pred')
@click.option('--n_pred')
def plot_raw(input, output, name, models, path_to_pred, n_pred):
    model_names = models.split()
    model_results = [path_to_pred + model for model in model_names]

    results = pd.DataFrame(columns=model_names)
    for result, model in zip(model_results, model_names):
        path = os.path.join(result, f'{name}.csv')
        df = pd.read_csv(path, parse_dates=True, index_col=0)
        series = pd.Series(df.values[:, -1], index=df.index)
        results[model] = series
    fig = get_fig(results, name, metric=mean_absolute_error, n_pred=n_pred)
    save_fig(fig, output)


if __name__ == '__main__':
    plot_raw()  # noqa
