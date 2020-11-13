import os

import click
import pandas as pd
import plotly.graph_objects as go


def get_fig(df: pd.DataFrame, name) -> go.Figure:
    plots = []
    for col in df.columns:
        plots.append(go.Scatter(x=df.index, y=df[col], name=col))
    return go.Figure(data=plots, layout={'title': name})


def save_fig(fig: go.Figure, path):
    fig.write_image(path)


@click.command()
@click.option('--input')
@click.option('--output')
@click.option('--name')
@click.option('--models')
@click.option('--path_to_pred')
def plot_raw(input, output, name, models, path_to_pred):
    model_names = models.split()
    model_results = [path_to_pred + model for model in model_names]

    results = pd.DataFrame(columns=model_names)
    for result, model in zip(model_results, model_names):
        path = os.path.join(result, f'{name}.csv')
        df = pd.read_csv(path, parse_dates=True, index_col='Date')
        results[model] = df.Prediction
    fig = get_fig(results, name)
    save_fig(fig, output)


if __name__ == '__main__':
    plot_raw() # noqa