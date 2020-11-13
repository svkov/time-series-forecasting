import click
import pandas as pd
import plotly.graph_objects as go


def get_fig(df, name) -> go.Figure:
    return go.Figure(go.Scatter(x=df.Date, y=df.Close), layout={'title': name})


def save_fig(fig: go.Figure, path):
    fig.write_image(path)


@click.command()
@click.option('--input')
@click.option('--output')
@click.option('--name')
def plot_raw(input, output, name):
    df = pd.read_csv(input, parse_dates=True)
    fig = get_fig(df, name)
    save_fig(fig, output)


if __name__ == '__main__':
    plot_raw() # noqa