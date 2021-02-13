import click
import pandas as pd
import plotly.graph_objects as go

from src.utils import send_to_telegram_if_fails
from src.utils.click_commands import InputCommand


def get_fig(df, name) -> go.Figure:
    return go.Figure(go.Scatter(x=df.Date, y=df.Close), layout={'title': name})


def save_fig(fig: go.Figure, path):
    fig.write_image(path)


@send_to_telegram_if_fails
@click.command(cls=InputCommand)
@click.option('--name')
def plot_raw(input, output, logs, name):
    df = pd.read_csv(input, parse_dates=True)
    fig = get_fig(df, name)
    save_fig(fig, output)


if __name__ == '__main__':
    plot_raw() # noqa