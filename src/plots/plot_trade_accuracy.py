import json

import click
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score

from src.utils.click_commands import InputCommand
from src.utils.file import save_plotly_fig


def open_json(path):
    with open(path) as file:
        res = json.loads(file.read())
    return res


def get_result_df(res, n):
    models = list(res.keys())
    res_df = pd.DataFrame()
    res_df['date'] = [i['date'] for i in res[models[0]]]
    res_df['target'] = [i['target'][0] for i in res[models[0]]]
    res_df.set_index('date', inplace=True)
    for model in models:
        res_df[model] = [accuracy_score(i['pred'][:n], i['test'][:n]) for i in res[model]]
    res_df = res_df.sort_index()[:'2021-01-01']
    res_df['random_prediction'] = 1 / 3
    return res_df


def plot_result_df(res_df):
    fig = go.Figure()
    df = res_df.drop('target', axis=1)
    for model in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[model], name=model))
    fig.update_layout(title='Compare accuracy to random prediction',
                      xaxis_title='Date',
                      yaxis_title='Accuracy')
    return fig


@click.command(cls=InputCommand)
@click.option('--n')
def plot_trade_accuracy(input, output, logs, n):
    n = int(n)
    res = open_json(input)
    res_df = get_result_df(res, n)
    fig = plot_result_df(res_df)
    save_plotly_fig(fig, output)


if __name__ == '__main__':
    plot_trade_accuracy()  # noqa
