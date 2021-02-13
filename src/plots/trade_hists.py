import click
import plotly.graph_objects as go

from src.trade.prepare_data import read_data, make_labels, prepare_data_without_window
from src.utils import save_plotly_fig
from src.utils.click_commands import InputCommand


@click.command(cls=InputCommand)
@click.option('--n')
@click.option('--instrument')
@click.option('--thresh')
def trade_hists(input, output, n, instrument, thresh):
    n = int(n)
    thresh = float(thresh)
    df1 = read_data(input)
    df1 = make_labels(df1, f'{instrument} Close', n=n, thresh=thresh).dropna()
    df2, optimal_thresh = prepare_data_without_window(input, instrument=instrument, n=n, bounds=[0.1, 20], return_thresh=True)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df1.label,
        histnorm='percent',
        name=f'До балансировки, t={thresh:.1f}',
        xbins=dict(
            start=-4.0,
            end=3.0,
            size=0.5
        ),
        opacity=0.75
    ))
    fig.add_trace(go.Histogram(
        x=df2.label,
        histnorm='percent',
        name=f'После балансировки, t={optimal_thresh:.1f}',
        xbins=dict(
            start=-3.0,
            end=4,
            size=0.5
        ),
        opacity=0.75
    ))

    fig.update_layout(
        title_text=f'Распределение классов до и после балансировки',  # title of plot
    )
    save_plotly_fig(fig, output)


if __name__ == '__main__':
    trade_hists()  # noqa
