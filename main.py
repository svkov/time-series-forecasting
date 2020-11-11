import dash
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State

from dash_src.layout import Layout
from data import load_data
from forecasting import forecaft_df_to_future


def forecast(df, n_pred, harm=10000):
    forecasted = forecaft_df_to_future(df, n_pred=n_pred, n_harm=harm, start_from=0)
    return forecasted


def get_layout():
    n_pred = 300
    n_harm = 100000
    train_size = 0.99
    df = load_data()
    forecasted = forecast(df, n_pred, n_harm)
    layout = Layout(df)

    final = html.Div(children=[
        html.H1('BTC Price Analysis', className='text', id='title'),
        layout.get_layout(forecasted, n_pred, train_size, n_harm)
    ], id='main-container')
    return final


# TODO: Remove global variables


app = dash.Dash(__name__)
app.layout = get_layout


@app.callback(
    Output('plot', 'figure'),
    [Input('input-harm', 'value'),
     Input('input-n-pred', 'value')],
    [State('data', 'children'),]
     # State('input-harm', 'value'),
     # State('input-n-pred', 'value')]
)
def update_harm_number(n_harm, n_pred, json_str):
    data = pd.read_json(json_str)
    if n_harm:
        n_harm = int(n_harm)
    else:
        n_harm = 1
    if n_pred:
        n_pred = int(n_pred)
    else:
        n_pred = 1
    forecasted = forecast(data, int(n_harm), int(n_pred))
    return Layout.get_figure_to_future(data, forecasted, int(n_pred))


if __name__ == '__main__':
    app.run_server(debug=True)
