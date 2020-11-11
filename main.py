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


app = dash.Dash(__name__)
app.layout = get_layout


def process_int_arg(arg, default_val=1):
    if arg:
        return int(arg)
    return default_val


def process_json_to_df_arg(arg):
    return pd.read_json(arg)


@app.callback(
    Output('plot', 'figure'),
    [Input('input-harm', 'value'),
     Input('input-n-pred', 'value')],
    [State('data', 'children')]
)
def update_harm_number(n_harm, n_pred, json_str):
    data = process_json_to_df_arg(json_str)
    n_harm = process_int_arg(n_harm)
    n_pred = process_int_arg(n_pred)
    forecasted = forecast(data, int(n_harm), int(n_pred))
    return Layout.get_figure_to_future(data, forecasted, int(n_pred))


if __name__ == '__main__':
    app.run_server(debug=True)
