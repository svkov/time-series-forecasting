import dash
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
from dash.dependencies import Input, Output, State
from dash_src.layout import Layout
from data import load_data
from forecasting import forecaft_df_to_future


def forecast(df, n_pred, harm=10000):
    return forecaft_df_to_future(df, n_pred=n_pred, n_harm=harm, start_from=0)


def get_default_params():
    return {
        'n_pred': 300,
        'n_harm': 1000,
        'train_size': 0.99,
        'freq_type': 'hour'
    }


def get_layout():
    params = get_default_params()
    df = load_data(params['freq_type'])
    forecasted = forecast(df, params['n_pred'], params['n_harm'])
    layout = Layout(df)

    final = html.Div(children=[
        dcc.Location(id='url', refresh=False),
        html.H1('BTC Price Analysis', className='text', id='title'),
        layout.data(),
        layout.get_layout(forecasted, params['n_pred'], params['train_size'], params['n_harm'])
    ], id='page-content')
    return final


app = dash.Dash(__name__)
app.layout = get_layout


def process_int_arg(arg, default_val=1):
    if arg:
        return int(arg)
    return default_val


def process_json_to_df_arg(arg):
    return pd.read_json(arg)


@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')],
              [State('data', 'children')])
def display_page(pathname, json_df):
    print(pathname)
    params = get_default_params()
    df = process_json_to_df_arg(json_df)
    layout = Layout(df)
    forecasted = forecast(df, params['n_pred'], params['n_harm'])
    return layout.get_layout(forecasted, params['n_pred'], params['train_size'], params['n_harm'])


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
    forecasted = forecast(data, n_harm, n_pred)
    return Layout.get_figure_to_future(data, forecasted, n_pred)


if __name__ == '__main__':
    app.run_server(debug=True)
