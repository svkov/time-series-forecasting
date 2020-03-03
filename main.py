import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output

from forecasting import forecaft_df, get_indexes_for_prediction

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


def get_input(id, value, type='number'):
    return html.Div(dcc.Input(id=id, value=value, type=type, className='input'), className='input-div')


def get_input_block():
    return html.Div([
        html.P('Type number of harmonics', className='text'),
        get_input(id='input-harm', value='10000'),
        html.P('How many hours to predict?', className='text'),
        get_input(id='input-n-pred', value='300'),
    ], className='input_div')


def load_data(start_from=80000):
    df = pd.read_csv('btcusd_full_hour_2020_02_01.csv', index_col='date')
    df = df[start_from:]
    return df


def forecast(n_pred, harm=10000):
    global df
    forecasted = forecaft_df(df, n_pred=n_pred, n_harm=harm, start_from=0)
    return forecasted


def get_figure(df, forecasted):
    return {
        'data': [{'x': df.index, 'y': df['avg'], 'type': 'plot'},
                 {'x': get_indexes_for_prediction(df, n_pred=n_pred), 'y': forecasted, 'type': 'plot'}],
        'layout': {
            'title': 'BTC-USD Historical Price',
            'plot_bgcolor': colors['background'],
            'paper_bgcolor': colors['background'],
            'font': {
                'color': colors['text']
            }
        }
    }


def get_graph(df, forecasted):
    return dcc.Graph(id='plot', figure=get_figure(df, forecasted))


# TODO: Remove global variables
n_pred = 300
harm = 10000
df = load_data()
app = dash.Dash(__name__)
forecasted = forecast(n_pred=n_pred, harm=harm)

app.layout = html.Div(children=[
    html.H1('BTC Price Analysis', className='text'),
    get_input_block(),
    get_graph(df, forecasted)], id='main-container')


@app.callback(
    Output(component_id='plot', component_property='figure'),
    [Input(component_id='input-harm', component_property='value'),
     Input(component_id='input-n-pred', component_property='value')]
)
def update_harm_number(new_harm=None, new_n_pred=None):
    global harm, n_pred
    if new_harm:
        new_harm = int(new_harm)
        harm = new_harm
    if new_n_pred:
        new_n_pred = int(new_n_pred)
        n_pred = new_n_pred
    forecasted = forecast(n_pred=harm, harm=n_pred)
    return get_figure(df, forecasted)


if __name__ == '__main__':
    app.run_server(debug=True)
