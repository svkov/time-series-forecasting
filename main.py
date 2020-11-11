import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

from dash_src.layout import get_input_block, get_graph, get_metrics, get_figure_to_future
from forecasting import forecaft_df_to_future, get_indexes_for_prediction, forecast_train_test_df, get_forecast_test, \
    calculate_metrics


def load_data(start_from=80000):
    df = pd.read_csv('btcusd_full_hour_2020_02_01.csv', index_col='date')
    df = df[start_from:]
    return df


def forecast(n_pred, harm=10000):
    global df
    forecasted = forecaft_df_to_future(df, n_pred=n_pred, n_harm=harm, start_from=0)
    return forecasted


# TODO: Remove global variables
n_pred = 300
harm = 100000
train_size = 0.99
df = load_data(start_from=80000)
app = dash.Dash(__name__)
forecasted = forecast(n_pred=n_pred, harm=harm)

app.layout = html.Div(children=[
    html.H1('BTC Price Analysis', className='text', id='title'),
    get_input_block(),
    get_graph(df, forecasted),
    get_metrics(df, n_harm=harm, train_size=train_size)
], id='main-container')


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
    return get_figure_to_future(df, forecasted, n_pred)
    # train-test graph
    # return get_figure_train_test(df, train_size=train_size)


if __name__ == '__main__':
    app.run_server(debug=True)
