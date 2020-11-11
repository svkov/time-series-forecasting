import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from forecasting import forecast_train_test_df, get_indexes_for_prediction, get_forecast_test

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


def get_figure_to_future(df, forecasted, n_pred):
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


def get_figure_train_test(df, **kwargs):
    train, test, forecasted = get_forecast_test(df, **kwargs)
    train_index = df.iloc[:len(train)].index
    test_index = df.iloc[len(train):].index
    return {
        'data': [{'x': train_index, 'y': train, 'type': 'plot', 'name': 'train'},
                 {'x': test_index, 'y': test, 'type': 'plot', 'name': 'test'},
                 {'x': test_index, 'y': forecasted, 'type': 'plot', 'name': 'forecast'}],
        'layout': {
            'title': 'Train-Test validation',
            'plot_bg_color': colors['background'],
            'paper_bgcolor': colors['background'],
            'font': {
                'color': colors['text']
            }
        }
    }



def get_graph(df, forecasted, *args, **kwargs):
    return dcc.Graph(id='plot', figure=get_figure_to_future(df, forecasted))
    # return dcc.Graph(id='plot', figure=get_figure_train_test(df, train_size=0.8))


def get_metrics(df, n_harm=10000, train_size=0.8):
    metrics = forecast_train_test_df(df, n_harm=n_harm, train_size=train_size)
    return html.Div([
        html.P(f"MAPE: {metrics['MAPE']}", className='text'),
        html.P(f"MAE: {metrics['MAE']}", className='text'),
        html.P(f"MSE: {metrics['MSE']}", className='text')]
    )
