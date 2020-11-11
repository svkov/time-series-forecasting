import dash_core_components as dcc
import dash_html_components as html

from forecasting import forecast_train_test_df, get_indexes_for_prediction, get_forecast_test
from data import load_data


class Layout:
    colors = {
        'background': '#111111',
        'text': '#7FDBFF'
    }

    def __init__(self, df):
        self.df = df

    def get_input(self, id, value, text, type='number'):
        return html.Div([
            html.P(text, className='text'),
            html.Div(dcc.Input(id=id, value=value, type=type, className='input'), className='input-div')
        ])

    def get_input_block(self):
        return html.Div([
            self.get_input(id='input-harm', value='10000', text='Type number of harmonics'),
            self.get_input(id='input-n-pred', value='300', text='How many hours to predict?'),
        ], className='input_div')

    @staticmethod
    def get_figure_to_future(df, forecasted, n_pred):
        return {
            'data': [
                {'x': df.index,
                 'y': df['close'],
                 'type': 'plot',
                 'name': 'Данные'},
                {'x': get_indexes_for_prediction(df, n_pred=n_pred),
                 'y': forecasted,
                 'type': 'plot',
                 'name': 'Прогноз'}
            ],
            'layout': {
                'title': 'BTC-USD Historical Price',
                'plot_bgcolor': Layout.colors['background'],
                'paper_bgcolor': Layout.colors['background'],
                'font': {
                    'color': Layout.colors['text']
                }
            }
        }

    def get_figure_train_test(self, **kwargs):
        train, test, forecasted = get_forecast_test(self.df, **kwargs)
        train_index = self.df.iloc[:len(train)].index
        test_index = self.df.iloc[len(train):].index
        return {
            'data': [{'x': train_index, 'y': train, 'type': 'plot', 'name': 'train'},
                     {'x': test_index, 'y': test, 'type': 'plot', 'name': 'test'},
                     {'x': test_index, 'y': forecasted, 'type': 'plot', 'name': 'forecast'}],
            'layout': {
                'title': 'Train-Test validation',
                'plot_bg_color': self.colors['background'],
                'paper_bgcolor': self.colors['background'],
                'font': {
                    'color': self.colors['text']
                }
            }
        }

    def get_graph(self, forecasted, n_pred, *args, **kwargs):
        return dcc.Graph(id='plot', figure=self.get_figure_to_future(self.df, forecasted, n_pred))

    def get_metrics(self, n_harm=10000, train_size=0.8):
        metrics = forecast_train_test_df(self.df, n_harm=n_harm, train_size=train_size)
        return html.Div([
            html.P(f"MAPE: {metrics['MAPE']}", className='text'),
            html.P(f"MAE: {metrics['MAE']}", className='text'),
            html.P(f"MSE: {metrics['MSE']}", className='text')]
        )

    def data(self):
        df = load_data()
        return html.Div(df.to_json(), id='data', style={'display': 'None'})

    def get_layout(self, forecasted, n_pred, train_size, n_harm):
        return html.Div([self.get_input_block(),
                         self.get_graph(forecasted, n_pred),
                         self.data(),
                         self.get_metrics(n_harm=n_harm, train_size=train_size)])
