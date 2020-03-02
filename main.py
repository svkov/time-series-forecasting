import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

main_style = {
    'backgroundColor': colors['background'],
    'top': 0,
    'left': 0,
    'margin': 0,
    'position': 'absolute',
    'width': '100%',
    'height': '100%'
}

app = dash.Dash(__name__)
df = pd.read_csv('btcusd_full_hour_2020_02_01.csv', index_col='date')
df = df[80000:]


def get_figure(df, title):
    return {
        'data': [{'x': df.index, 'y': df['avg'], 'type': 'plot', 'name': 'some shit'}],
        'layout': {
            'title': title,
            'plot_bgcolor': colors['background'],
            'paper_bgcolor': colors['background'],
            'font': {
                'color': colors['text']
            }
        }
    }


def get_graph(df, title='BTC-USD Historical Price'):
    return dcc.Graph(id='plot', style=colors, figure=get_figure(df, title))


app.layout = html.Div(children=[
    html.H1('BTC Price Analysis',
            style={'color': colors['text'],
                   'textAlign': 'center'}),
    get_graph(df),
    html.P('Type number of harmonics', style={'color': colors['text']}),
    dcc.Input(id='input-harm', value='10000', type='number', )],
    style=main_style
)


@app.callback(
    Output(component_id='plot', component_property='figure'),
    [Input(component_id='input-harm', component_property='value')]
)
def update_harm_number(new_harm):
    return get_figure(df, title=new_harm)


if __name__ == '__main__':
    app.run_server(debug=True)
