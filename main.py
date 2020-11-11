import dash
import dash_core_components as dcc
import dash_html_components as html

from app import app
from apps import crypto_app


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')]
              )
def display_page(pathname):
    print(pathname)
    if pathname == '/btc':
        return crypto_app.layout()
    else:
        return html.P('404', style={'color': 'white'})


if __name__ == '__main__':
    app.run_server(debug=True)
