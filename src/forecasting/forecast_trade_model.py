import click
import json
import pandas as pd

from src.trade.model import get_cv_train_test, fit_predict
from src.trade.prepare_data import generate_window


@click.command()
@click.option('--input')
@click.option('--output')
@click.option('--window')
@click.option('--n')
@click.option('--model_types')
def forecast_trade(input, output, window, n, model_types):
    n = int(n)
    model_types = model_types.split()
    with open(window) as file:
        best_widows = json.loads(file.read())

    df = pd.read_csv(input, index_col=0, parse_dates=True)

    result = []
    for model in model_types:
        window_model = int(best_widows[model]['window'])
        window_df = generate_window(df, window_model)
        print(window_model, model)
        for train, test in get_cv_train_test(window_df, train_size=0.9):
            y_pred, y_test = fit_predict(train, test, window_model, model)
            date = train.index[-1]
            result.append({'pred': y_pred.tolist(), 'test': y_test.tolist(), 'date': date.strftime('%Y-%m-%d')})
    print(result)
    with open(output, 'w') as file:
        file.write(json.dumps(result))


if __name__ == '__main__':
    forecast_trade() # noqa