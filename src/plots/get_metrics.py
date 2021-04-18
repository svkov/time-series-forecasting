import click
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.plots import get_results
from src.utils import mean_absolute_percentage_error, send_to_telegram_if_fails
from src.utils.click_commands import InputCommand

metrics_map = {
    'MAE': mean_absolute_error,
    'MAPE': mean_absolute_percentage_error,
    'MSE': mean_squared_error,
    'RMSE': lambda x, y: np.sqrt(mean_squared_error(x, y))
}


def get_metrics_df(results, metrics_list):
    metrics_dict = {}
    index = []
    for metric in metrics_list:
        index.append(metric)
        for model in results.drop('test', axis=1).columns:
            if model not in metrics_dict:
                metrics_dict[model] = []
            metric_value = metrics_map[metric](results[model], results['test']).__round__(2)
            metrics_dict[model].append(metric_value)

    metrics_df = pd.DataFrame(metrics_dict, index=index)
    metrics_df.index.name = 'Metric'
    return metrics_df


@send_to_telegram_if_fails
@click.command(cls=InputCommand)
@click.option('--name')
@click.option('--models')
@click.option('--n_pred')
@click.option('--metrics')
def get_metrics(input, output, logs, name, models, n_pred, metrics):
    n_pred = int(n_pred)
    df = pd.read_csv(input, index_col=0, parse_dates=True)
    results = get_results(df, models, n_pred)

    metrics = metrics.split()
    get_metrics_df(results, metrics).to_csv(output)


if __name__ == '__main__':
    get_metrics() # noqa