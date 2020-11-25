import click
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.plots import get_results
from src.utils import mean_absolute_percentage_error, send_to_telegram_if_fails

metrics_map = {
    'MAE': mean_absolute_error,
    'MAPE': mean_absolute_percentage_error,
    'MSE': mean_squared_error
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
@click.command()
@click.option('--input')
@click.option('--output')
@click.option('--name')
@click.option('--models')
@click.option('--path_to_pred')
@click.option('--n_pred')
@click.option('--metrics')
def get_metrics(input, output, name, models, path_to_pred, n_pred, metrics):
    results = get_results(models, path_to_pred, name)

    metrics = metrics.split()
    get_metrics_df(results, metrics).to_csv(output)


if __name__ == '__main__':
    get_metrics() # noqa