import os
import pandas as pd
import click

from src.utils.click_commands import InputCommand


@click.command(cls=InputCommand)
def aggregate_metrics(input, output, **kwargs):
    path_to_tables = input.split()
    full_df = pd.DataFrame()
    for path in path_to_tables:
        ticker = os.path.split(path)[-1].replace('metrics_', '').replace('.csv', '')
        df = pd.read_csv(path)
        df['ticker'] = ticker
        df = df.set_index(['ticker', 'metric'])
        df = df.melt(var_name='model', ignore_index=False)
        full_df = full_df.append(df)

    # full_df['metric'] = full_df.index
    # full_df = full_df.set_index(['ticker', 'metric'])

    full_df.to_csv(output)


if __name__ == '__main__':
    aggregate_metrics()  # noqa
