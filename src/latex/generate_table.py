import os

import pandas as pd

from src.latex.latex_generator import LatexTableGenerator
from src.utils import send_to_telegram_if_fails
import click

from src.utils.click_commands import InputCommand, LatexPictureCommand


@send_to_telegram_if_fails
@click.command(cls=LatexPictureCommand)
def generate_table(input, output, logs, name):
    path_to_tables = input.split()
    full_df = pd.DataFrame()
    for path in path_to_tables:
        ticker = os.path.split(path)[-1].replace('metrics_', '').replace('.csv', '')
        df = pd.read_csv(path, index_col=0)
        df['ticker'] = ticker
        full_df = full_df.append(df)
    full_df['metric'] = full_df.index
    full_df = full_df.set_index(['ticker', 'metric'])

    table = LatexTableGenerator()
    # name = "результаты"
    table.df_to_latex(full_df, name)
    table.save(output)


if __name__ == '__main__':
    generate_table()  # noqa
