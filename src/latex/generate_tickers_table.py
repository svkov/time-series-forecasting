import click
import yaml
import pandas as pd

from src.latex.latex_generator import LatexTableGenerator
from src.utils import read_yaml


def save_tickers_table_df_in_latex(df, output, name, label):
    table_generator = LatexTableGenerator()
    table_generator.index_cell_width = 6
    table_generator.columns_cell_width = 10
    table_generator.df_to_latex(df, name, label)
    table_generator.save(output)


@click.command()
@click.option('--input')
@click.option('--output')
@click.option('--name')
@click.option('--labels')
def generate_tickers_table(input, output, name, labels):
    df = pd.read_csv(input, index_col=0)
    data = read_yaml(labels)
    table_name = data[name]['name']
    table_label = data[name]['label']
    save_tickers_table_df_in_latex(df, output, table_name, table_label)


if __name__ == '__main__':
    generate_tickers_table()  # noqa
