import click
import yaml
import pandas as pd

from src.latex.latex_generator import LatexTableGenerator


def parse_config(config):
    data = {}
    with open(config, 'r', encoding='utf-8') as stream:
        try:
            data = yaml.safe_load(stream)['tickers_desc']
        except yaml.YAMLError as e:
            print(e)
        except KeyError as e:
            print('No key:', e, 'in', config)
    return data


def generate_df_from_config_data(data):
    indices = []
    descriptions = []
    for row in data:
        indices.append(list(row.keys())[0])
        descriptions.append(row[indices[-1]])

    df = pd.DataFrame({'Код инструмента': indices, 'Показатель': descriptions})
    df = df.set_index('Код инструмента', drop=True)
    return df


def save_tickers_table_df_in_latex(df, output):
    table_generator = LatexTableGenerator()
    table_generator.index_cell_width = 6
    table_generator.columns_cell_width = 10
    table_generator.df_to_latex(df, 'Биржевые показатели, взятые для анализа')
    table_generator.save(output)


@click.command()
@click.option('--config')
@click.option('--output')
def generate_tickers_table(config, output):
    data = parse_config(config)
    df = generate_df_from_config_data(data)
    save_tickers_table_df_in_latex(df, output)


if __name__ == '__main__':
    generate_tickers_table()  # noqa
