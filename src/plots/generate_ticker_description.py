import click
import yaml
import pandas as pd

from src.utils.click_commands import InputCommand


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


@click.command(cls=InputCommand)
def generate_ticker_description(input, output, **kwargs):
    data = parse_config(input)
    df = generate_df_from_config_data(data)
    df.to_csv(output)


if __name__ == '__main__':
    generate_ticker_description()  # noqa
