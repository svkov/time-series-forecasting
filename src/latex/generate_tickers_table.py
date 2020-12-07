import click
import yaml
from src.latex import generate_columns, generate_row, generate_table_header, save_latex


@click.command()
@click.option('--config')
@click.option('--output')
def generate_tickers_table(config, output):
    data = {}
    with open(config, 'r', encoding='utf-8') as stream:
        try:
            data = yaml.safe_load(stream)['tickers_desc']
        except yaml.YAMLError as e:
            print(e)
        except KeyError as e:
            print('No key:', e, 'in', config)

    table = generate_columns(['Код инструмента'], ['Показатель'])
    for row in data:
        ticker = list(row.keys())[0]
        description = row[ticker]
        table += generate_row(ticker, [description])
    table += '\\hline\n'
    table = generate_table_header(table, 'Биржевые показатели, взятые для анализа', [6, 10], label='tickers')
    save_latex(table, output)


if __name__ == '__main__':
    generate_tickers_table()  # noqa
