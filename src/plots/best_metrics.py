import pandas as pd
import click
import os
import telegram_send


def parse_row(name, row):
    argmin = row.argmin()
    return {
        'metric': name,
        'model': row.index[argmin],
        'value': row[argmin]
    }


def send_df_to_telegram(df):
    message = f'```{str(df)}```'
    telegram_send.send(messages=[message], parse_mode='markdown')


@click.command()
@click.option('--input')
@click.option('--output')
def best_metrics(input, output):
    input = input.split()
    full_df = pd.DataFrame()
    for i in input:
        ticker = os.path.split(i)[-1].replace('metrics_', '').replace('.csv', '')
        df = pd.read_csv(i, index_col=0)
        for name, row in df.iterrows():
            parsed_row = parse_row(name, row)
            parsed_df = pd.DataFrame(parsed_row, index=[ticker])
            full_df = full_df.append(parsed_df)

    send_df_to_telegram(full_df)
    full_df.to_csv(output)


if __name__ == '__main__':
    best_metrics()  # noqa
