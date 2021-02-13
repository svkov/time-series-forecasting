import os

import pandas as pd
from src.utils import send_to_telegram_if_fails
from src.latex import generate_columns, generate_row, generate_table_header
import click

from src.utils.click_commands import InputCommand

"""
\begin{center}
    \begin{longtable}{|p{2cm}|p{3cm}|p{7cm}|p{3cm}|}
    \caption{Заголовок таблицы}\\
    \hline
    1 & 2 & 3 & 4\\ 
    \hline 
    2 & 2 & 3 & 4\\
    \hline
    3 & 2 & 3 & 4\\
    \hline
    4 & 2 & 3 & 4\\
    \hline
    5 & 2 & 3 & 4\\
    \hline
    6 & 2 & 3 & 4\\
    \hline
    7 & 2 & 3 & 4\\
    \hline
    8 & 2 & 3 & 4\\
    \hline
    9 & 2 & 3 & 4\\
    \hline
    10 & 2 & 3 & 4\\
    \hline
    
    
    \end{longtable}
\end{center}
"""


def df_to_latex(df: pd.DataFrame, caption):
    body = ''
    columns_width = [1.5 for i in range(df.index.nlevels)] + [1.5 for i in range(len(df.columns))]

    body += generate_columns(df.index.names, df.columns)
    for name, row in df.iterrows():
        row = generate_row(name, row)
        body += row
    body += '\\hline\n'
    return generate_table_header(body, caption, columns_width)


@send_to_telegram_if_fails
@click.command(cls=InputCommand)
def generate_table(input, output):
    path_to_tables = input.split()
    full_df = pd.DataFrame()
    for path in path_to_tables:
        ticker = os.path.split(path)[-1].replace('metrics_', '').replace('.csv', '')
        df = pd.read_csv(path, index_col=0)
        df['ticker'] = ticker
        full_df = full_df.append(df)
    full_df['metric'] = full_df.index
    full_df = full_df.set_index(['ticker', 'metric'])
    table = df_to_latex(full_df, 'Результаты')

    with open(output, 'w', encoding='utf-8') as f:
        f.write(table)


if __name__ == '__main__':
    generate_table()  # noqa
