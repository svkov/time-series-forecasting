import os

import click

from src.utils import send_to_telegram_if_fails
from src.utils.click_commands import InputCommand




@send_to_telegram_if_fails
@click.command(cls=InputCommand)
def generate(input, output):
    path_to_figures = input.replace('\\', '/').split()
    name = 'Какое-то название'
    figs = [generate_figure(f'../{path}', name, f'graph{i}') for i, path in enumerate(path_to_figures)]
    body = concat_parts(*figs)
    # tex = generate_header(body)
    with open(output, 'w', encoding='utf-8') as f:
        f.write(body)


if __name__ == '__main__':
    generate()  # noqa
