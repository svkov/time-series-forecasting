import os

import pandas as pd
import yaml

from src.latex.latex_generator import LatexTableGenerator
from src.utils import send_to_telegram_if_fails, read_yaml
import click

from src.utils.click_commands import InputCommand, LatexPictureCommand


@send_to_telegram_if_fails
@click.command(cls=LatexPictureCommand)
def generate_table(input, output, logs, name, labels):
    data = read_yaml(labels)
    table_name = data[name]['name']
    table_label = data[name]['label']
    pic = LatexTableGenerator(path=input, caption=table_name, label=table_label)
    pic.save(output)


if __name__ == '__main__':
    generate_table()  # noqa
