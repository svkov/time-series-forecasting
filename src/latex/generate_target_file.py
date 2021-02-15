import os

import click

from src.latex.latex_generator import LatexPictureGenerator, LatexGenerator
from src.utils import send_to_telegram_if_fails
from src.utils.click_commands import InputCommand


@send_to_telegram_if_fails
@click.command(cls=InputCommand)
def generate(input, output):
    paths = input.split()

    name = 'Какое-то название'
    labels = [f'graph{i}' for i in range(len(paths))]
    pictures = LatexGenerator()
    generators = [LatexPictureGenerator(path=path,
                                        name=name,
                                        label=label) for path, label in zip(paths, labels)]
    pictures.concat_parts(*generators)
    pictures.save(output)


if __name__ == '__main__':
    generate()  # noqa
