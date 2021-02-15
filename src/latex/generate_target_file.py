import os

import click

from src.latex.latex_generator import LatexPictureGenerator, LatexGenerator
from src.utils import send_to_telegram_if_fails
from src.utils.click_commands import InputCommand


@send_to_telegram_if_fails
@click.command(cls=InputCommand)
def generate(input, output):
    path_to_figures = input.replace('\\', '/').split()

    name = 'Какое-то название'
    paths = [f'../{path}' for path in path_to_figures]
    labels = [f'graph{i}' for i in range(len(path_to_figures))]
    pictures = LatexGenerator()
    generators = [LatexPictureGenerator(path=path,
                                        name=name,
                                        label=label) for path, label in zip(paths, labels)]
    pictures.concat_parts(generators)
    pictures.save(output)


if __name__ == '__main__':
    generate()  # noqa
