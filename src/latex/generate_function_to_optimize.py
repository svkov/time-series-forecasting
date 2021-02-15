import click

from src.latex.latex_generator import LatexPictureGenerator
from src.utils.click_commands import InputCommand


@click.command(cls=InputCommand)
def generate_function_to_optimize(input, output):
    pic = LatexPictureGenerator(path=input, name='Функция для оптимизации', label='graph-optimization')
    pic.save(output)


if __name__ == '__main__':
    generate_function_to_optimize() # noqa