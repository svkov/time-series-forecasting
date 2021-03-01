import click

from src.latex.latex_generator import LatexPictureGenerator
from src.utils import read_yaml
from src.utils.click_commands import InputCommand, LatexPictureCommand


@click.command(cls=LatexPictureCommand)
def generate_function_to_optimize(input, output, logs, name, labels):
    data = read_yaml(labels)
    pic = LatexPictureGenerator(path=input, name=data[name], label='graph-optimization')
    pic.save(output)


if __name__ == '__main__':
    generate_function_to_optimize() # noqa