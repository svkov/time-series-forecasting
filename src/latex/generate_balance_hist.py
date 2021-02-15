import click

from src.latex.latex_generator import LatexPictureGenerator
from src.utils.click_commands import InputCommand, LatexPictureCommand


@click.command(cls=LatexPictureCommand)
def generate_balance_hist(input, output, logs, name):
    print(name)
    pic = LatexPictureGenerator(path=input, name=name, label='graph-balance')
    pic.save(output)


if __name__ == '__main__':
    generate_balance_hist() # noqa