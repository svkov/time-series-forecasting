import click

from src.latex.latex_generator import LatexPictureGenerator
from src.utils.click_commands import InputCommand


@click.command(cls=InputCommand)
def generate_balance_hist(input, output):
    pic = LatexPictureGenerator(path=input, name='Диаграмма сбалансированности классов', label='graph-balance')
    pic.save(output)
    

if __name__ == '__main__':
    generate_balance_hist() # noqa