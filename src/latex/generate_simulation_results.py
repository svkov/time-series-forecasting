import click

from src.latex.latex_generator import LatexTableGenerator
from src.utils import read_yaml
from src.utils.click_commands import LatexPictureCommand


@click.command(cls=LatexPictureCommand)
def generate_simulation_results(input, output, logs, name, labels):
    data = read_yaml(labels)
    table = LatexTableGenerator(path=input, caption=data[name], label='simulation_results', index_col=0)
    table.save(output)


if __name__ == '__main__':
    generate_simulation_results()  # noqa
