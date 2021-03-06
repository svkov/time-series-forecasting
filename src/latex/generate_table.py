import click

from src.latex import Config
from src.latex.latex_generator import LatexTableGenerator
from src.utils import send_to_telegram_if_fails
from src.utils.click_commands import LatexPictureCommand


@send_to_telegram_if_fails
@click.command(cls=LatexPictureCommand)
def generate_table(input, output, logs, name, labels):
    config = Config(labels, name)

    pic = LatexTableGenerator(path=input,
                              caption=config.get('name', name),
                              label=config.get('label', name),
                              index_col=config.get('index_col', 0),
                              index_cell_width=config.get('index_width', 1.5),
                              columns_cell_width=config.get('columns_width', 1.5))
    pic.save(output)


if __name__ == '__main__':
    generate_table()  # noqa
