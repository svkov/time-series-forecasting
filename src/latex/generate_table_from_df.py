import click

from src.latex.latex_generator import LatexTableGenerator
from src.utils import send_to_telegram_if_fails, read_yaml
from src.utils.click_commands import LatexPictureCommand


@send_to_telegram_if_fails
@click.command(cls=LatexPictureCommand)
def generate_table(input, output, logs, name, labels):
    data = read_yaml(labels)
    try:
        table_name = data[name]['name']
    except KeyError:
        table_name = name

    try:
        table_label = data[name]['label']
    except KeyError:
        table_label = name

    pic = LatexTableGenerator(path=input, caption=table_name, label=table_label, index_col=0)
    pic.save(output)


if __name__ == '__main__':
    generate_table()  # noqa
