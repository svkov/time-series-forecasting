import click

from src.latex import Config
from src.latex.latex_generator import LatexPictureGenerator
from src.utils.click_commands import LatexPictureCommand


@click.command(cls=LatexPictureCommand)
def generate_picture(input, output, logs, name, labels):
    config = Config(labels, name)
    picture_name = config.get('name', name)
    picture_label = config.get('label', name)

    pic = LatexPictureGenerator(path=input, name=picture_name, label=picture_label)
    pic.save(output)


if __name__ == '__main__':
    generate_picture() # noqa