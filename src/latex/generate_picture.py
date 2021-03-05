import click

from src.latex.latex_generator import LatexPictureGenerator
from src.utils import read_yaml
from src.utils.click_commands import InputCommand, LatexPictureCommand


@click.command(cls=LatexPictureCommand)
def generate_picture(input, output, logs, name, labels):
    data = read_yaml(labels)
    print(data)
    print(name)
    picture_name = data[name]['name']
    picture_label = data[name]['label']
    pic = LatexPictureGenerator(path=input, name=picture_name, label=picture_label)
    pic.save(output)


if __name__ == '__main__':
    generate_picture() # noqa