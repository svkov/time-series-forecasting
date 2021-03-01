import click

from src.latex.latex_generator import LatexPictureGenerator, LatexGenerator
from src.utils import send_to_telegram_if_fails, read_yaml
from src.utils.click_commands import LatexPictureCommand
import yaml


@send_to_telegram_if_fails
@click.command(cls=LatexPictureCommand)
def generate_target_file(input, output, logs, name, labels):
    paths = input.split()

    data = read_yaml(labels)

    # name = 'Какое-то название'
    labels = [f'graph{i}' for i in range(len(paths))]
    pictures = LatexGenerator()
    generators = [LatexPictureGenerator(path=path,
                                        name=data[name],
                                        label=label) for path, label in zip(paths, labels)]
    pictures.concat_parts(*generators)
    pictures.save(output)


if __name__ == '__main__':
    generate_target_file()  # noqa
