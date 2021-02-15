import click

from src.utils.click_commands import InputCommand


@click.command(clas=InputCommand)
def generate_function_to_optimize(input, output):
    pass

if __name__ == '__main__':
    generate_function_to_optimize() # noqa