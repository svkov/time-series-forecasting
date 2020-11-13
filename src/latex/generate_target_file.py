import click

@click.command()
@click.option('--input')
@click.option('--output')
def generate(input, output):
    path_to_figures = input.split()



if __name__ == '__main__':
    generate() # noqa