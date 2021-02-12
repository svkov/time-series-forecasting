import click


class InputCommand(click.core.Command):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input_option = click.core.Option(('--input',), help='Path to input file')
        output_option = click.core.Option(('--output',), help='Path to output file')
        self.params.insert(0, input_option)
        self.params.insert(0, output_option)

