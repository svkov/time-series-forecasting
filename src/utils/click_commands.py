import click


class InputCommand(click.core.Command):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_option('--input', 'Path to input file')
        self.add_option('--output', 'Path to output file')
        self.add_option('--logs', 'Path to log file')

    def add_option(self, command, help):
        new_option = click.core.Option((command,), help=help)
        self.params.insert(0, new_option)


class LatexPictureCommand(InputCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_option('--name', 'Text under the picture')


class ModelCommand(InputCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_option('--n_pred', 'Days to predict')
        self.add_option('--date_start', 'Starting date of the prediction')
        self.add_option('--date_end', 'Ending date of the prediction')
        self.add_option('--model', 'Model name')
        self.add_option('--ticker', 'Ticker (instrument) name to predict')
