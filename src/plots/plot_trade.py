import click
import pandas as pd
import json


@click.command()
@click.option('--input')
@click.option('--output')
def plot_trade(input, output):
    pass

if __name__ == '__main__':
    plot_trade() # noqa