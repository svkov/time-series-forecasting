import os
import click
import pandas as pd


def profit(revenue, cost):
    return (revenue - cost) / cost * 100


@click.command()
@click.option('--input')
@click.option('--output')
@click.option('--budget')
def simulation_results(input, output, budget):
    budget = int(budget)
    input = input.split()

    tickers = []
    models = []
    caps = []
    profits = []
    for result_file in input:
        filename = os.path.split(result_file)[-1]
        ticker, model = filename.replace('.csv', '').split('_')
        df = pd.read_csv(result_file, index_col=0)
        cap = round(float(df.index.name), 2)

        tickers.append(ticker)
        models.append(model)
        caps.append(cap)

        profit_value = round(profit(cap, budget), 2)
        profits.append(profit_value)

    results = {
        'ticker': tickers,
        'models': models,
        'start_budget': [budget for _ in range(len(tickers))],
        'budget': caps,
        'profit%': profits
    }
    results = pd.DataFrame(results)
    results.to_csv(output)


if __name__ == '__main__':
    simulation_results()  # noqa
