import click
from src.trade import Simulation
import numpy as np
from src.plots.plot_trade_accuracy import open_json
import pandas as pd

from src.utils.click_commands import InputCommand


def map_int_to_str(x):
    if x == 0:
        return 'hold'
    if x == 1:
        return 'sell'
    if x == 2:
        return 'buy'
    raise ValueError(f'Bad value: {x}, must be 0, 1 or 2')


def get_simulation_df(res):
    models = list(res.keys())
    res_df = pd.DataFrame()
    res_df['date'] = [i['date'] for i in res[models[0]]]
    res_df['target'] = [i['target'][0] for i in res[models[0]]]
    res_df['actual_labels'] = [i['test'][0] for i in res[models[0]]]
    res_df.set_index('date', inplace=True)

    for model in models:
        res_df[model] = [i['pred'][0] for i in res[model]]

    res_df = res_df.sort_index()[:'2021-01-01']
    res_df['random_labels'] = np.random.randint(0, 3, (len(res_df),)).tolist()
    res_df['random_labels'] = res_df['random_labels'].apply(map_int_to_str)
    return res_df


def play(simulation_df, model, cap=15000):
    cap, log = Simulation(cap).play_simulation(simulation_df, model)
    return cap, log


@click.command(cls=InputCommand)
@click.option('--n')
@click.option('--budget')
@click.option('--model_type')
def play_simulation(input, output, logs, n, budget, model_type):
    budget = int(budget)
    n = int(n)
    res = open_json(input)
    simulation_df = get_simulation_df(res)
    cap, log = play(simulation_df, model_type, cap=budget)
    log.index.name = cap
    log.to_csv(output)


if __name__ == '__main__':
    play_simulation()  # noqa
