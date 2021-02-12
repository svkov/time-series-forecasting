rule trade_data:
    input: rules.processed.output
    output: 'data\\processed\\trade\\{ticker}.csv'
    params: n=config['n_trade']
    log: 'logs\\trade_data\\{ticker}.log'
    conda: 'envs/default.yaml' # noqa
    shell: 'python -m src.data.to_trade --input {input} --output {output} --n {params.n}'


rule best_window_sizes:
    input: rules.trade_data.output
    output: 'reports\\trade\\window_sizes\\{ticker}.json'
    params: n=config['n_trade'], models=config['models_trade']
    log: 'logs\\best_window_sizes\\{ticker}.log'
    conda: 'envs/default.yaml' # noqa
    shell: 'python -m src.data.find_best_window_size_trade --input {input} --output {output} --n {params.n} --model_types "{params.models}"'


rule forecast_trade:
    input:
        input=rules.trade_data.output,
        window=rules.best_window_sizes.output
    output:
          'reports\\trade\\forecast\\{ticker}.json'
    params:
          n=config['n_trade'],
          model_types=config['models_trade']
    log: 'logs\\forecast_trade\\{ticker}.log'
    conda: 'envs/default.yaml' # noqa
    shell:
         'python -m src.forecasting.forecast_trade_model --input {input.input}'
         ' --output {output} --window {input.window}'
         ' --n {params.n} --model_types "{params.model_types}"'

rule plot_trade_accuracy:
    input: rules.forecast_trade.output
    output: 'reports\\trade\\figures_accuracy\\{ticker}.png'
    params: n=config['n_trade']
    log: 'logs\\plot_trade_accuracy\\{ticker}.log'
    conda: 'envs/default.yaml' # noqa
    shell: 'python -m src.plots.plot_trade_accuracy --input {input} --output {output} --n {params.n}'

rule play_simulation:
    input: rules.forecast_trade.output
    output: 'reports\\trade\\simulation\\logs\\{ticker}_{model_type}.csv'
    params: n=config['n_trade'], budget=config['trade_budget']
    log: 'logs\\play_simulation\\{ticker}\\{model_type}.log'
    conda: 'envs/default.yaml' # noqa
    shell: 'python -m src.plots.play_simulation --input {input} --output {output}' \
    ' --n {params.n} --model_type {wildcards.model_type} --budget {params.budget}'

rule simulation_results:
    input: expand('reports\\trade\\simulation\\logs\\{ticker}_{model_type}.csv', ticker=config['tickers'], model_type=config['models_trade'])
    output: 'reports\\trade\\simulation\\result.csv'
    params: budget=config['trade_budget']
    log: 'logs\\simulation_results.log'
    conda: 'envs/default.yaml' # noqa
    shell: 'python -m src.plots.simulation_results --input "{input}" --output {output} --budget {params.budget}'

rule trade_hists:
    input: rules.processed.output
    output: 'reports\\trade\\figures\\hist.png'
    log: 'logs\\trade_hist.log'
    conda: 'envs/default.yaml' # noqa
    params: n=config['n_trade'], thresh=config['thresh_trade_hist'], instrument=config['trade_instrument_hist']
    shell: 'python -m src.plots.trade_hists --input {input} --output {output} --n {params.n} --thresh {params.thresh} --instrument {params.instrument}'

rule plot_function_to_optimize:
    input: rules.processed.output
    output: 'reports\\trade\\figures\\function_to_optimize.png'
    log: 'logs\\trade\\trade_hist.log'
    conda: 'envs/default.yaml' # noqa
    params: n=config['n_trade'], instrument=config['trade_instrument_hist'], freq=config['trade_imbalance_freq']
    shell: 'python -m src.plots.plot_trade_function_to_optimize --input {input} --output {output}'
           ' --n {params.n} --instrument {params.instrument} --freq {params.freq}'
