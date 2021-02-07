include: 'process_data.smk'
include: 'forecast.smk'

rule plot_raw:
    input: rules.download_yahoo.output
    output: 'reports\\figures_raw\\{ticker}.png'
    log: 'logs\\plot_raw\\{ticker}.log'
    conda: 'envs/default.yaml' # noqa
    shell: 'python -m src.plots.plot_raw --input {input} --output {output} --name {wildcards.ticker}'

rule plot_pred:
    input: rules.aggregate_forecast.output
    params: models=config['models'] + ['stacking'], n_pred=config['n_pred']
    output: 'reports\\forecast\\figures\\{ticker}.png'
    log: 'logs\\plot_pred\\{ticker}.log'
    conda: 'envs/default.yaml' # noqa
    shell: 'python -m src.plots.plot_pred --input {input} --output {output} --models "{params.models}" --name {wildcards.ticker} --n_pred {params.n_pred}'

rule metrics:
    input: rules.aggregate_forecast.output
    params: models=config['models'] + ['stacking'], n_pred=config['n_pred'], metrics=config['metrics']
    output: 'reports\\forecast\\metrics\\{ticker}.csv'
    log: 'logs\\metrics\\{ticker}.log'
    conda: 'envs/default.yaml' # noqa
    shell: 'python -m src.plots.get_metrics --input {input} --output {output} --models "{params.models}" --name {wildcards.ticker} --n_pred {params.n_pred} --metrics "{params.metrics}"'

rule generate_latex:
    input: expand('reports\\forecast\\figures\\{ticker}.png', ticker=config['tickers']),
    output: 'reports\\diploma.tex'
    log: 'logs\\generate_latex.log'
    conda: 'envs/default.yaml' # noqa
    shell:
        'python -m src.latex.generate_target_file --input "{input}" --output {output}'

# replace path to config with values from it
rule generate_ticker_table:
    input: 'workflow\\config.yaml'
    output: 'reports\\tickers.tex'
    log: 'logs\\generate_ticker_table.log'
    conda: 'envs/default.yaml' # noqa
    shell:
        'python -m src.latex.generate_tickers_table --config {input} --output {output}'

rule generate_result_table:
    input: expand('reports\\forecast\\metrics\\{ticker}.csv', ticker=config['tickers'])
    output: 'reports\\result_table.tex'
    log: 'logs\\generate_results_table.log'
    conda: 'envs/default.yaml' # noqa
    shell: 'python -m src.latex.generate_table --input "{input}" --output {output}'

rule generate_all:
    input:
         rules.generate_latex.output,
         rules.generate_result_table.output,
         rules.generate_ticker_table.output,
         'spbu_diploma\\main_example.tex'
    output: 'spbu_diploma\\main_example.pdf'
    log: 'logs\\generate_all.log'
    conda: 'envs/default.yaml' # noqa
    shell: 'python spbu_diploma\\generate.py'

rule best_metrics:
    input: expand('reports\\forecast\\metrics\\{ticker}.csv', ticker=config['tickers']),
    output: 'reports\\forecast\\metrics\\best\\metrics.csv'
    log: 'logs\\best_metrics\\log.logs'
    conda: 'envs/default.yaml' # noqa
    shell: 'python -m src.plots.best_metrics --input "{input}" --output {output}'

rule best_window_sizes:
    input: rules.trade_data.output
    output: 'reports\\trade\\window_sizes\\{ticker}.json'
    params: n=config['n_trade'], models=config['models_trade']
    log: 'logs\\best_window_sizes\\{ticker}.log'
    conda: 'envs/default.yaml' # noqa
    shell: 'python -m src.data.find_best_window_size_trade --input {input} --output {output} --n {params.n} --model_types "{params.models}"'

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
