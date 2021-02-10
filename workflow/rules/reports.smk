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
