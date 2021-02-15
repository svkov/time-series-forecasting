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

rule best_metrics:
    input: expand('reports\\forecast\\metrics\\{ticker}.csv', ticker=config['tickers']),
    output: 'reports\\forecast\\metrics\\best\\metrics.csv'
    log: 'logs\\best_metrics\\log.logs'
    conda: 'envs/default.yaml' # noqa
    shell: 'python -m src.plots.best_metrics --input "{input}" --output {output}'
