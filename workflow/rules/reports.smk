rule plot_raw:
    input: 'data/raw/data_{ticker}.csv'
    output: 'reports/figures_raw/{ticker}.png'
    shell: 'python -m src.plots.plot_raw --input {input} --output {output} --name {wildcards.ticker}'

rule plot_pred:
    input: 'data/interim/{ticker}.csv', expand('reports/forecast_{model}/{{ticker}}.csv', model=config['models'])
    params: models=config['models'], path_to_pred='reports/forecast_'
    output: 'reports/figures_pred/{ticker}.png'
    shell: 'python -m src.plots.plot_pred --input {input[0]} --output {output} --models "{params.models}" --name {wildcards.ticker} --path_to_pred {params.path_to_pred}'