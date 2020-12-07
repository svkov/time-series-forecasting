rule plot_raw:
    input: 'data\\raw\\data_{ticker}.csv'
    output: 'reports\\figures_raw\\{ticker}.png'
    shell: 'python -m src.plots.plot_raw --input {input} --output {output} --name {wildcards.ticker}'

rule plot_pred:
    input: 'reports\\forecast\\{ticker}.csv'
    params: models=config['models'] + ['stacking'], n_pred=config['n_pred']
    output: 'reports\\figures_pred\\{ticker}.png'
    shell: 'python -m src.plots.plot_pred --input {input} --output {output} --models "{params.models}" --name {wildcards.ticker} --n_pred {params.n_pred}'

rule metrics:
    input: 'reports\\forecast\\{ticker}.csv'
    params: models=config['models'] + ['stacking'], n_pred=config['n_pred'], metrics=config['metrics']
    output: 'reports\\metrics\\metrics_{ticker}.csv'
    shell: 'python -m src.plots.get_metrics --input {input} --output {output} --models "{params.models}" --name {wildcards.ticker} --n_pred {params.n_pred} --metrics "{params.metrics}"'

rule generate_latex:
    input:
        expand('reports\\figures_pred\\{ticker}.png', ticker=config['tickers']),
    output: 'reports\\diploma.tex'
    shell:
        'python -m src.latex.generate_target_file --input "{input}" --output {output}'

rule generate_ticker_table:
    input:
        'workflow\\config.yaml'
    output:
        'reports\\tickers.tex'
    shell:
        'python -m src.latex.generate_tickers_table --config {input} --output {output}'

rule generate_result_table:
    input: expand('reports\\metrics\\metrics_{ticker}.csv', ticker=config['tickers'])
    output: 'reports\\result_table.tex'
    shell: 'python -m src.latex.generate_table --input "{input}" --output {output}'

rule generate_all:
    input: 'reports\\diploma.tex', 'reports\\result_table.tex', 'reports\\tickers.tex'
    output: 'spbu_diploma\\main_example.pdf'
    shell: 'python spbu_diploma\\generate.py'

rule best_metrics:
    input: expand('reports\\metrics\\metrics_{ticker}.csv', ticker=config['tickers']),
    output: 'reports\\metrics\\best_metrics.csv'
    shell: 'python -m src.plots.best_metrics --input "{input}" --output {output}'
