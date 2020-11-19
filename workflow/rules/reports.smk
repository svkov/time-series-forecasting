rule plot_raw:
    input: 'data\\raw\\data_{ticker}.csv'
    output: 'reports\\figures_raw\\{ticker}.png'
    shell: 'python -m src.plots.plot_raw --input {input} --output {output} --name {wildcards.ticker}'

rule plot_pred:
    input: 'data\\interim\\{ticker}.csv', expand('reports\\forecast_{model}\\{{ticker}}.csv', model=config['models'])
    params: models=config['models'], path_to_pred='reports/forecast_', n_pred=config['n_pred']
    output: 'reports\\figures_pred\\{ticker}.png'
    shell: 'python -m src.plots.plot_pred --input {input[0]} --output {output} --models "{params.models}" --name {wildcards.ticker} --path_to_pred {params.path_to_pred} --n_pred {params.n_pred}'

rule generate_latex:
    input:
        expand('reports\\figures_pred\\{ticker}.png', ticker=config['tickers']),
    output: 'reports\\diploma.tex'
    shell:
        'python -m src.latex.generate_target_file --input "{input}" --output {output}'

rule generate_all:
    input: 'reports\\diploma.tex'
    output: 'spbu_diploma\\main_example.pdf'
    shell: 'python spbu_diploma\\generate.py'