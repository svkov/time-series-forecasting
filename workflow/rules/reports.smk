rule plot_raw:
    input: 'data\\raw\\data_{ticker}.csv'
    output: 'reports\\figures_raw\\{ticker}.png'
    shell: 'python -m src.plots.plot_raw --input {input} --output {output} --name {wildcards.ticker}'

rule plot_pred:
    input: 'reports\\forecast\\aggregated\\{ticker}.csv'
    params: models=config['models'] + ['stacking'], n_pred=config['n_pred']
    output: 'reports\\forecast\\figures\\{ticker}.png'
    shell: 'python -m src.plots.plot_pred --input {input} --output {output} --models "{params.models}" --name {wildcards.ticker} --n_pred {params.n_pred}'

rule metrics:
    input: 'reports\\forecast\\aggregated\\{ticker}.csv'
    params: models=config['models'] + ['stacking'], n_pred=config['n_pred'], metrics=config['metrics']
    output: 'reports\\forecast\\metrics\\{ticker}.csv'
    shell: 'python -m src.plots.get_metrics --input {input} --output {output} --models "{params.models}" --name {wildcards.ticker} --n_pred {params.n_pred} --metrics "{params.metrics}"'

rule generate_latex:
    input:
        expand('reports\\forecast\\figures\\{ticker}.png', ticker=config['tickers']),
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
    input: expand('reports\\forecast\\metrics\\{ticker}.csv', ticker=config['tickers'])
    output: 'reports\\result_table.tex'
    shell: 'python -m src.latex.generate_table --input "{input}" --output {output}'

rule generate_all:
    input: 'reports\\diploma.tex', 'reports\\result_table.tex', 'reports\\tickers.tex', 'spbu_diploma\\main_example.tex'
    output: 'spbu_diploma\\main_example.pdf'
    shell: 'python spbu_diploma\\generate.py'

rule best_metrics:
    input: expand('reports\\forecast\\metrics\\{ticker}.csv', ticker=config['tickers']),
    output: 'reports\\forecast\\metrics\\best\\metrics.csv'
    shell: 'python -m src.plots.best_metrics --input "{input}" --output {output}'

rule best_window_sizes:
    input: 'data\\processed\\trade\\{ticker}.csv'
    output: 'reports\\trade\\window_sizes\\{ticker}.json'
    params: n=config['n_trade'], models=config['models_trade']
    shell: 'python -m src.data.find_best_window_size_trade --input {input} --output {output} --n {params.n} --model_types "{params.models}"'

rule plot_trade_accuracy:
    input: 'reports\\trade\\forecast\\{ticker}.json'
    output: 'reports\\trade\\figures_accuracy\\{ticker}.png'
    params: n=config['n_trade']
    shell: 'python -m src.plots.plot_trade_accuracy --input {input} --output {output} --n {params.n}'

rule play_simulation:
    input: 'reports\\trade\\forecast\\{ticker}.json'
    output: 'reports\\trade\\simulation\\logs\\{ticker}_{model_type}.csv'
    params: n=config['n_trade'], budget=config['trade_budget']
    shell: 'python -m src.plots.play_simulation --input {input} --output {output}' \
    ' --n {params.n} --model_type {wildcards.model_type} --budget {params.budget}'

rule simulation_results:
    input: expand('reports\\trade\\simulation\\logs\\{ticker}_{model_type}.csv', ticker=config['tickers'], model_type=config['models_trade'])
    output: 'reports\\trade\\simulation\\result.csv'
    params: budget=config['trade_budget']
    shell: 'python -m src.plots.simulation_results --input "{input}" --output {output} --budget {params.budget}'
