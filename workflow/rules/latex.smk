rule generate_hist_balance:
    input: 'reports\\trade\\figures\\hist.png'
    output: 'reports\\balance.tex'
    shell: 'python -m src.latex.generate_balance_hist --input {input} --output {output}'

rule generate_function_to_optimize_plot:
    input: 'reports\\trade\\figures\\function_to_optimize.png'
    output: 'reports\\function_to_optimize.tex'
    shell: 'python -m src.latex.generate_function_to_optimize --input {input} --output {output}'

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
         rules.generate_function_to_optimize_plot.output,
         rules.generate_hist_balance.output,
         'spbu_diploma\\main_example.tex'
    output: 'spbu_diploma\\main_example.pdf'
    log: 'logs\\generate_all.log'
    conda: 'envs/default.yaml' # noqa
    shell: 'python spbu_diploma\\generate.py'