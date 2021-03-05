import os

labels = 'workflow\\labels.yaml'

shell_template = 'python -m src.latex.{} --input {} --output {} --name {} --labels {} --logs {}'

pictures = {
    'function_to_optimize': 'reports\\trade\\figures\\function_to_optimize.png',
    'balance_hist': 'reports\\trade\\figures\\hist.png',
}

tables = {
    'simulation_results': 'reports\\trade\\simulation\\result.csv',
    'result_table': 'reports\\forecast\\metrics\\all\\all.csv',
}

rule generate_picture:
    input: lambda wildcards: pictures[wildcards.picture]
    output: 'reports\\latex\\pictures\\{picture}.tex'
    params: name=lambda wildcards, output: wildcards['picture']
    log: 'logs\\picture_{picture}.log'
    run:
        command = shell_template.format('generate_picture', input, output, params, labels, log)
        shell(command)

rule generate_table:
    input: lambda wildcards: tables[wildcards.table]
    output: 'reports\\latex\\tables\\{table}.tex'
    params: name=lambda wildcards, output: wildcards['table']
    log: 'logs\\table_{table}.log'
    run:
        command = shell_template.format('generate_table_from_df', input, output, params, labels, log)
        shell(command)

rule generate_latex:
    input: expand('reports\\forecast\\figures\\{ticker}.png', ticker=config['tickers']),
    output: 'reports\\diploma.tex'
    log: 'logs\\generate_latex.log'
    conda: 'envs/default.yaml' # noqa
    params: name='latex_tickers_name'
    shell:
        'python -m src.latex.generate_target_file --input "{input}" --output {output} --name "{params.name}" --logs {log} --labels {labels}'

# replace path to config with values from it
rule generate_ticker_table:
    input: 'workflow\\config.yaml'
    output: 'reports\\tickers.tex'
    log: 'logs\\generate_ticker_table.log'
    conda: 'envs/default.yaml' # noqa
    params: name='ticker_table_name'
    shell:
        'python -m src.latex.generate_tickers_table --config {input} --output {output} --name "{params.name}"  --labels {labels}'

# rule generate_result_table:
#     input: expand('reports\\forecast\\metrics\\{ticker}.csv', ticker=config['tickers'])
#     output: 'reports\\result_table.tex'
#     log: 'logs\\generate_results_table.log'
#     conda: 'envs/default.yaml' # noqa
#     params: name='results_table_name'
#     shell: 'python -m src.latex.generate_table --input "{input}" --output {output} --name "{params.name}" --logs {log}  --labels {labels}'

rule generate_all:
    input:
         expand('reports\\latex\\pictures\\{picture}.tex', picture=pictures.keys()),
         expand('reports\\latex\\tables\\{table}.tex', table=tables.keys()),
         rules.generate_latex.output,
         rules.generate_ticker_table.output,
         'spbu_diploma\\main_example.tex'
    output: 'spbu_diploma\\main_example.pdf'
    log: 'logs\\generate_all.log'
    conda: 'envs/default.yaml' # noqa
    shell: 'python spbu_diploma\\generate.py'