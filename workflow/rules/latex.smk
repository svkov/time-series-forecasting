import os

labels = 'workflow\\labels.yaml'

shell_template = 'python -m src.latex.{} --input {} --output {} --name {} --labels {} --logs {}'


figures_template = 'reports\\forecast\\figures\\{}.png'
# figures_forecast = {ticker: figures_template.format(ticker) for ticker in config['tickers']}
figures_forecast = {ticker: figures_template.format(ticker) for ticker in ['BTC-USD']}

pictures = {
    'function_to_optimize': 'reports\\trade\\figures\\function_to_optimize.png',
    'balance_hist': 'reports\\trade\\figures\\hist.png',
    'figures_accuracy': 'reports\\trade\\figures_accuracy\\BTC-USD.png'
}

pictures.update(figures_forecast)

tables = {
    'simulation_results': 'reports\\trade\\simulation\\result.csv',
    'result_table': 'reports\\forecast\\metrics\\all\\all.csv',
    'tickers': 'reports\\description\\tickers.csv',
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
        command = shell_template.format('generate_table', input, output, params, labels, log)
        shell(command)

rule generate_all:
    input:
         expand('reports\\latex\\pictures\\{picture}.tex', picture=pictures.keys()),
         expand('reports\\latex\\tables\\{table}.tex', table=tables.keys()),
         'spbu_diploma\\main_example.tex'
    output: 'spbu_diploma\\main_example.pdf'
    log: 'logs\\generate_all.log'
    conda: 'envs/default.yaml' # noqa
    shell: 'python spbu_diploma\\generate.py'