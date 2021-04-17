

rule interim:
    input: rules.download_yahoo.output
    output: 'data\\interim\\{ticker}.csv'
    log: 'logs\\interim\\{ticker}.log'
    conda: 'envs/default.yaml' # noqa
    shell: 'python -m src.data.to_interim --input {input} --output {output} --logs {log}'

rule processed:
    input: expand('data\\interim\\{ticker}.csv', ticker=config['tickers'])
    output: 'data\\processed\\all.csv'
    log: 'logs\\processed.log'
    conda: 'envs/default.yaml' # noqa
    shell: 'python -m src.data.to_processed --input "{input}" --output {output}'


