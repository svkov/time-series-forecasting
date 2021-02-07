include: 'download_data.smk'


rule interim:
    input: i=rules.download_yahoo.input
    # input: 'data\\raw\\data_{ticker}.csv'
    output: 'data\\interim\\{ticker}.csv'
    log: 'logs\\interim\\{ticker}.log'
    conda: 'envs/default.yaml' # noqa
    shell: 'python -m src.data.to_interim --input {input} --output {output}'

rule processed:
    input: expand('data\\interim\\{ticker}.csv', ticker=config['tickers'])
    output: 'data\\processed\\all.csv'
    log: 'logs\\processed.log'
    conda: 'envs/default.yaml' # noqa
    shell: 'python -m src.data.to_processed --input "{input}" --output {output}'

rule trade_data:
    input: rules.processed.output
    output: 'data\\processed\\trade\\{ticker}.csv'
    params: n=config['n_trade']
    log: 'logs\\trade_data\\{ticker}.log'
    conda: 'envs/default.yaml' # noqa
    shell: 'python -m src.data.to_trade --input {input} --output {output} --n {params.n}'
