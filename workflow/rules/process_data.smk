rule interim:
    input: 'data\\raw\\data_{ticker}.csv'
    output: 'data\\interim\\{ticker}.csv'
    shell: 'python -m src.data.to_interim --input {input} --output {output}'

rule processed:
    input: expand('data\\interim\\{ticker}.csv', ticker=config['tickers'])
    output: 'data\\processed\\all.csv'
    shell: 'python -m src.data.to_processed --input "{input}" --output {output}'

rule trade_data:
    input: 'data\\processed\\all.csv'
    output: 'data\\trade\\{ticker}.csv'
    params: n=config['n_trade']
    shell: 'python -m src.data.to_trade --input {input} --output {output} --n {params.n}'
