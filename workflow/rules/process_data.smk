rule interim:
    input: 'data\\raw\\data_{ticker}.csv'
    output: 'data\\interim\\{ticker}.csv'
    shell: 'python -m src.data.to_interim --input {input} --output {output}'

rule processed:
    input: expand('data\\interim\\{ticker}.csv', ticker=config['tickers'])
    output: 'data\\processed\\all.csv'
    shell: 'python -m src.data.to_processed --input "{input}" --output {output}'

