rule process:
    input: 'data\\raw/data_{ticker}.csv'
    output: 'data\\interim\\{ticker}.csv'
    shell: 'python -m src.data.to_interim --input {input} --output {output}'
