rule download_yahoo:
    output: 'data\\raw\\{ticker}.csv'
    log: 'logs\\download_yahoo\\{ticker}.log'
    conda: 'envs/default.yaml' # noqa
    shell: 'python -m src.data.download --output {output} --ticker {wildcards.ticker}'
