rule download_yahoo:
    output:
          'data/raw/data_{ticker}.csv'
    shell:
         'python -m src.data.download --output {output} --ticker {wildcards.ticker}'