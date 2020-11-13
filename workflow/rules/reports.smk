rule plot_raw:
    input: 'data/raw/data_{ticker}.csv'
    output: 'reports/figures_raw/{ticker}.png'
    shell: 'python -m src.plots.plot_raw --input {input} --output {output} --name {wildcards.ticker}'