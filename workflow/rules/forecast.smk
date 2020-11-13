rule forecast_model:
    input: 'data/interim/{ticker}.csv'
    output: 'reports/forecast_{model}/{ticker}.csv'
    params: n_pred=config['n_pred'], date_start=config['date_start']
    shell: 'python -m src.forecasting.{wildcards.model} --input {input} --output {output} --n_pred {params.n_pred} --date_start {params.date_start}'
