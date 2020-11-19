rule forecast_model:
    input: 'data\\interim\\{ticker}.csv'
    output: 'reports\\forecast_{model}\\{ticker}.csv'
    params: n_pred=config['n_pred'], date_start=config['date_start'], date_end=config['date_end']
    shell: 'python -m src.forecasting.forecast_model --input {input} --output {output} --n_pred {params.n_pred} --date_start {params.date_start} --date_end {params.date_end} --model {wildcards.model}'
