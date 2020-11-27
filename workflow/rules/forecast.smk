rule forecast_model:
    input: 'data\\processed\\all.csv'
    output: 'reports\\forecast_{model}\\{ticker}.csv'
    params: n_pred=config['n_pred'], date_start=config['date_start'], date_end=config['date_end']
    shell: 'python -m src.forecasting.forecast_model --input {input} --output {output} --n_pred {params.n_pred} --date_start {params.date_start} --date_end {params.date_end} --model {wildcards.model} --ticker {wildcards.ticker}'

rule forecast_stacking:
    input: 'data\\processed\\all.csv'
    output: 'reports\\meta_forecast_stacking\\{ticker}.csv'
    params: input='reports/forecast_', models=config['models']
    shell: 'python -m src.forecasting.forecast_stacking --input {params.input} --input_all {input} --output {output} --ticker {wildcards.ticker} --models "{params.models}"'
