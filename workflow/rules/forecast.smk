rule forecast_model:
    input: 'data\\processed\\all.csv'
    output: 'reports\\forecast_{model}\\{ticker}.csv'
    params: n_pred=config['n_pred'], date_start=config['date_start'], date_end=config['date_end']
    shell: 'python -m src.forecasting.forecast_model --input {input} --output {output} --n_pred {params.n_pred} --date_start {params.date_start} --date_end {params.date_end} --model {wildcards.model} --ticker {wildcards.ticker}'

rule forecast_stacking:
    input: 'data\\processed\\all.csv',
         expand('reports\\forecast_{model}\\{ticker}.csv', model=config['models'], ticker=config['tickers'])
    output: 'reports\\meta_forecast_stacking\\{ticker}.csv'
    params: input='reports/forecast_', models=config['models']
    shell: 'python -m src.forecasting.forecast_stacking --input {params.input} --input_all {input[0]} --output {output} --ticker {wildcards.ticker} --models "{params.models}"'

rule aggregate_forecast:
    input: expand('reports\\forecast_{model}\\{{ticker}}.csv', model=config['models']) + ['reports\\meta_forecast_stacking\\{ticker}.csv']
    output: 'reports\\forecast\\{ticker}.csv'
    shell: 'python -m src.forecasting.aggregate_forecast --input "{input}" --output {output}'

rule forecast_trade:
    input:
        input='data\\processed\\trade\\{ticker}.csv',
        window='reports\\trade\\window_sizes\\{ticker}.json'
    output:
          'reports\\trade\\forecast\\{ticker}.json'
    params:
          n=config['n_trade'],
          model_types=config['models_trade']
    shell:
         'python -m src.forecasting.forecast_trade_model --input {input.input}'
         ' --output {output} --window {input.window}'
         ' --n {params.n} --model_types "{params.model_types}"'