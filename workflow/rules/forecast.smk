# forecast time series

rule forecast_model:
    input: 'data\\processed\\all.csv'
    output: 'reports\\forecast\\models\\{model}\\{ticker}.csv'
    params: n_pred=config['n_pred'], date_start=config['date_start'], date_end=config['date_end']
    shell: 'python -m src.forecasting.forecast_model --input {input} --output {output} --n_pred {params.n_pred} --date_start {params.date_start} --date_end {params.date_end} --model {wildcards.model} --ticker {wildcards.ticker}'

rule forecast_stacking:
    input:
        all='data\\processed\\all.csv',
        # to be sure that all predictions are generated
        models=expand('reports\\forecast\\models\\{model}\\{{ticker}}.csv', model=config['models'])
    output: 'reports\\forecast\\stacking\\{ticker}.csv'
    params:
        models=config['models'],
        main_dir='reports\\forecast\\models\\'

    shell: 'python -m src.forecasting.forecast_stacking --input {params.main_dir}'
           ' --input_all {input.all} --output {output}'
           ' --ticker {wildcards.ticker} --models "{params.models}"'

rule aggregate_forecast:
    input: expand('reports\\forecast\\models\\{model}\\{{ticker}}.csv', model=config['models']) + ['reports\\forecast\\stacking\\{ticker}.csv']
    output: 'reports\\forecast\\aggregated\\{ticker}.csv'
    shell: 'python -m src.forecasting.aggregate_forecast --input "{input}" --output {output}'


# _________________________________________________
# forecast trade
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