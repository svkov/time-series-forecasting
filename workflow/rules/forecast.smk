include: 'utils.smk'
include: 'process_data.smk'
include: 'reports.smk'
# forecast time series

rule forecast_model:
    input: rules.processed.output
    output: 'reports\\forecast\\models\\{model}\\{ticker}.csv'
    params: n_pred=config['n_pred'], date_start=config['date_start'], date_end=config['date_end']
    log: 'logs\\forecast_model\\{model}\\{ticker}.log'
    conda: 'envs/default.yaml' # noqa
    shell: 'python -m src.forecasting.forecast_model --input {input} --output {output} --n_pred {params.n_pred} --date_start {params.date_start} --date_end {params.date_end} --model {wildcards.model} --ticker {wildcards.ticker}'


rule forecast_stacking:
    input:
        all=rules.processed.output,
        # to be sure that all predictions are generated
        models=rules.forecast_model.output
    output: 'reports\\forecast\\stacking\\{ticker}.csv'
    params:
        models=config['models'],
        main_dir=forecast_stacking_dirname # noqa
    log: 'logs\\forecast_stacking\\{ticker}.csv'
    conda: 'envs/default.yaml' # noqa
    shell: 'python -m src.forecasting.forecast_stacking --input {params.main_dir}'
           ' --input_all {input.all} --output {output}'
           ' --ticker {wildcards.ticker} --models "{params.models}"'

rule aggregate_forecast:
    input: expand('reports\\forecast\\models\\{model}\\{{ticker}}.csv', model=config['models']) + ['reports\\forecast\\stacking\\{ticker}.csv']
    output: 'reports\\forecast\\aggregated\\{ticker}.csv'
    log: 'logs\\aggregate_forecast\\{ticker}.log'
    conda: 'envs/default.yaml' # noqa
    shell: 'python -m src.forecasting.aggregate_forecast --input "{input}" --output {output}'


# _________________________________________________
# forecast trade
rule forecast_trade:
    input:
        input=rules.trade_data.output,
        window=rules.best_window_sizes.output
    output:
          'reports\\trade\\forecast\\{ticker}.json'
    params:
          n=config['n_trade'],
          model_types=config['models_trade']
    log: 'logs\\forecast_trade\\{ticker}.log'
    conda: 'envs/default.yaml' # noqa
    shell:
         'python -m src.forecasting.forecast_trade_model --input {input.input}'
         ' --output {output} --window {input.window}'
         ' --n {params.n} --model_types "{params.model_types}"'