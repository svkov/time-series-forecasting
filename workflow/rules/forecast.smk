rule forecast_fourier:
    input: 'data/raw/data_{ticker}.csv'
    output: 'reports/forecast_fourier/{ticker}.csv'
    params: n_pred=config['n_pred'], date_start=config['date_start']
    shell: 'python -m src.forecasting.fourier --input {input} --output {output} --n_pred {params.n_pred} --date_start {params.date_start}'


rule forecast_wavelet:
    input: 'data/raw/data_{ticker}.csv'
    output: 'reports/forecast_wavelet/{ticker}.csv'
    params: n_pred=config['n_pred'], date_start=config['date_start']
    shell: 'python -m src.forecasting.wavelet --input {input} --output {output} --n_pred {params.n_pred} --date_start {params.date_start}'
