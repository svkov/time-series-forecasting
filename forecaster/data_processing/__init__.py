import pandas as pd


def raw_to_interim(df):
    date_start, date_end = df.index[0], df.index[-1]
    dates = pd.date_range(date_start, date_end)
    df = df.reindex(dates).ffill().bfill()
    df['Date'] = df.index
    df.set_index('Date', inplace=True)
    return df


def transform_data(data):
    res = pd.DataFrame()
    for key, df in data.items():
        res[f'{key} Open'] = df['Open']
        res[f'{key} Close'] = df['Close']
        res[f'{key} High'] = df['High']
        res[f'{key} Low'] = df['Low']
        res[f'{key} Volume'] = df['Volume']
    return res
