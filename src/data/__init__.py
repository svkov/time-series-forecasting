import datetime
import pandas as pd
# from src.data import DataFetcher


def get_name(freq_type):
    return f'data_{freq_type}.csv'


def load_data(freq_type='day'):
    df = load_data_file(get_name(freq_type), freq_type=freq_type, index_col='date', parse_dates=True)
    last_date = df.index[-1]
    if datetime.datetime.now().date() != last_date:
        return fetch_all_data(freq_type)
    return df


def load_data_file(file, freq_type='day', **kwargs):
    try:
        return pd.read_csv(file, **kwargs)
    except FileNotFoundError:
        return fetch_all_data(freq_type)


def fetch_all_data(freq_type: str):
    """
    :param freq_type: day/hour
    :return: df с данными нужной частоты
    """
    try:
        with open('api_key.txt', 'r') as f:
            key = f.read()
            # fetcher = DataFetcher(key, freq_type=freq_type)
            # df = fetcher.fetch(1)
            # df.to_csv(get_name(freq_type))
            # return df
    except FileNotFoundError:
        raise FileNotFoundError('Создайте файл api_key.txt и запишите туда ключ API')


if __name__ == '__main__':
    fetch_all_data('day')
