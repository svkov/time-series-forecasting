from data.fetch import DataFetcher
import datetime
import pandas as pd


def load_data():
    df = load_data_file('data.csv', index_col='date', parse_dates=True)
    last_date = df.index[-1]
    if datetime.datetime.now().date() != last_date:
        return fetch_all_data()
    return df


def load_data_file(file='data.csv', **kwargs):
    try:
        return pd.read_csv(file, **kwargs)
    except FileNotFoundError:
        return fetch_all_data()


def fetch_all_data():
    try:
        with open('api_key.txt', 'r') as f:
            key = f.read()
            fetcher = DataFetcher(key)
            df = fetcher.fetch(1)
            df.to_csv('data.csv')
            return df
    except FileNotFoundError:
        raise FileNotFoundError('Создайте файл api_key.txt и запишите туда ключ API')


if __name__ == '__main__':
    fetch_all_data()
