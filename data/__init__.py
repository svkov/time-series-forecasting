from data.fetch import DataFetcher
import pandas as pd

def load_data():
    return pd.read_csv('data.csv', index_col='date')


if __name__ == '__main__':
    with open('api_key.txt', 'r') as f:
        key = f.read()
        fetcher = DataFetcher(key)
        df = fetcher.fetch(1)
        df.to_csv('data.csv')
