import requests
import pandas as pd
from datetime import datetime


class DataFetcher:

    def __init__(self, api_key, limit=2000, freq_type='day', fsym='BTC', tsym='USD'):
        self.api_key = api_key
        self.limit = limit
        self.freq_type = freq_type
        self.fsym = fsym
        self.tsym = tsym

    @property
    def url(self):
        return f'https://min-api.cryptocompare.com/data/v2/histo{self.freq_type}?fsym={self.fsym}&tsym={self.tsym}&limit={self.limit}&api_key={self.api_key}&all_data=true'

    @staticmethod
    def get_df_from_url(url):
        resp = requests.get(url)
        data = resp.json()['Data']['Data']
        return pd.DataFrame(data)

    @staticmethod
    def timestamp_to_date(df):
        df['date'] = df['time'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M'))
        return df.set_index('date')

    def get_url_timestamp(self, min_timestamp):
        return f'{self.url}&toTs={min_timestamp}'

    def fetch(self, max_iter):
        df = self.get_df_from_url(self.url)
        for i in range(max_iter):
            min_timestamp = df['time'].min()
            url_timestamp = self.get_url_timestamp(min_timestamp)
            new_part = self.get_df_from_url(url_timestamp)
            df = df.append(new_part)
        df.drop_duplicates(inplace=True)
        df = self.timestamp_to_date(df)
        df.sort_index(inplace=True)
        return df
