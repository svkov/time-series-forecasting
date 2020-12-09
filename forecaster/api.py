from typing import Union

from flask_restful import Resource, reqparse
import sys
import requests
import json

from simple_ts_forecast.wavelet import Wavelet
from simple_ts_forecast.baseline import Baseline

import config
import pandas as pd


class Hello(Resource):
    def get(self):
        return {'routes': ['/<ticker>/<model>']}


class DataLoaderAPI(Resource):
    host = f'http://{config.data_loader_host}'
    port = f':{config.data_loader_port}/'

    endpoints = ('exist', 'data', 'download')

    @property
    def url(self):
        return self.host + self.port

    def make_request(self, ticker, url):
        if url not in self.endpoints:
            raise ValueError(f'No endpoint {url}')

        final_url = f'{self.url}{url}?ticker={ticker}'
        response = requests.get(final_url)
        if response.status_code != 200:
            return {'error': f'Status code: {response.status_code}'}

        parsed_json = json.loads(response.content)
        return parsed_json

    def load_data(self, ticker) -> Union[pd.DataFrame, dict]:
        if not self.make_request(ticker, 'exist')['response']:
            self.make_request(ticker, 'download')
        parsed_json = self.make_request(ticker, 'data')
        data = pd.DataFrame(parsed_json['response'])
        if data.empty:
            return {'error': f'Cannot load data: {parsed_json}'}
        return data

    def predict(self, df, model):
        try:
            prediction = ModelsManager().predict(df, model)
        except ValueError as e:
            return {'error': f'Cannot make prediction for: {df} because: {e}'}
        return prediction.tolist()

    def get(self, ticker, model):
        data = self.load_data(ticker)
        if 'error' in data:
            return {ticker: data}
        return {ticker: self.predict(data, model)}


class ModelsManager:
    model_map = {
        'baseline': Baseline,
        'wavelet': Wavelet
    }

    def __init__(self):
        self.models_list = list(self.model_map.keys())

    def _is_input_valid(self, df: pd.DataFrame, model):
        if model not in self.models_list:
            raise ValueError(f'Model {model} is not supported!')

        if 'close' not in df.columns:
            raise ValueError(f'No close price in df with columns {df.columns}')

        return True

    def predict(self, df: pd.DataFrame, model: str):
        self._is_input_valid(df, model)
        return self.model_map[model](df, column_name='close').predict(df)
