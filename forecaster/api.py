from flask_restful import Resource, reqparse
import sys
import requests
import json
import config
sys.path.append('../')


# from src.models.baseline import Baseline

class Hello(Resource):
    def get(self):
        return {'hello': 'world'}


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
        print(final_url)
        response = requests.get(final_url)
        if response.status_code != 200:
            return {'error': f'Status code: {response.status_code}'}

        return json.loads(response.content)

    def get(self):
        return {'btc-usd': self.make_request('btc-usd', 'data')}
