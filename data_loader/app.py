import time

from flask import Flask
from flask_restful import Resource, Api
import yfinance as yf
import redis

app = Flask(__name__)
api = Api(app)
cache = redis.Redis(host='redis', port=6379)


def get_hit_count():
    retries = 5
    while True:
        try:
            return cache.incr('hits')
        except redis.exceptions.ConnectionError as exc:
            if retries == 0:
                raise exc
            retries -= 1
            time.sleep(0.5)


def download_ticker(ticker, period='3y', interval='1d'):
    return yf.Ticker(ticker).history(period=period, interval=interval)


class DataDownload(Resource):
    def get(self):
        df = download_ticker('BTC-USD', '1y')
        return {'data': df.to_json()}


class HelloWorld(Resource):
    def get(self):
        return {'hit_count': get_hit_count()}


api.add_resource(HelloWorld, '/')
api.add_resource(DataDownload, '/data/')

if __name__ == '__main__':
    app.run()
