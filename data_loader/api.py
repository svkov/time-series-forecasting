from datetime import time

import redis
import yfinance as yf
from flask_restful import Resource, reqparse

from cache import cache
from model import Ticker


def Hello(Resource):

    def get(self):
        return {'routes': ['download', 'exist', 'data']}

class DB(Resource):
    @staticmethod
    def download_ticker(ticker, period='3y', interval='1d'):
        return yf.Ticker(ticker).history(period=period, interval=interval)

    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('ticker')
        args = parser.parse_args()
        ticker = args['ticker']

        if ticker:
            try:
                df = self.download_ticker(ticker)
                Ticker.save_df(ticker, df)
                last_date = df.index[-1].strftime('%Y-%m-%d')
            except IndexError:
                return {'error': 'Invalid ticker name'}
        else:
            last_date = None
            ticker = None
        return {'added': ticker, 'last_date': last_date, 'all_len': len(Ticker.get_ticker(ticker))}


class IsExist(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('ticker')
        args = parser.parse_args()
        ticker = args['ticker']
        return {'response': Ticker.is_ticker_exist(ticker)}


class Data(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('ticker')
        args = parser.parse_args()
        ticker = args['ticker']
        return {'response': Ticker.get_ticker(ticker)}
