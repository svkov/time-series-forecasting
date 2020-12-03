import time

from flask import Flask
from flask_restful import Resource, Api
import yfinance as yf
import redis
from flask_sqlalchemy import SQLAlchemy

from db_config import DATABASE_CONNECTION_URI

app = Flask(__name__)
api = Api(app)
cache = redis.Redis(host='redis', port=6379)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_CONNECTION_URI
db = SQLAlchemy(app)



class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String, unique=False, nullable=False)
    email = db.Column(db.String, unique=False, nullable=False)


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


class DB(Resource):
    def get(self):
        response = {'len': len(User.query.all())}
        user = User(username='abc', email='sdfdss')
        db.session.add(user)
        db.session.commit()
        return response


api.add_resource(DB, '/')
api.add_resource(DataDownload, '/data/')

if __name__ == '__main__':
    db.create_all()
    app.run()
