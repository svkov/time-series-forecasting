import datetime

from app import db
from typing import List


class Ticker(db.Model):
    __tablename__ = 'ticker'

    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String, unique=False, nullable=False)
    date = db.Column(db.Date, unique=False, nullable=False)
    open = db.Column(db.Numeric, unique=False, nullable=True)
    close = db.Column(db.Numeric, unique=False, nullable=True)
    high = db.Column(db.Numeric, unique=False, nullable=True)
    low = db.Column(db.Numeric, unique=False, nullable=True)
    volume = db.Column(db.Numeric, unique=False, nullable=True)

    @property
    def serialize(self):
        return {
            'id': self.id,
            'ticker': self.ticker,
            'date': self.date.strftime('%Y-%m-%d'),
            'open': float(self.open),
            'close': float(self.close),
            'high': float(self.high),
            'low': float(self.low),
            'volume': float(self.volume)
        }

    @staticmethod
    def delete_ticker(ticker):
        delete_q = Ticker.__table__.delete().where(Ticker.ticker == ticker)
        db.session.execute(delete_q)
        db.session.commit()

    @staticmethod
    def save_df(ticker, df):
        values = []
        for name, row in df.iterrows():
            # date = datetime.datetime.strptime(name, '%Y-%m-%d')
            date = name.date()
            t = Ticker(ticker=ticker, date=date, open=row.Open, close=row.Close, high=row.High, low=row.Low,
                       volume=row.Volume)
            values.append(t)

        Ticker.delete_ticker(ticker)
        db.session.add_all(values)
        db.session.commit()

    @staticmethod
    def get_len():
        return len(Ticker.query.all())

    @staticmethod
    def is_ticker_exist(ticker):
        ticker = Ticker.query.filter_by(ticker=ticker).first()
        if ticker is None:
            return False
        return True

    @staticmethod
    def get_ticker(ticker):
        return [i.serialize for i in Ticker.query.filter_by(ticker=ticker).all()]
