import pandas as pd
import numpy as np
from utils import transform_date_start
from forecasting.model import Model


class Baseline(Model):

    def __init__(self, n=14, column_name='price', **kwargs):
        super().__init__(n, **kwargs)
        self.column_name = column_name

    def predict(self, X):
        return np.array([X[self.column_name].values[-1]] * self.n)

    def predict_for_report(self, X, date_start, date_end):
        X = X[self.column_name]
        # index = X.loc[date_start:date_end].index
        index = pd.date_range(date_start, date_end)

        preds = []
        for pivot in index:
            signal = X.loc[:pivot].dropna().values
            train = signal[-self.n]
            preds.append([train] * self.n)

        date_start = transform_date_start(date_start, self.n)
        date_end = transform_date_start(date_end, self.n)
        # dates = X[date_start:date_end].index
        dates = pd.date_range(date_start, date_end)
        columns = [f'n{i + 1}' for i in range(self.n)]
        return pd.DataFrame(preds, index=dates, columns=columns)
