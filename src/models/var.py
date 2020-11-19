import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from src.models.model import Model
from src.utils import transform_date_start


class MyVAR(Model):

    def __init__(self, df, n=14, column_name='price', freq='1d', **kwargs):
        super().__init__(n, **kwargs)
        self.column_name = column_name
        self.model = VAR(df, freq=freq)
        self.results = self.model.fit()

    def predict(self, X):
        lag_order = self.results.k_ar
        return self.results.forecast(X.values[-lag_order:], self.n)

    def predict_for_report(self, X, date_start, date_end):
        index = pd.date_range(date_start, date_end)
        preds = {}
        for pivot in index:
            signal = X.loc[:pivot].dropna()
            pred = MyVAR(signal, n=self.n).predict(signal)
            for i, column in enumerate(signal.columns):
                for n in range(self.n):
                    key = f'{column} n{n + 1}'
                    val = pred[n, i]
                    if key in preds:
                        preds[key].append(val)
                    else:
                        preds[key] = [val]
        date_start = transform_date_start(date_start, self.n)
        date_end = transform_date_start(date_end, self.n)
        dates = pd.date_range(date_start, date_end)
        return pd.DataFrame(preds, index=dates)
