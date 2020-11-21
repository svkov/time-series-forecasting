import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from src.models.model import Model
from src.utils import transform_date_start


class MyVAR(Model):

    def __init__(self, df, n=14, column_name='price', freq='1d', **kwargs):
        super().__init__(df, n, **kwargs)
        self.column_name = column_name
        self.model = VAR(df, freq=freq)
        self.results = self.model.fit()

    def predict(self, X):
        lag_order = self.results.k_ar
        return self.results.forecast(X.values[-lag_order:], self.n)

    @staticmethod
    def get_val_by_pred(pred, n, i, **kwargs):
        return pred[n, i]
