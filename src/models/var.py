import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from statsmodels.tsa.api import VAR
from src.models.model import Model
from src.utils import substract_n_days


class MyVAR(Model):

    def __init__(self, df, n=14, column_name='price', freq='1d', **kwargs):
        super().__init__(df, n, **kwargs)
        self.column_name = column_name
        self.model = VAR(df.dropna(), freq=freq)
        try:
            self.results = self.model.fit()
        except LinAlgError:
            raise ValueError('Не получилось зафитить VAR')

    def predict(self, X):
        lag_order = self.results.k_ar
        return self.results.forecast(X.values[-lag_order:], self.n)

    @staticmethod
    def get_val_by_pred(pred, n, i, **kwargs):
        return pred[n, i]
