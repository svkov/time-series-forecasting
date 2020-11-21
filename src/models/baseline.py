import pandas as pd
import numpy as np
from src.utils import transform_date_start
from src.models.model import Model


class Baseline(Model):

    def __init__(self, df, n=14, column_name='price', **kwargs):
        super().__init__(df, n, **kwargs)
        self.column_name = column_name

    def predict(self, X):
        return np.array([X[self.column_name].values[-1]] * self.n)

    # def predict_for_report(self, df, date_start, date_end):
    #     df = df[self.column_name]
    #     # index = df.loc[date_start:date_end].index
    #     index = pd.date_range(date_start, date_end)
    #
    #     preds = []
    #     for pivot in index:
    #         signal = df.loc[:pivot].dropna().values
    #         train = signal[-self.n]
    #         preds.append([train] * self.n)
    #
    #     date_start = transform_date_start(date_start, self.n)
    #     date_end = transform_date_start(date_end, self.n)
    #     # dates = df[date_start:date_end].index
    #     dates = pd.date_range(date_start, date_end)
    #     columns = [f'n{i + 1}' for i in range(self.n)]
    #     return pd.DataFrame(preds, index=dates, columns=columns)
