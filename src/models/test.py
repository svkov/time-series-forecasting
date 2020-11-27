from datetime import timedelta

from src.models.model import Model
import pandas as pd

from src.utils import transform_date_start


class Test(Model):

    def __init__(self, df, n, column_name='price', verbose=False, **kwargs):
        super().__init__(n, verbose=verbose, **kwargs)
        self.df = df
        self.n = n
        self.column_name = column_name

    def predict(self, X, date_start=None):
        if date_start is None:
            return X[self.column_name][:-self.n]

        date_range = pd.date_range(date_start, periods=self.n)
        if date_range[-1] not in X.index:
            raise ValueError('Невозможно сгенерировать тест по таким данным!')

        return X.loc[date_range, self.column_name]

    def predict_for_report(self, df, date_start, date_end):
        df = df[self.column_name]
        index = pd.date_range(date_start, date_end)

        preds = []
        for pivot in index:
            signal = df.loc[pivot:pivot + timedelta(days=self.n - 1)].dropna().values
            preds.append(signal)

        date_start = transform_date_start(date_start, self.n)
        date_end = transform_date_start(date_end, self.n)
        # dates = df[date_start:date_end].index
        dates = pd.date_range(date_start, date_end)
        columns = [f'{self.column_name} n{i + 1}' for i in range(self.n)]
        return pd.DataFrame(preds, index=dates, columns=columns)