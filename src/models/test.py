from src.models.model import Model
import pandas as pd


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
