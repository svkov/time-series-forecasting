import pandas as pd

from src.utils import substract_n_days


class Model:

    def __init__(self, df, n=14, verbose=False, column_name='price', **params):
        self.n = n
        self.verbose = verbose
        self.column_name = column_name
        self.params = params

    def fit(self, df, verbose=False):
        pass

    def predict(self, df):
        raise NotImplementedError()

    def predict_for_report(self, df, date_start, date_end):
        dates = pd.date_range(date_start, date_end)
        preds = {}
        for pivot in dates:
            signal = df.loc[:pivot].dropna()
            pred = self.__class__(signal, n=self.n, column_name=self.column_name, **self.params).predict(signal)
            self.insert_to_dict(preds, [self.column_name], pred)
        date_start = substract_n_days(date_start, self.n)
        date_end = substract_n_days(date_end, self.n)
        dates = pd.date_range(date_start, date_end)
        return pd.DataFrame(preds, index=dates)

    def insert_to_dict(self, preds, columns, pred):
        for i, column in enumerate(columns):
            for n in range(self.n):
                key = f'{column} n{n + 1}'
                val = self.get_val_by_pred(pred, n, i)
                if key in preds:
                    preds[key].append(val)
                else:
                    preds[key] = [val]

    @staticmethod
    def get_val_by_pred(pred, n, i, *args, **kwargs):
        return pred[n]

    def _print(self, message):
        if self.verbose:
            print(message)
