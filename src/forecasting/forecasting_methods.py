import pandas as pd

from src.models.arima import ARIMA
from src.models.baseline import Baseline
from src.models.fourier import Fourier
from src.models.test import Test
from src.models.var import MyVAR
from src.models.wavelet import Wavelet


class Forecaster:

    def __init__(self, model_type):
        self.model_type = model_type

    def _model(self, df, n, column_name, **kwargs):
        return self.model_type(df, n=n, column_name=column_name, **kwargs)

    def get_model(self, df, n, column_name):
        return self._model(df, n, column_name)

    def forecast(self, df, n_pred, column_name, date_start, date_end):
        return self.get_model(df, n_pred, column_name).predict_for_report(df, date_start, date_end)


class FourierForecaster(Forecaster):
    def __init__(self):
        super().__init__(Fourier)

    def get_model(self, df, n, column_name):
        return self._model(df, n, column_name, n_harm=1, trend_deg=0)


models = {
    'arima': Forecaster(ARIMA),
    'baseline': Forecaster(Baseline),
    'test': Forecaster(Test),
    'fourier': FourierForecaster(),
    'wavelet': Forecaster(Wavelet),
    'var': Forecaster(MyVAR)
}
