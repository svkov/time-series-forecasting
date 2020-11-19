import pandas as pd

from src.models.arima import ARIMA
from src.models.baseline import Baseline
from src.models.fourier import Fourier
from src.models.test import Test
from src.models.wavelet import Wavelet


def forecast_arima(df: pd.DataFrame, n_pred: int, date_start: str, date_end: str) -> pd.DataFrame:
    model = ARIMA(df, n=n_pred, column_name='Close')
    return model.predict_for_report(df, date_start, date_end)


def forecast_baseline(df: pd.DataFrame, n_pred: int, date_start: str, date_end: str) -> pd.DataFrame:
    model = Baseline(n=n_pred, column_name='Close')
    return model.predict_for_report(df, date_start, date_end)


def forecast_fourier(df: pd.DataFrame, n_pred: int, date_start: str, date_end: str) -> pd.DataFrame:
    model = Fourier(df, n_harm=1, trend_deg=0, n=n_pred, column_name='Close')
    return model.predict_for_report(df, date_start, date_end)


def forecast_test(df: pd.DataFrame, n_pred: int, date_start: str, date_end: str) -> pd.DataFrame:
    model = Test(df, n=n_pred, column_name='Close')
    return model.predict_for_report(df, date_start, date_end)


def forecast_wavelet(df: pd.DataFrame, n_pred: int, date_start: str, date_end: str) -> pd.DataFrame:
    model = Wavelet(df, n=n_pred, column_name='Close')
    return model.predict_for_report(df, date_start, date_end)


models = {
    'arima': forecast_arima,
    'baseline': forecast_baseline,
    'test': forecast_test,
    'fourier': forecast_fourier,
    'wavelet': forecast_wavelet
}
