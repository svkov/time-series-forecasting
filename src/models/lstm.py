import time
import os
import datetime
from collections import namedtuple

import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.optim as optim

from dateutil.relativedelta import relativedelta
from src.models import Model

SavedFit = namedtuple('SavedFit', 'filename date_test_start datetime_fit mape')


def r2_score(y_test, y_pred, torch_order=False):
    if torch_order:
        y_test, y_pred = y_pred, y_test
    if isinstance(y_test, np.ndarray) and isinstance(y_pred, np.ndarray):
        return 1 - np.mean((y_test - y_pred) ** 2) / np.mean((y_test - np.mean(y_test)) ** 2)
    elif isinstance(y_test, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        return 1 - torch.mean((y_test - y_pred) ** 2).item() / torch.mean((y_test - torch.mean(y_test)) ** 2).item()
    else:
        raise TypeError(f"input_ array must be np.ndarray or torch.Tensor, got {type(y_test)}, {type(y_pred)}")


def mean_absolute_percent_error(y_test, y_pred, torch_order=False):
    if torch_order:
        y_test, y_pred = y_pred, y_test
    if isinstance(y_test, np.ndarray) and isinstance(y_pred, np.ndarray):
        return np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    elif isinstance(y_test, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        return torch.mean(torch.abs((y_test - y_pred) / y_test)) * 100
    else:
        raise TypeError(f"input_ array must be np.ndarray or torch.Tensor, got {type(y_test)}, {type(y_pred)}")


class LSTM(Model):
    """Use this class as another classic models"""

    class _Model(nn.Module):
        """PyTorch RNN model"""

        def __init__(self, input_size, hidden_size, output_size, device):
            super().__init__()
            self.device = device
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.lstm_1 = nn.LSTMCell(self.input_size, self.hidden_size)
            self.lstm_2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
            self.dropout_1 = nn.Dropout(p=0.5)
            self.dropout_2 = nn.Dropout(p=0.1)
            self.linear = nn.Linear(self.hidden_size, self.input_size)
            self.out_linear = nn.Linear(self.input_size, self.output_size)

        def forward(self, x, future=1):
            x = x.to(self.device)
            outputs = []
            # reset the state of LSTM
            # the state is kept till the end of the sequence
            h_t1, c_t1 = self.init_hidden(x.size(0))
            h_t2, c_t2 = self.init_hidden(x.size(0))

            for input_t in x.split(1, dim=1):
                h_t1, c_t1 = self.lstm_1(input_t.squeeze(1), (h_t1, c_t1))
                h_t1 = self.dropout_1(h_t1)
                h_t2, c_t2 = self.lstm_2(h_t1, (h_t2, c_t2))

                output = self.linear(self.dropout_2(h_t2))
                outputs += [self.out_linear(output)]

            for i in range(future - 1):
                h_t1, c_t1 = self.lstm_1(output, (h_t1, c_t1))
                h_t1 = self.dropout_1(h_t1)
                h_t2, c_t2 = self.lstm_2(h_t1, (h_t2, c_t2))
                output = self.linear(self.dropout_2(h_t2))
                outputs += [self.out_linear(output)]
            outputs = torch.stack(outputs, 1).squeeze(2)
            return outputs

        def init_hidden(self, batch_size):
            h_t = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32).to(self.device)
            c_t = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32).to(self.device)
            return h_t, c_t

    def __init__(self, n=14, window=35, lr=0.005, sched_step_size=10, sched_gamma=0.5,
                 model_params=None, model_input_size=1, model_hidden_size=300, model_output_size=1, scaler=None,
                 device=None, gpu_num=0, train_set_prop=0.9, batch_size=175, n_epochs=30,
                 models_dir='lstm_saves/ts_mnpz/',
                 days_between_fits=31, n_fits=3, search_window=14, post_process_coef=0.75):
        """Init model

        Args:
            n (int, optional): future days num to predict. Defaults to 14.
            window (int, optional): window of past data from predict. Defaults to 35.
            lr (float, optional): learning rate of optimizer. Defaults to 0.005.
            sched_step_size (int, optional): lr_scheduler.StepLR step size. Defaults to 10.
            sched_gamma (float, optional): lr_scheduler.StepLR gamma. Defaults to 0.5.
            model_params (dict, optional): dict of params = args to model init. Defaults to dict of 3 params below.
            model_input_size (int, optional): param of Model, num input_ features. Defaults to 1.
            model_hidden_size (int, optional): param of Model, size of hidden layers. Defaults to 300.
            model_output_size (int, optional): param of Model, size of output. Defaults to 1.
            scaler (sklearn.preprocessing.*Scaler, optional): class Scaler for features. Defaults to sklearn.preprocessing.StandardScaler.
            device (torch.device, optional): device train on. Defaults to gpu, if available.
            gpu_num (int, optional): gpu num in sys. Defaults to 0.
            train_set_prop (float, optional): if not providing sate_test_start uses these coef to slicing train data. Defaults to 0.9.
            batch_size (int, optional): batch size for train. Defaults to 175.
            n_epochs (int, optional): number epochs for train. Defaults to 30.
            models_dir (str, optional): path to saves of models. Defaults to 'lstm_saves/ts_mnpz/'.
            days_between_fits (int, optional): days between fits for predict for report. Defaults to 31.
            n_fits (int, optional): number of fits for one test data. Defaults to 3.
            search_window (int, optional): search saved fit up to search_window days back. Defaults to 14.
            post_process_coef (float, optional): in [0, 1]. Defaults to 0.75.
        """
        super().__init__()
        self.model_params = model_params or dict(input_size=model_input_size, hidden_size=model_hidden_size,
                                                 output_size=model_output_size)
        self.device = device or torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
        self.cpu_device = torch.device('cpu')
        self.model = self._Model(**self.model_params, device=self.cpu_device)
        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.sched_step_size = sched_step_size
        self.sched_gamma = sched_gamma
        self.Scaler = scaler or sklearn.preprocessing.StandardScaler
        self.scalers = []
        self.n_in = window
        self.n_out = n
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seeds = [0, 42, 1, 123, 1337, 2000, -1000, 300]
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.days_between_fits = days_between_fits
        self._filename_pattern = 'model_{date_test_start}_{datetime_fit}_{mape:.2f}_.pt'
        self.train_set_prop = train_set_prop
        self.n_fits = n_fits
        self.search_window = search_window
        self.post_process_coef = post_process_coef

    def fit(self, X, verbose=False, date_test_start=None, force_fit=False, load_from_filename=None, saving=True):
        """fit or load LSTM model

        Args:
            X ([pd.DataFrame]): all series to train (and testing model) without Nan
            verbose (bool, optional): if True prints verbose information. Defaults to False.
            date_test_start (str or datetime): Date for first n_out prediction. Defaults to end of 90% of X.
            force_fit (bool, optional): Fit even if exist saved. Defaults to False.
            load_from_filename (str, optional): Filename load from (without dirname). Defaults to None.
        """
        ind = pd.to_datetime(X.index)
        X = X.values
        n_features = X.shape[1]

        if date_test_start is None:
            test_start = int(len(X) * self.train_set_prop)
            date_test_start = pd.to_datetime(ind[test_start])
        else:
            test_start = ind.get_loc(date_test_start) + 1 - self.n_in - self.n_out
        self._test_start = test_start
        self.date_test_start = pd.to_datetime(date_test_start)

        train = X[:test_start].reshape(-1, n_features)
        test = X[test_start:].reshape(-1, n_features)
        trains = []
        tests = []

        for i in range(n_features):
            scaler = self.Scaler()
            series = train[:, i].reshape(-1, 1)
            scaler = scaler.fit(series)
            trains.append(scaler.fit_transform(series))
            tests.append(scaler.transform(test[:, i].reshape(-1, 1)))
            self.scalers.append(scaler)

        shift_size = self.n_in
        train_arr = np.concatenate(trains, 1)
        test_arr = np.concatenate(tests, 1)

        x_train, y_train = self.series_to_supervised(train_arr, self.n_in, self.n_out, shift_size, for_new_arch=True)
        self._x_train = x_train
        self._y_train = y_train

        x_test, y_test = self.series_to_supervised(test_arr, self.n_in, self.n_out, shift_size, for_new_arch=True)
        self._x_test = x_test
        self._y_test = y_test

        if load_from_filename and not force_fit:
            self.load_model(self.models_dir + load_from_filename)
        elif force_fit:
            self._n_fits(self.n_fits, verbose, saving)
        else:
            filename = self.find_nearest_save(self.date_test_start)
            if filename:
                self.load_model(self.models_dir + filename)
            else:
                self._n_fits(self.n_fits, verbose, saving)

    def _n_fits(self, n_fits=3, verbose=False, saving=True):
        info = []
        min_mape = float('inf')
        min_mape_i = 0
        for i in range(n_fits):
            if i < len(self.seeds):
                torch.manual_seed(self.seeds[i])
            else:
                torch.seed()

            self.model = self._Model(**self.model_params, device=self.device)
            self.model.to(self.device)

            self.loss_fn = nn.MSELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.sched_step_size,
                                                       gamma=self.sched_gamma)
            if verbose:
                print(f'START fit {i}')

            train_loss, val_loss, tttime, mape = self.train(self._x_train, self._y_train, self._x_test, self._y_test,
                                                            verbose=verbose)
            if verbose:
                print(f'MAPE on {i} fit = {mape:.4f}, last best = {min_mape:.4f}, elapsed {tttime / 60:.2f}min.\n')
            if min_mape > mape:
                min_mape = mape
                min_mape_i = i
            info.append((self.model, self.loss_fn, self.optimizer, self.scheduler))
            self.model.to(self.cpu_device)
            self.model.device = self.cpu_device
        if verbose:
            print(f'\nTHE BEST Model is {min_mape_i} with MAPE = {min_mape:.4f}\n')

        self.model, self.loss_fn, self.optimizer, self.scheduler = info[min_mape_i]
        self.mape_on_val = min_mape
        if saving:
            self.save_fit()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, X, dates_from_predict=None, post_process=True):
        """
        :param X: all series, same as in fit(), but with additional data at the end
        :type X: pd.DataFrame or np.ndarray
        :param dates_from_predict: indexes of days in X to predict
                                 if None predicts for last date in X
        :return: np.array if predictions for each day in dates_to_predict
        """
        n_features = X.shape[1]
        trains = []
        for i in range(n_features):
            scaler = self.scalers[i]
            series = X.iloc[:, i:i + 1].values
            trains.append(scaler.transform(series))

        X = pd.DataFrame(np.concatenate(trains, 1), index=X.index)
        ind = X.index
        if dates_from_predict is None:
            dates_from_predict = [ind[-1]]

        to_predict = []
        for date in dates_from_predict:
            end_ind = ind.get_loc(date)
            x = X.iloc[end_ind - self.n_in:end_ind, :].values
            to_predict.append(x)

        to_predict = np.array(to_predict)
        x = torch.from_numpy(to_predict).float()

        with torch.no_grad():
            self.model.eval()
            y_pred = self.model(x, future=self.n_out).cpu()
            y_pred = y_pred[:, -self.n_out:].numpy()

        predicted_scaled = self._scale_all_predictions(y_pred)
        predicted_scaled = np.array(predicted_scaled).reshape(len(dates_from_predict), self.n_out)
        columns = [f'n{i + 1}' for i in range(self.n_out)]
        pred = pd.DataFrame(predicted_scaled, index=dates_from_predict, columns=columns)
        if post_process:
            ma = X.loc[pred.index].values[:, :1]
            ppc = self.post_process_coef
            pred = pred - predicted_scaled[:, :1] + (ma * ppc + predicted_scaled[:, :1] * (1 - ppc))
        return pred

    def predict_for_report(self, X, date_start, date_end, current_fit=False, force_fits=False, verbose=False,
                           saving=True, post_process=True):
        date_start = pd.to_datetime(date_start)
        date_end = pd.to_datetime(date_end)
        columns = [f'n{i + 1}' for i in range(self.n_out)]

        if current_fit:
            predicted = self._evaluate_all(self._x_test, self._y_test)
            start = date_start - relativedelta(days=self.n_out)
            ind = pd.date_range(start, periods=len(predicted))
            return pd.DataFrame(predicted, index=ind, columns=columns)

        flag = False
        preds = []
        l_range = (date_end - date_start).days
        for i in range(0, l_range, self.days_between_fits):
            if l_range - (i + self.days_between_fits) < self.n_out:
                flag = True

            new_date_start = date_start + relativedelta(days=i)
            new_end = new_date_start + relativedelta(days=self.days_between_fits - 1)

            if flag:
                new_end = date_end

            if force_fits:
                self.fit(X.loc[:new_end], date_test_start=new_date_start, force_fit=True, verbose=verbose,
                         saving=saving)
            else:
                saved_fit_fn = self.find_nearest_save(new_date_start)
                if saved_fit_fn:
                    self.fit(X.loc[:new_end], date_test_start=new_date_start, load_from_filename=saved_fit_fn,
                             verbose=verbose, saving=saving)
                else:
                    self.fit(X.loc[:new_end], date_test_start=new_date_start, force_fit=True, verbose=verbose,
                             saving=saving)

            predicted = self._evaluate_all(self._x_test, self._y_test)
            start = new_date_start - relativedelta(days=self.n_out)
            ind = pd.date_range(start, periods=len(predicted))
            preds.append(pd.DataFrame(predicted, index=ind, columns=columns))
            if flag:
                break
        pred = pd.concat(preds)
        if post_process:
            predicted_scaled = pred.values
            ma = X.loc[pred.index].values[:, :1]
            ppc = self.post_process_coef
            pred = pred - predicted_scaled[:, :1] + (ma * ppc + predicted_scaled[:, :1] * (1 - ppc))
        return pred

    def save_fit(self):
        checkpoint = {
            'model': self._Model(**self.model_params, device=self.cpu_device),
            'date_test_start': self.date_test_start,
            'state_dict': self.model.state_dict(),
            'mape_on_val': self.mape_on_val
        }
        torch.save(checkpoint,
                   self.models_dir + self._filename_pattern.format(date_test_start=self.date_test_start.date(),
                                                                   datetime_fit=datetime.datetime.now().strftime(
                                                                       "%Y-%m-%d %H%M%S"),
                                                                   mape=self.mape_on_val))

    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.cpu_device)
        self.model = checkpoint['model']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.model.to(self.cpu_device)
        self.mape_on_val = checkpoint['mape_on_val']
        self.date_test_start = checkpoint['date_test_start']

    def list_saved_fits(self):
        filenames = [fn for fn in os.listdir(self.models_dir) if fn.endswith('.pt')]
        list_of_fits = []
        for fn in filenames:
            _, date_test_start, datetime_fit, mape, _ = fn.split('_')
            date_test_start = pd.to_datetime(date_test_start)
            datetime_fit = pd.to_datetime(datetime_fit, format="%Y-%m-%d %H%M%S")
            mape = float(mape)
            list_of_fits.append(SavedFit(fn, date_test_start, datetime_fit, mape))
        return list_of_fits

    def find_nearest_save(self, date):
        date = pd.to_datetime(date)
        saved_fits = self.list_saved_fits()
        nearest = None
        for fit in saved_fits:
            days = (date - fit.date_test_start).days
            if 0 <= days <= self.search_window:
                if (nearest is None or nearest.date_test_start < fit.date_test_start
                        or nearest.datetime_fit < fit.datetime_fit):
                    nearest = fit

        if nearest is None:
            return

        return nearest.filename

    def generate_batch_data(self, x, y, batch_size=None):
        batch_size = batch_size or self.batch_size
        dataset = torch.utils.data.TensorDataset(x.to(self.device), y.to(self.device))
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        yield from loader

    def train(self, x_train, y_train, x_val=None, y_val=None, verbose=False):
        total_train_time = 0
        for epoch in range(self.n_epochs):
            self.model.train()
            start_time = time.time()

            train_loss = 0
            n_batches = 0
            for x_batch, y_batch in self.generate_batch_data(x_train, y_train):
                y_pred = self.model(x_batch, future=self.n_out)
                self.optimizer.zero_grad()
                loss = self.loss_fn(y_pred, y_batch)
                y_pred = y_pred[:, -self.n_out:]
                loss.backward()
                self.optimizer.step()

                n_batches += 1
                train_loss += loss.item()
            self.scheduler.step()
            train_loss /= n_batches

            val_loss, mape = self._validation(x_val, y_val)
            elapsed = time.time() - start_time
            total_train_time += elapsed
            if verbose:
                print(f"Epoch {str(epoch + 1):>02}"
                      f" Train loss: {train_loss:.4f}."
                      f" Validation loss: {val_loss:.4f}."
                      f" Elapsed time: {elapsed:.2f}s.")

        return train_loss, val_loss, total_train_time, mape

    def _validation(self, x_val, y_val):
        if x_val is None or y_val is None:
            return
        with torch.no_grad():
            self.model.eval()
            y_val = y_val.cpu()
            predicted_all = self.model(x_val, future=self.n_out).cpu()
            y_pred = predicted_all[:, -self.n_out:]
            loss = self.loss_fn(predicted_all, y_val)

            y_pred = np.array(self._scale_all_predictions(y_pred.numpy()))
            y_val = np.array(self._scale_all_predictions(y_val[:, -self.n_out:].numpy()))

            mape = mean_absolute_percent_error(y_pred.ravel(), y_val.ravel())
            return loss.item(), mape

    def _evaluate_all(self, x_test, y_test):
        with torch.no_grad():
            self.model.eval()
            predicted_all = self.model(x_test, future=self.n_out).cpu()
            y_pred = predicted_all[:, -self.n_out:]
            loss = self.loss_fn(predicted_all, y_test)

            y_pred = np.array(self._scale_all_predictions(y_pred.numpy()))
            return y_pred.reshape(-1, self.n_out)

    @staticmethod
    def series_to_supervised(data, n_in=1, n_out=1, shift_size=None, index_of_target=0, out_type='torch',
                             for_new_arch=False):
        """Transform n_vars time series with to array of samples and truth future data

        :param data: time series table with dim=(n, n_vars)
        :type data: pd.DataFrame or np.ndarray
        :param n_in: length of sample sequence
        :param n_out: length of truth prediction sample
        :param shift_size: size of shift between sample and future
        :param index_of_target: index in data along second dim of target series
        :param out_type: 'torch' or 'np'
        :return: two np.array: (samples, forecasts) dim=(n_samples, n_features)
        """
        if shift_size is None:
            shift_size = n_in
        x, y = [], []
        n_samples = len(data) - (shift_size + n_out - 1)
        for i in range(n_samples):
            x_i = data[i: i + n_in]
            if for_new_arch:
                y_i = data[i + 1: i + n_out + n_in]
            else:
                y_i = data[i + shift_size: i + n_out + shift_size]
            x.append(x_i)
            y.append(y_i)
        x_arr = np.array(x)
        y_arr = np.array(y)[:, :, index_of_target]
        if out_type == 'torch':
            x_var = torch.from_numpy(x_arr).float()
            y_var = torch.from_numpy(y_arr).float()
            return x_var, y_var
        elif out_type == 'np':
            return x_arr, y_arr
        return x_arr, y_arr

    def _scale_all_predictions(self, predicted):
        predicted_scaled = []
        for prediction in predicted:
            prediction = np.array(prediction)
            pred = prediction.reshape(self.n_out, 1)
            scaled_pred = self.scalers[0].inverse_transform(pred)
            predicted_scaled.append(scaled_pred)
        return predicted_scaled

    def __repr__(self):
        return f'LSTM: model_params={self.model_params}, n={self.n_out}, window_in={self.n_out}, batch_size={self.batch_size}, n_epoch={self.n_epochs}'
