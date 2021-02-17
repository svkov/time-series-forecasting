import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class RawData:

    def __init__(self, df, ticker, train_ratio=0.5, val_ratio=0.3):
        self.df = df[[f'{ticker} Open', f'{ticker} Close', f'{ticker} High', f'{ticker} Low', f'{ticker} Volume']]
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self._split_data()
        self._normalize_data()

    def _split_data(self):
        train_pivot = int(self.n * self.train_ratio)
        val_pivot = int(self.n * (self.train_ratio+self.val_ratio))

        self._train_df = self.df[:train_pivot]
        self._val_df = self.df[train_pivot:val_pivot]
        self._test_df = self.df[val_pivot:]

    def _normalize_data(self):
        self._train_mean = self._train_df.mean()
        self._train_std = self._train_df.std()

        self._norm_train = (self._train_df - self._train_mean) / self._train_std
        self._norm_val = (self._val_df - self._train_mean) / self._train_std
        self._norm_test = (self._test_df - self._train_mean) / self._train_std

    @property
    def n(self):
        return len(self.df)

    @property
    def train(self):
        return self._norm_train

    @property
    def val(self):
        return self._norm_val

    @property
    def test(self):
        return self._norm_test


class WindowGenerator:
    def __init__(self, input_width, label_width, shift, raw_data,
                 label_columns=None):
        self.raw_data = raw_data

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(self.raw_data.train.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, plot_col, model=None, max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=32, )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.raw_data.train)

    @property
    def val(self):
        return self.make_dataset(self.raw_data.val)

    @property
    def test(self):
        return self.make_dataset(self.raw_data.test)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
