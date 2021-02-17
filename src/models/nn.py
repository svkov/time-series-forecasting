import click
import tensorflow as tf
import pandas as pd

from src.models.neuron_prepare import RawData, WindowGenerator
from src.utils.click_commands import ModelCommand


def compile_and_fit(model, window, patience=2, max_epochs=30):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


@click.command(cls=ModelCommand)
def nn(input, output, logs, n_pred, date_start, date_end, model, ticker):
    n_in = 50
    n_pred = int(n_pred)
    df = pd.read_csv(input, index_col=0, parse_dates=True)
    raw_data = RawData(df)
    window_generator = WindowGenerator(n_in, n_pred, n_pred, raw_data)

    column_indices = {name: i for i, name in enumerate(df.columns)}
    num_features = df.shape[1]

    multi_linear_model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        tf.keras.layers.Dense(n_pred * num_features,
                              kernel_initializer=tf.initializers.zeros),
        tf.keras.layers.Reshape([n_pred, num_features])
    ])

    history = compile_and_fit(multi_linear_model, window_generator)

    print(history)

    val_performance = multi_linear_model.evaluate(window_generator.val)
    test_performance = multi_linear_model.evaluate(window_generator.test, verbose=0)

    print(val_performance)
    print(test_performance)

    # multi_window.plot(multi_linear_model)


if __name__ == '__main__':
    nn()  # noqa
