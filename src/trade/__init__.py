import pandas as pd
from sklearn.metrics import accuracy_score

from src.trade.model import fit_model, get_x_y_train_test


class Simulation:
    vol = 0
    log = []

    def __init__(self, cap, verbose=False):
        self.cap = cap
        self.verbose = verbose

    def play_simulation(self, df, label_column='label'):
        self.log = []
        for idx, row in enumerate(df.iterrows()):
            ts, label, price = row[0], row[1][label_column], row[1].target

            if label == 'buy' and self.can_buy(price):
                self.buy(price)

            if label == 'sell' and self.can_sell(price):
                self.sell(price)

        self.sell_all(price)
        return self.cap, pd.DataFrame(self.log)

    def buy(self, price):
        available_vol = int(self.cap / price)
        self.vol += available_vol
        self.cap -= price * available_vol
        self.write_to_log('buy', price, available_vol)

    def sell(self, price):
        available_vol = self.vol
        self.vol -= available_vol
        self.cap += price * available_vol
        self.write_to_log('sell', price, available_vol)

    def can_sell(self, price):
        return self.vol > 0

    def can_buy(self, price):
        return self.cap >= price

    def write_to_log(self, type, price, vol):
        self.log.append({'type': type, 'price': price, 'vol': vol, 'cap': self.cap, 'cur_vol': self.vol})

    def sell_all(self, price):
        if self.can_sell(price):
            self.sell(price)


def train_test_split(df, train_size=0.9):
    pivot = int(len(df) * train_size)
    train = df[:pivot].dropna()
    test = df[pivot:].dropna()
    return train, test


def add_results_to_df(df_window, window_len):
    train, test = train_test_split(df_window, train_size=0.9)
    x_train, x_test, y_train, y_test = get_x_y_train_test(train, test, window_len)
    model = fit_model(x_train, y_train)
    y_pred = model.predict(x_test)
    print('Accuracy:', accuracy_score(y_pred, y_test))
    df_window['predicted_label'] = pd.Series(y_pred, index=df_window.index[-y_pred.shape[0]:])
    df_window.dropna(inplace=True)
    return df_window


def get_simulation_results(df_window):
    res, log = Simulation(1000).play_simulation(df_window, label_column='predicted_label')
    best_res, log = Simulation(1000).play_simulation(df_window, label_column='label')
    return res, best_res
