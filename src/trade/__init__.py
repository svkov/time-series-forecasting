import pandas as pd


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
            if self.verbose:
                print(f'Volume: {self.vol}, price: {price}, idx:{idx}, label: {label}')

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

