class Model():

    def __init__(self, n=14, verbose=False, **params):
        self.n = n
        self.verbose = verbose

    def fit(self, X, verbose=False):
        pass

    def predict(self, X):
        pass

    def predict_for_report(self, X, date_start, date_end):
        pass

    def _print(self, message):
        if self.verbose:
            print(message)
