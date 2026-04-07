from sklearn.linear_model import Ridge as SkRidge
from openprc.reservoir.readout.base import BaseReadout

class Ridge(BaseReadout):
    def __init__(self, regularization=1e-6):
        # fit_intercept=False is CRITICAL here
        # because the input matrix now includes a column of 1s
        self.model = SkRidge(alpha=regularization, fit_intercept=False)
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        self.model.fit(X, y)
        self.weights = self.model.coef_
        self.bias = 0.0 # Bias is handled by the first weight

    def predict(self, X):
        return self.model.predict(X)