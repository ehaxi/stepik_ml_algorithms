import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

class MyLineReg():
    def __init__(self, n_iter, learning_rate, weights=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.noise = 5

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        n_samples = X.shape[0]
        dummy_features = pd.DataFrame(np.ones(n_samples))
        X_with_dummy = pd.concat([dummy_features, X], axis=1)
        X_with_dummy.columns = [f"col_{col}" for col in X_with_dummy.columns]
        n_features = X_with_dummy.shape[1]

        if self.weights is None:
            self.weights = np.ones(n_features)

        if verbose:
            print(f"start | loss: {np.mean((np.dot(X_with_dummy, self.weights) - y) ** 2)}")

        for iter in range(self.n_iter):
            y_pred = np.dot(X_with_dummy, self.weights)
            error = y_pred - y
            gradient = 2 * (np.dot(error, X_with_dummy)) / len(y)
            self.weights -= self.learning_rate * gradient

            if verbose and iter % verbose == 0:
                print(f"{iter} | loss: {np.mean(error ** 2)}")

        # y_pred = np.dot(X_with_dummy, self.weights) # For local tests
        # print(y_pred) # For local tests

    def get_coef(self):
        return np.mean(self.weights[1:]) # Exclude the first value because it's matches dummy features

    def predict(self, X: pd.DataFrame):
        n_samples = X.shape[0]
        dummy_features = pd.DataFrame(np.ones(n_samples))
        X_with_dummy = pd.concat([dummy_features, X], axis=1)
        X_with_dummy.columns = [f"col_{col}" for col in X_with_dummy.columns]
        y_pred = np.dot(X_with_dummy, self.weights)
        return y_pred

# FOR LOCAL TESTS

# X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
# X = pd.DataFrame(X)
# y = pd.Series(y)
# print(MyLineReg(50, 0.1).fit(X, y))