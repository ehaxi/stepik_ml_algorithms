import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_regression

class MyLineReg():
    def __init__(self, n_iter, learning_rate=None, sgd_sample=None, metric=None,
                 reg=None, l1_coef=None, l2_coef=None, weights=None):
        self.n_iter = n_iter
        self.learning_rate = 0.1 if learning_rate == None else learning_rate
        self.weights = weights
        self.noise = 5
        self.metric = metric
        self.score = 0
        self.reg = reg
        self.l1_coef = 0 if l1_coef == None else l1_coef
        self.l2_coef = 0 if l2_coef == None else l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = 42

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def _l1(self, X_mini_batches, error, y_mini_batch):
        lasso = self.l1_coef * np.sign(self.weights)
        nabla_cost = 2 * (np.dot(error, X_mini_batches)) / len(y_mini_batch) + lasso
        return nabla_cost

    def _l2(self, X_mini_batches, error, y_mini_batch):
        ridge = self.l2_coef * 2 * self.weights
        nabla_cost = 2 * (np.dot(error, X_mini_batches)) / len(y_mini_batch) + ridge
        return nabla_cost

    def _elasticnet(self, X_mini_batches, error, y_mini_batch):
        lasso = self.l1_coef * np.sign(self.weights)
        ridge = self.l2_coef * 2 * self.weights
        elastic_net = lasso + ridge
        nabla_cost = 2 * (np.dot(error, X_mini_batches)) / len(y_mini_batch) + elastic_net
        return nabla_cost

    def _None(self, X_mini_batches, error, y_mini_batch):
        nabla_cost = 2 * (np.dot(error, X_mini_batches)) / len(y_mini_batch)
        return nabla_cost

    def _mae(self, y, y_pred):
        error = y_pred - y
        return np.mean(abs(error))

    def _mse(self, y, y_pred):
        error = y_pred - y
        return np.mean(error ** 2)

    def _rmse(self, y, y_pred):
        error = y_pred - y
        return np.mean(error ** 2) ** 0.5

    def _mape(self, y, y_pred):
        error = y_pred - y
        return 100 * np.mean(abs(error / y))

    def _r2(self, y, y_pred):
        error = y_pred - y
        mean_y = np.mean(y)
        return 1 - (np.mean(error ** 2)) / (np.mean((y - mean_y) ** 2))

    def get_new_rate(self, nabla_cost, iter):
        if type(self.learning_rate) == float:
            return self.weights - self.learning_rate * nabla_cost
        else:
            learning_rate = self.learning_rate(iter+1)
            return self.weights - learning_rate * nabla_cost

    def create_mini_batches(self, X_with_dummy, y):
        if self.sgd_sample == None:
            return X_with_dummy, y

        elif type(self.sgd_sample) == int:
            mini_batches = pd.DataFrame(index=range(self.sgd_sample), columns=range(15))
            mini_batch_answers = pd.Series(index=range(self.sgd_sample))
            sample_row_idx = random.sample(range(X_with_dummy.shape[0]), self.sgd_sample)
            for idx in range(self.sgd_sample):
                mini_batches.iloc[idx] = X_with_dummy.iloc[sample_row_idx[idx]]
                mini_batch_answers.iloc[idx] = y.iloc[sample_row_idx[idx]]
            return mini_batches, mini_batch_answers

        else:
            sgd_sample = round(self.sgd_sample * X_with_dummy.shape[0])
            mini_batches = pd.DataFrame(index=range(sgd_sample), columns=range(15))
            mini_batch_answers = pd.Series(index=range(sgd_sample))
            sample_row_idx = random.sample(range(X_with_dummy.shape[0]), sgd_sample)
            for idx in range(sgd_sample):
                mini_batches.iloc[idx] = X_with_dummy.iloc[sample_row_idx[idx]]
                mini_batch_answers.iloc[idx] = y.iloc[sample_row_idx[idx]]
            return mini_batches, mini_batch_answers

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        new_index = [i for i in range(X.shape[0])]
        reindex_X = X.set_index(pd.Index(new_index))
        random.seed(self.random_state)
        n_samples = reindex_X.shape[0]
        dummy_features = pd.DataFrame(np.ones(n_samples))
        X_with_dummy = pd.concat([dummy_features, reindex_X], axis=1)
        X_with_dummy.columns = [col for col in range(X_with_dummy.shape[1])]
        n_features = X_with_dummy.shape[1]

        if self.weights is None:
            self.weights = np.ones(n_features)

        if verbose:
            start_error = np.mean((np.dot(X_with_dummy, self.weights) - y) ** 2)
            print(f"start | loss: {start_error} | {self.metric}: {self.score}")

        for iter in range(self.n_iter):
            X_mini_batches, y_mini_batch = getattr(self, "create_mini_batches")(X_with_dummy, y)
            y_pred = np.dot(X_with_dummy, self.weights)
            y_pred_mini_batch = np.dot(X_mini_batches, self.weights)
            error = y_pred - y
            error_mini_batch = y_pred_mini_batch-y_mini_batch
            nabla_cost = getattr(self, '_' + str(self.reg))(X_mini_batches, error_mini_batch, y_mini_batch)
            self.weights = getattr(self, "get_new_rate")(nabla_cost, iter)

            if self.metric:
                self.score = getattr(self, '_' + self.metric)(y, y_pred)

            if verbose and iter % verbose == 0:
                print(f"{iter} | loss: {np.mean(error ** 2)} | {self.metric}: {self.score}")

    def get_coef(self):
        return np.mean(self.weights[1:]) # Exclude the first value because it's matches dummy features

    def predict(self, X: pd.DataFrame):
        new_index = [i for i in range(X.shape[0])]
        reindex_X = X.set_index(pd.Index(new_index))
        n_samples = X.shape[0]
        dummy_features = pd.DataFrame(np.ones(n_samples))
        X_with_dummy = pd.concat([dummy_features, reindex_X], axis=1)
        X_with_dummy.columns = [f"col_{col}" for col in X_with_dummy.columns]
        y_pred = np.dot(X_with_dummy, self.weights)
        return y_pred

    def get_best_score(self):
        return self.score

# FOR LOCAL TESTS

X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
print(MyLineReg(50, 0.1, 0.1).fit(X, y))