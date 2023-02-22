import os
import sys

server_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(server_path)

from abc import ABCMeta, abstractmethod

from sklearn.metrics import mean_absolute_error

def eval_goodness(target_container_data, target_powers, label_col):
    data = target_container_data.copy()
    data[label_col] = target_powers
    corr = data.corr().loc[label_col].drop(index=[label_col]).dropna()
    max_corr = 0
    if len(corr) > 0:
        max_corr = max(0, max(corr))
    return max_corr

def eval_err(actual_values, predicted_values):
    return mean_absolute_error(actual_values, predicted_values)

class IsolationTrainer(metaclass=ABCMeta):
    @abstractmethod
    def train(self, data, feature_cols, label_col):
        return NotImplemented # set scaler, fe, and model, return mae

    @abstractmethod
    def predict(self, data, feature_cols):
        return NotImplemented # apply scaler,fe and predict power

    def train_eval(self, data, target_container_data, conditional_data, feature_cols, label_col):
        self.train(data, feature_cols, label_col)
        actual_values = data[label_col].values
        predicted_values = self.predict(data, feature_cols)
        err_val = eval_err(actual_values, predicted_values)
        conditional_powers = self.predict(conditional_data, feature_cols)
        target_powers = target_container_data[label_col] - conditional_powers
        isolation_goodness = eval_goodness(target_container_data, target_powers, label_col)
        return target_powers, isolation_goodness, err_val