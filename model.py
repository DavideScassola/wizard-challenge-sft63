#!/usr/bin/env python3
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split


class MyModel:
    """
    This class implements a basic xgboost model for time series
    """

    # XGBoost parameters
    MAX_TRAIN_STEPS = 100
    EARLY_STOPPING_ROUNDS = 8
    MAX_TREES_DEPTH = 2
    TEST_SIZE = 0.15
    ETA = 0.2
    LAMBDA = 2
    SUBSAMPLE = 0.3
    VERBOSE = True

    def __init__(self, experiment):
        self.experiment = experiment  # useless in this case, but I want it to be compatible with the interface
        self.model = None

    def feature_extension(self, x):
        """
        This method adds two statistics to each time series: a weighted moving average and a weighted moving variance
        :param x: the input dataset (n_samples x time_steps)
        :return: the input dataset in which each time series is extended with a set of statistics
        """
        basis = 1.1
        d_average = MyModel.exponential_decay_average(x, basis)
        d_variance = MyModel.exponential_decay_variance(x, basis)
        return np.concatenate([d_average, d_variance, x], axis=1)

    def fit(self, train_input, train_labels):
        # extending train input with more features
        extended_train_input = self.feature_extension(train_input)

        # splitting into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(extended_train_input, train_labels,
                                                          test_size=MyModel.TEST_SIZE)

        # transforming data in order to be compatible with xgboost
        d_train = xgb.DMatrix(X_train, label=MyModel.to_xgb_labels(y_train))
        d_valid = xgb.DMatrix(X_val, label=MyModel.to_xgb_labels(y_val))

        # setting xgboost training parameters
        params = {'objective': 'multi:softmax',
                  "eval_metric": "mlogloss",
                  'num_class': 3,
                  'max_depth': MyModel.MAX_TREES_DEPTH,
                  'lambda': MyModel.LAMBDA,
                  'subsample': MyModel.SUBSAMPLE,
                  'eta': MyModel.ETA}

        # training the model
        self.model = xgb.train(params, d_train, MyModel.MAX_TRAIN_STEPS, [(d_valid, 'validation')],
                               early_stopping_rounds=MyModel.EARLY_STOPPING_ROUNDS,
                               verbose_eval=MyModel.VERBOSE)

        # evaluating accuracy on validation set
        if MyModel.VERBOSE:
            print("final validation accuracy:",
                  np.mean(MyModel.from_xgb_labels(self.model.predict(d_valid)) == y_val.reshape(-1)))

    def predict(self, test_input):
        # extending train input with more features
        test_input = self.feature_extension(test_input)

        # transforming data in order to be compatible with xgboost
        dtest = xgb.DMatrix(test_input)

        # predicting
        xgb_prediction = self.model.predict(dtest)

        return MyModel.from_xgb_labels(xgb_prediction)

    @staticmethod
    def to_xgb_labels(labels):
        """
        This method transforms the labels in order to make them compatible with xgboost
        :param labels:
        :return: labels compatible with xgboost
        """
        return labels + 1

    @staticmethod
    def from_xgb_labels(labels):
        """
        This method transforms the output labels of xgboost back to the original representation
        :param labels: xgboost compatible labels
        :return: original labels
        """
        return (labels - 1).astype(np.int8)

    @staticmethod
    def exponential_decay_average(x, basis=1.5):
        decay = MyModel.exponential_decay_vector(x.shape[-1], basis)
        return x.dot(decay).reshape(-1, 1)

    @staticmethod
    def exponential_decay_variance(x, basis=1.5):
        decay = MyModel.exponential_decay_vector(x.shape[-1], basis)
        variance = ((x - np.mean(x, axis=1).reshape(-1, 1)) ** 2)
        return variance.dot(decay).reshape(-1, 1)

    @staticmethod
    def exponential_decay_vector(size, basis):
        decay = float(basis) ** (np.arange(size) - size + 1)
        decay = decay / sum(decay)
        return decay
