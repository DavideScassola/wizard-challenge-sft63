#!/usr/bin/env python3
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split


class MyModel:

    # XGBoost parameters
    MAX_TRAIN_STEPS = 50
    EARLY_STOPPING_ROUNDS = 3
    MAX_TREES_DEPTH = 3
    TEST_SIZE = 0.2
    LAMBDA = 4
    SUBSAMPLE = 0.9
    VERBOSE = True

    def __init__(self, experiment):
        self.experiment = experiment  # useless in this case, but I want it to be compatible with the interface
        self.model = None

    def fit(self, train_input, train_labels):
        # splitting into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(train_input, train_labels, test_size=MyModel.TEST_SIZE)

        # transforming data in order to be compatible with xgboost
        d_train = xgb.DMatrix(X_train, label=MyModel.to_xgb_labels(y_train))
        d_valid = xgb.DMatrix(X_val, label=MyModel.to_xgb_labels(y_val))

        # setting xgboost training parameters
        params = {'objective': 'multi:softmax',
                  "eval_metric": "mlogloss",
                  'num_class': 3,
                  'max_depth': MyModel.MAX_TREES_DEPTH,
                  'lambda': MyModel.LAMBDA,
                  'subsample': MyModel.SUBSAMPLE}

        self.model = xgb.train(params, d_train, MyModel.MAX_TRAIN_STEPS, [(d_valid, 'validation')],
                               early_stopping_rounds=MyModel.EARLY_STOPPING_ROUNDS,
                               verbose_eval=MyModel.VERBOSE)

        if MyModel.VERBOSE:
            print("final validation accuracy:", np.mean(self.predict(X_val) == y_val.reshape(-1)))

    def predict(self, test_input):
        dtest = xgb.DMatrix(test_input)
        xgb_prediction = self.model.predict(dtest)
        return MyModel.from_xgb_labels(xgb_prediction)

    @staticmethod
    def to_xgb_labels(labels):
        return labels + 1

    @staticmethod
    def from_xgb_labels(labels):
        return (labels - 1).astype(np.int8)
