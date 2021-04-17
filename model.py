#!/usr/bin/env python3
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split


class MyModel:

    MAX_TRAIN_STEPS = 50
    EARLY_STOPPING_ROUNDS = 5
    MAX_TREES_DEPTH = 4
    TEST_SIZE = 0.2
    VERBOSE_VALIDATION = False

    def __init__(self, experiment):
        # useless but I want it to be compatible with the interface
        self.experiment = experiment
        self.model = None

    def fit(self, train_input, train_labels):
        X_train, X_test, y_train, y_test = train_test_split(train_input, train_labels, test_size=MyModel.TEST_SIZE)
        d_train = xgb.DMatrix(X_train, label=self.to_xgb_labels(y_train))
        d_valid = xgb.DMatrix(X_test, label=self.to_xgb_labels(y_test))

        params = {'objective': 'multi:softmax', "eval_metric": "mlogloss", 'num_class': 3, 'max_depth': MyModel.MAX_TREES_DEPTH}
        validation = [(d_valid, 'validation')]
        gbst = xgb.train(params, d_train, MyModel.MAX_TRAIN_STEPS, validation,
                         early_stopping_rounds=MyModel.EARLY_STOPPING_ROUNDS,
                         verbose_eval=MyModel.VERBOSE_VALIDATION)

        self.model = gbst

    def predict(self, test_input):
        dtest = xgb.DMatrix(test_input)
        xgb_prediction = self.model.predict(dtest)
        return self.from_xgb_labels(xgb_prediction)

    @staticmethod
    def to_xgb_labels(labels):
        return labels + 1

    @staticmethod
    def from_xgb_labels(labels):
        return (labels - 1).astype(np.int8)
