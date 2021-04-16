#!/usr/bin/env python3
import xgboost as xgb
import numpy as np

class MyModel:
    def __init__(self, experiment):
        #TODO

        # useless but I want it to be compatible with the interface
        self.experiment = experiment
        self.model = self.initialize_model()

    def initialize_model(self):
        return None

    def fit(self, train_input, train_labels):
        dtrain = xgb.DMatrix(train_input, label=train_labels + 1)
        param = {'objective': 'multi:softmax', "eval_metric": "merror", 'num_class': 3}
        gboost = xgb.train(param, dtrain, 5)
        self.model = gboost

    def predict(self, test_input):
        dtest = xgb.DMatrix(test_input)
        return (self.model.predict(dtest)-1).astype(np.int)