#!/usr/bin/env python3

from sklearn.metrics import f1_score, get_scorer
import numpy as np


class Pipeline:

    # set of evaluation metrics
    evaluationMetrics = {"f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average=None),
                         "accuracy": get_scorer("accuracy")._score_func}

    def __init__(self, name: str, experiment):
        self.name = name
        self.experiment = experiment
        self.data = None
        self.preprocessor = None
        self.model = None
        self.data_loader = Pipeline.default_data_loader
        self.data_writer = Pipeline.default_data_writer

    def set_data(self, data: dict):
        self.data = data

    def add_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor

    def add_model(self, model):
        self.model = model

    def fit(self):
        print("fitting")
        train_input, train_labels = self.get_preprocessed_train_data()
        self.model.fit(train_input, train_labels)

    def eval(self, metrics):
        print("evaluating")

        train_input, true_labels = self.get_preprocessed_train_data()
        predicted_labels = self.model.predict(train_input)
        evaluations = {m: Pipeline.evaluationMetrics[m](true_labels, predicted_labels) for m in metrics}
        return evaluations

    def run(self, f):
        f(self.experiment, self)

    def get_unprocessed_data(self, data_name):
        if data_name is None:
            return None
        file = self.data[data_name]
        return self.data_loader(file)

    def get_preprocessed_data(self, data_input_name, data_labels_name=None):
        unprocessed_data_input = self.get_unprocessed_data(data_input_name)
        unprocessed_data_labels = self.get_unprocessed_data(data_labels_name)
        return self.preprocessor.preprocess(unprocessed_data_input, unprocessed_data_labels)

    def get_preprocessed_train_data(self):
        return self.get_preprocessed_data("train_input", "train_labels")

    def write_prediction(self):
        test_input = self.get_preprocessed_data("test_input")
        predicted_labels = self.model.predict(test_input)
        self.data_writer(predicted_labels, "./test_labels.txt")

    @staticmethod
    def default_data_loader(file):
        return np.loadtxt(file, delimiter=',')

    @staticmethod
    def default_data_writer(x, file):
        if x.dtype == np.int8:
            np.savetxt(file, x, fmt='%i')
        else:
            np.savetxt(file, x)
