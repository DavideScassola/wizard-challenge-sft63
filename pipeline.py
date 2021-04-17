#!/usr/bin/env python3

from sklearn.metrics import f1_score, get_scorer
import pandas as pd
import numpy as np


class Pipeline:
    evaluationMetrics = {"f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average=None),
                         "accuracy": get_scorer("accuracy")._score_func}

    @staticmethod
    def data_loader(file):
        return pd.read_csv(file, header=None).to_numpy()

    @staticmethod
    def data_writer(x, file):
        if x.dtype == np.int8:
            np.savetxt(file, x, fmt='%i')
        else:
            np.savetxt(file, x)

    def __init__(self, name: str, experiment):
        self.name = name
        self.experiment = experiment
        self.data = None
        self.preprocessor = None
        self.model = None

    def set_data(self, data: dict):
        self.data = data
        print("setting the data")

    def add_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor
        print("adding preprocessor")

    def add_model(self, model):
        self.model = model
        print("adding model")

    def fit(self):
        print("fitting")
        train_input, train_labels = self.get_preprocessed_train_data()
        self.model.fit(train_input, train_labels)

    def eval(self, metrics):
        print("evaluating")

        train_input, true_labels = self.get_preprocessed_train_data()
        predicted_labels = self.model.predict(train_input)
        evaluations = [{m: Pipeline.evaluationMetrics[m](true_labels, predicted_labels)} for m in metrics]
        return evaluations

    def run(self, f):
        f(self.experiment, self)

    def get_unprocessed_data(self, data_name):
        file = self.data[data_name]
        unprocessed_data = Pipeline.data_loader(file)
        return unprocessed_data

    def get_preprocessed_data(self, data_input_name, data_labels_name=None):
        unprocessed_data_input = self.get_unprocessed_data(data_input_name)
        unprocessed_data_labels = self.get_unprocessed_data(data_labels_name) if data_labels_name is not None else None
        preprocessed_data = self.preprocessor.preprocess(unprocessed_data_input, unprocessed_data_labels)
        return preprocessed_data

    def get_preprocessed_train_data(self):
        return self.get_preprocessed_data("train_input", "train_labels")

    def write_prediction(self):
        test_input = self.get_preprocessed_data("test_input")
        predicted_labels = self.model.predict(test_input)
        Pipeline.data_writer(predicted_labels, "./test_labels.txt")
