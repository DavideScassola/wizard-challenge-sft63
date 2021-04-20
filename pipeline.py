#!/usr/bin/env python3

from sklearn.metrics import f1_score, get_scorer, confusion_matrix


class Pipeline:

    # set of evaluation metrics
    evaluationMetrics = {"f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average=None),
                         "accuracy": get_scorer("accuracy")._score_func,
                         "confusion_matrix": lambda y_true, y_pred: confusion_matrix(y_true, y_pred, normalize = "all")}

    def __init__(self, name, experiment):
        self.name = name
        self.experiment = experiment
        self.data = None
        self.preprocessor = None
        self.model = None

    def set_data(self, data):
        self.data = data

    def add_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor

    def add_model(self, model):
        self.model = model

    def fit(self):
        print("fitting the model ...")
        train_input, train_labels = self.get_preprocessed_train_data()
        self.model.fit(train_input, train_labels)

    def eval(self, metrics, input_name="train_input", labels_name="train_labels"):
        """
        This method evaluates the model on a chosen dataset with a chosen metric
        :param metrics: a list of strings representing the metrics used to evaluate the prediction
        :param input_name: name indicating the input dataset, the default is "train_input"
        :param labels_name: name indicating the labels dataset, the default is "train_labels"
        :return: dictionary associating the name of each metric to the result of its application
        """

        input_data, labels = self.get_preprocessed_data(input_name, labels_name)
        predicted_labels = self.model.predict(input_data)
        evaluations = {m: Pipeline.evaluationMetrics[m](labels, predicted_labels) for m in metrics}
        return evaluations

    def run(self, f):
        f(self.experiment, self)

    def get_unprocessed_data(self, data_name):
        """
        This method loads the selected dataset, using the data loader defined by the preprocessor
        :param data_name: the name of the dataset to load
        :return: the loaded dataset
        """

        if data_name is None:
            return None
        file = self.data[data_name]
        return self.preprocessor.data_loader(file)

    def get_preprocessed_data(self, data_input_name, data_labels_name=None):
        """
        This method loads the selected dataset and preprocesses it using the preprocessor
        :param data_input_name: the name of the input dataset to load
        :param data_labels_name: the name of the labels dataset to load
        :return: preprocessed input data and preprocessed labels
        """
        unprocessed_data_input = self.get_unprocessed_data(data_input_name)
        unprocessed_data_labels = self.get_unprocessed_data(data_labels_name)
        return self.preprocessor.preprocess(unprocessed_data_input, unprocessed_data_labels)

    def get_preprocessed_train_data(self):
        return self.get_preprocessed_data("train_input", "train_labels")

    def write_prediction(self):
        """
        This method computes the predictions for a given test input dataset and then writes them in a text file
        """
        test_input = self.get_preprocessed_data("test_input")
        predicted_labels = self.model.predict(test_input)
        self.preprocessor.data_writer(predicted_labels, "./test_labels.txt")


