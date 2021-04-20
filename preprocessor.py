#!/usr/bin/env python3
import numpy as np


class MyPreprocessor:

    def preprocess(self, data_input, data_labels=None):
        """
        This method transforms each series into the series of log-returns, then it normalizes the entire dataset
        :param data_input: input dataset
        :param data_labels: input labels
        :return: preprocessed dataset
        """
        preprocessed_data_input = MyPreprocessor.to_log_returns(data_input)
        preprocessed_data_input = MyPreprocessor.normalize(preprocessed_data_input)
        return (preprocessed_data_input, data_labels) if data_labels is not None else preprocessed_data_input

    def data_loader(self, file):
        """
        This method loads the dataset
        :param file: file path
        :return: loaded dataset
        """
        return np.loadtxt(file, delimiter=',')

    def data_writer(self, x, file):
        """
        This method writes labels on a text file
        :param x: labels
        :param file: file path of the new file
        """
        if x.dtype == np.int8:
            np.savetxt(file, x, fmt='%i')
        else:
            np.savetxt(file, x)

    @staticmethod
    def to_log_returns(stocks):
        """
        This function gets the set of series of log-returns from a given set of time series.
        :param stocks: a matrix containing several time series of stocks prices
        :return: a matrix where each stock price p_t is substituted with log(p(t)/p(t-1))
        """

        return np.log(stocks[:,1:])-np.log(stocks[:,:-1])

    @staticmethod
    def normalize(data):
        """
        This method normalizes the dataset with a global mean and a global variance
        :param dataset:
        :return: normalized dataset
        """
        return (data - np.mean(data))/np.std(data)