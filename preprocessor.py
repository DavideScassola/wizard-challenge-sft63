#!/usr/bin/env python3
import numpy as np


class MyPreprocessor:

    def preprocess(self, data_input, data_labels=None):
        preprocessed_data_input = MyPreprocessor.to_log_returns(data_input)
        preprocessed_data_input = MyPreprocessor.normalize(preprocessed_data_input)
        return (preprocessed_data_input, data_labels) if data_labels is not None else preprocessed_data_input

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
        return (data - np.mean(data))/np.std(data)
