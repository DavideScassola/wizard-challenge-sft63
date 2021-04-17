#!/usr/bin/env python3
import numpy as np


class MyPreprocessor:

    def preprocess(self, data_input, data_labels=None):
        preprocessed_data_input = MyPreprocessor.to_log_returns(data_input)
        return (preprocessed_data_input, data_labels) if data_labels is not None else preprocessed_data_input

    @staticmethod
    def to_log_returns(stocks):
        return MyPreprocessor.get_returns(stocks, log=True, normalize=False)

    @staticmethod
    def get_returns(stocks, log=True, normalize=False):

        tot_tickers, tot_days = stocks.shape
        returns = np.zeros((tot_tickers, tot_days - 1))
        for d in range(1, tot_days):
            returns[:, d - 1] = stocks[:, d] / stocks[:, d - 1]

        if log:
            returns = np.log(returns)

        if normalize:
            means = np.mean(returns, axis=1)
            stds = np.std(returns, axis=1)
            means = means.reshape(means.shape[0], 1)
            stds = stds.reshape(stds.shape[0], 1)
            returns = (returns - means) / stds

        return returns
