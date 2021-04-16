#!/usr/bin/env python3

class MyPreprocessor:

    def preprocess(self,data_input,data_labels=None):
        return (data_input, data_labels) if data_labels is not None else data_input
