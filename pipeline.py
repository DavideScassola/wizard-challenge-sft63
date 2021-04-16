#!/usr/bin/env python3

class Pipeline:
    def __init__(self, name: str, experiment):
        # TODO
        self.name = name
        self.experiment = experiment

    def set_data(self, data: dict):
        # TODO
        print("setting the data")
        pass

    def add_preprocessor(self,preprocessor):
        # TODO
        print("adding preprocessor")
        pass

    def add_model(self,model):
        # TODO
        print("adding model")
        pass

    def fit(self):
        # TODO
        print("fitting")
        pass

    def eval(self,metrics):
        # TODO
        print("evaluating")
        pass

    def run(self, f):
        # TODO
        f(self.experiment, self)
