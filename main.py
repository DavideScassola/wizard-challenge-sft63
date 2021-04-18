#!/usr/bin/env python3
from pipeline import Pipeline
from model import MyModel
from preprocessor import MyPreprocessor


class Experiment(object):
    data = {
        "train_input": 'data/train_input.txt',
        "train_labels": 'data/train_labels.txt',
        "test_input": 'data/test_input.txt',
    }

def main(experiment: Experiment, pipeline: Pipeline):

    pipeline.set_data(experiment.data)

    # Add preprocessors
    pipeline.add_preprocessor(MyPreprocessor())

    # Add models
    pipeline.add_model(MyModel(experiment))

    # Fit
    pipeline.fit()

    # Evaluation
    results = pipeline.eval(metrics=["f1","accuracy"])
    print(results)

    # Prediction
    pipeline.write_prediction()


if __name__ == '__main__':
    pipeline = Pipeline("my_pipeline", Experiment())
    pipeline.run(main)

