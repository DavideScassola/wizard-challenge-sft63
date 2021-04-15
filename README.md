# Trality Wizard Challenge

Welcome to the Trality Wizard Challenge! Once upon a time an infamous wizard wandered the lands of Cryptopia. People
gathered from all regions of the world when he appeared and requested his guidance. The wizard had a special gift: 
He was able to make predictions about the future. While the neighboring land Fiatia was still trading with sticks and stones, Cryptopia was
an advanced culture, where all transactions were made using a digital currency called Bitib. The wizards predictions were always right about
future price movements of Bittib, and the wizard was often handsomely rewarded for his efforts to make predictions.

You however, are a believer in science. You don't believe in wizardry, but in the power of sound statistical analysis. You want to prove
to the people that you can easily achieve what the wizard has been doing for centuries using the newly emerging field of Machine Learning 
in Cryptopia.

## Data 

In the `data` folder, you are given three files.

* `train_input.txt` 36.000 x 100 entries (comma-separated)
* `train_labels.txt`  36.000 x 1 entries
* `test_input.txt` 4000 x 100 entries

## Tasks

1. Your main goal is to try to accurately predict the labels for `test_input.txt` and thus prove the wizard's magic wrong. 
2. Structure your code according to the description beneath and thereby ensure reusability to permanently put the wizard out of business. 

### Architecture

You are free to add to or improve the architecture, this is only a guideline
for the basic structure. 

The provided `main.py` uses three classes `Pipeline`, `Preprocessor` and `Model`
which are left for you to implement.
`Model` is supposed to handle building, fitting and predicting with the model.
`Preprocessor` shall contain all logic for preprocessing of the data.
`Pipeline` handles the execution.

In the end, the whole training, prediction and evaluation should be triggered by
executing `python3 main.py`.

## Deliverables

1. `test_labels.txt` containing the predicted labels for `test_input.txt`
2. `wizard.zip` containing your code
3. A short description of how you approached the task and interpretation of the results

E-mail both to christopher@trality.com.
If you don't feel comfortable sharing your ml-model you can obfuscate or
replace it.

## Evaluation

We will evaluate your `test_labels.txt` using F1-score and the methodology and
code you have used to make your predictions.

