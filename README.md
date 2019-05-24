# Low-rank-Multimodal-Fusion

This is the repository for "Efficient Low-rank Multimodal Fusion with Modality-Specific Factors", Liu and Shen, et. al. ACL 2018.

## Dependencies

Python 2.7

PyTorch 0.3.0

sklearn

numpy


## Data for Experiments

The processed data for the experiments (CMU-MOSI, IEMOCAP, POM) can be downloaded here:

https://drive.google.com/open?id=1CixSaw3dpHESNG0CaCJV6KutdlANP_cr

To run the code, you should download the pickled datasets and put them in the `data` directory.

Note that there might be NaN values in acoustic features, you could replace them with 0s.

## Training Your Model

To run the code for experiments (grid search), use the scripts `train_xxx.py`. They have some commandline arguments as listed here:

________________________________

`run_id`: an user-specified unique ID to ensure that saved results/models don't override each other.

`epochs`: the number of maximum epochs in training. Note that the actual number of epochs will be determined also by the `patience` argument.

`patience`: the number of epochs the model is allowed to fluctuate without improvements on the validation set during training. E.g when the patience is set to 5 and the model's performance fluctuates without exceeding previous best for 5 epochs consecutively, the training stops.

`output_dim`: for regression tasks it is usually set as 1. But for IEMOCAP and POM the tasks can be viewed as a multitask learning problem where the model is required to predict the level of presence of all emotions/speaker traits at once. In that case you can set it to multiple dimensions.

`signiture`: optional string for comment.

`cuda`: whether or not to use GPU in training

`data_path`: the path to the data directory. Defaults to './data', but if you prefer storing the data else where you can change this.

`model_path`: the path to the directory where the models that are trained are saved.

`output_path`: the path to the directory where the grid search results are saved.

`max_len`: the maximum length of training data. Longer/shorter sequences will be truncated/padded.

`emotion`: (exclusive for IEMOCAP) specifies which emotion category you want to train the model to predict. Can be 'happy', 'sad', 'angry', 'neutral'.

_____________________________

An example would be

`python train_mosi.py --run_id 19260817 --epochs 50 --patience 20 --output_dim 1 --signiture test_run`

## Hyperparameters

Some hyper parameters for reproducing the results in the paper are in the `hyperparams.txt` file.
