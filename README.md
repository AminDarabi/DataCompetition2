# IFT6390 Kaggle Competition 2: ASCII Sign Language Classification

This project contains three methods for the second IFT6390 Kaggle competition,
which aims to classify sign language images into 26 classes (24) and then
perform a ascii trick to predict test set labels.

- The first method CNN model developed using PyTorch.
- The second method is a Random Forest model developed from scratch using
only NumPy and Pandas.
- The third method is a AdaBoost model developed using SciKit-Learn.

Requirements
All methods are implemented in Python 3.11 and require the following packages:
- NumPy
- Pandas

The CNN also requires:
- PyTorch
- ndimage from SciPy

The AdaBoost also requires:
- SciKit-Learn

### Installation
To install the project, you can clone this repository or download it as a zip
file. Then, you can install the required packages using pip:
```
$ pip install -r requirements.txt
```

### Usage
To run the models and generate predictions for the Kaggle competition,
you can use one of the following files, based on the method you want to use.

## convolutional.py
This file contains the CNN model developed using PyTorch. It has the following
functions and classes:

- read_sign_mnist_train: This function reads the sign language images from the
csv file and returns training data and labels, and validation data and labels
if the validation mode is enabled, as numpy arrays. It also applies some
augmentation on the data.
- read_sign_mnist_test: This function reads the sign language images from the
csv file and returns test data as two numpy arrays (one for each image of the
test set).
- ascii_task_output: This function takes the predicted labels for both images
of the test set and performs the ascii trick to generate the final predictions.
- Convolutional: This class implements the CNN model. It has the following
public methods:
    - fit: This method trains the model on the given data and labels.
    - predict: This method predicts the labels for the given data.
    - score: This method calculates the accuracy and loss of the model for
    the given data and labels.
- main: This function runs the model and prints the results using the arguments
provided by the user.

To run the model with default hyperparameters using the competition data,
you can use the following command in the terminal:
```
$ python convolutional.py path_to_train.csv path_to_test.csv -o path_to_output.csv
```

You can also customize the following arguments:

- -e or --epochs: This argument specifies the number of epochs for training.
The default value is 25. For example:
    ```
    $ python convolutional.py TRAIN_DATA TEST_DATA -e 10
    ```

- --batch_size: This argument specifies the batch size for training.
The default value is 32. For example:
    ```
    $ python convolutional.py TRAIN_DATA TEST_DATA --batch_size 32
    ```

- --shuffle: This argument enables the shuffling of the training data before
starting the training. For example:
    ```
    $ python convolutional.py TRAIN_DATA TEST_DATA --shuffle
    ```

- -p or --print_loss: This argument enables the printing of the loss and
accuracy during training. For example:
    ```
    $ python convolutional.py TRAIN_DATA TEST_DATA -p
    ```

- --device: This argument specifies the device to use for training and
prediction. The available options are cpu, mps, cuda, and auto.
The default option is auto, which uses cuda or mps if available. For example:
    ```
    $ python convolutional.py TRAIN_DATA TEST_DATA --device cpu
    ```

- --valid_split: This argument set the validation split ratio for splitting
the training data into train and validation sets. The default value is 0.
    ```
    $ python convolutional.py TRAIN_DATA TEST_DATA --valid_split 0.2
    ```

- --without_augment: This argument disables the augmentation of the training
data. For example:
    ```
    $ python convolutional.py TRAIN_DATA TEST_DATA --without_augment
    ```

- -h or --help: This argument shows a help message with the description and
usage of the arguments. For example:
    ```
    $ python base.py -h
    ```

## random_forest.py
This file contains the Random Forest model developed from scratch using only
NumPy and Pandas. It has the following functions and classes:

- read_sign_mnist_train: This function reads the sign language images from the
csv file and returns training data and labels, and validation data and labels
if the validation mode is enabled, as numpy arrays.
- read_sign_mnist_test: This function reads the sign language images from the
csv file and returns test data as two numpy arrays (one for each image of the
test set).
- ascii_task_output: This function takes the predicted labels for both images
of the test set and performs the ascii trick to generate the final predictions.
- pre_process: This function pre-processes the data by convolving some filters
on the images and then max-pooling the results. At the end, it flattens the
images and returns the flattened data.
- RandomForestClassifier: This class implements the Random Forest model. It
has the following public methods:
    - fit: This method trains the model on the given data and labels.
    - predict: This method predicts the labels for the given data.
    - score: This method calculates the accuracy of the model for the given
    data and labels.
- DecisionTreeClassifier: This class implements the Decision Tree model. It
has the following public methods:
    - fit: This method trains the model on the given data and labels.
    - predict: This method predicts the labels for the given data.
    - score: This method calculates the accuracy of the model for the given
    data and labels.
- main: This function runs the model and prints the results using the arguments
provided by the user.

To run the model with default hyperparameters using the competition data,
you can use the following command in the terminal:
```
$ python random_forest.py path_to_train.csv path_to_test.csv -o path_to_output.csv
```

However, you can also customize the following arguments:

- -n or --n_estimators: This argument specifies the number of trees in the
forest. The default value is 300. For example:
    ```
    $ python random_forest.py TRAIN_DATA TEST_DATA -n 50
    ```

- -p or --print_progress: This argument enables the printing of the progress
during training. For example:
    ```
    $ python random_forest.py TRAIN_DATA TEST_DATA -p
    ```

- --valid_split: This argument set the validation split ratio for splitting
the training data into train and validation sets. The default value is 0.
    ```
    $ python random_forest.py TRAIN_DATA TEST_DATA --valid_split 0.2
    ```

- -t or --tree_depth: This argument specifies the maximum depth of the trees.
The default value is 15. For example:
    ```
    $ python random_forest.py TRAIN_DATA TEST_DATA -t 5
    ```

- -b or --bootstrap: This argument specifies bootstrap ratio for sampling the
training data. The default value is 0.3. For example:
    ```
    $ python random_forest.py TRAIN_DATA TEST_DATA -b 0.5
    ```

- -f or --features_ratio: This argument specifies the ratio of features to
randomly select for each tree. The default value is 'sqrt'. acceptable values
are 'sqrt', 'log2', and 'all', an integer numbe, or a float number between 0
and 1. For example:
    ```
    $ python random_forest.py TRAIN_DATA TEST_DATA -f 0.2
    $ python random_forest.py TRAIN_DATA TEST_DATA -f all
    $ python random_forest.py TRAIN_DATA TEST_DATA  12
    ```

- -m or --min_samples: This argument specifies the minimum number of
samples accepted in each leaf. The default value is 1. For example:
    ```
    $ python random_forest.py TRAIN_DATA TEST_DATA -m 5
    ```

- -h or --help: This argument shows a help message with the description and
usage of the arguments. For example:
    ```
    $ python advanced.py -h
    ```

## ada_boost.py
This file contains the AdaBoost model that trains multiple Decision Trees
developed using SciKit-Learn. It has the
following functions and classes:

- read_sign_mnist_train: This function reads the sign language images from the
csv file and returns training data and labels, and validation data and labels
if the validation mode is enabled, as numpy arrays.
- read_sign_mnist_test: This function reads the sign language images from the
csv file and returns test data as two numpy arrays (one for each image of the
test set).
- ascii_task_output: This function takes the predicted labels for both images
of the test set and performs the ascii trick to generate the final predictions.
- main: This function runs the model and prints the results using the arguments
provided by the user.

To run the model with default hyperparameters using the competition data,
you can use the following command in the terminal:
```
$ python ada_boost.py path_to_train.csv path_to_test.csv -o path_to_output.csv
```

However, you can also customize the following arguments:

- -n or --n_estimators: This argument specifies the number of trees in the
forest. The default value is 30. For example:
    ```
    $ python ada_boost.py TRAIN_DATA TEST_DATA -n 50
    ```
- -p or --print_loss: This argument enables the printing of the loss and
accuracy during training. For example:
    ```
    $ python ada_boost.py TRAIN_DATA TEST_DATA -p
    ```

- --valid_split: This argument set the validation split ratio for splitting
the training data into train and validation sets. The default value is 0.
    ```
    $ python ada_boost.py TRAIN_DATA TEST_DATA --valid_split 0.2
    ```

- -lr or --learning_rate: This argument specifies the learning rate of the
model. The default value is 1. For example:
    ```
    $ python ada_boost.py TRAIN_DATA TEST_DATA -lr 0.5
    ```

- -t or --tree_depth: This argument specifies the maximum depth of the trees.
The default value is 15. For example:
    ```
    $ python ada_boost.py TRAIN_DATA TEST_DATA -t 5
    ```

## Acknowledgments
This project, developed by Amin Darabi and Seyed Armin Hosseini based on
the second Kaggle competition of the IFT6390 course at the University of
Montreal.
The data and the problem description can be found on the [Kaggle page](https://www.kaggle.com/competitions/ascii-sign-language).
The code is mainly inspired by the course's lectures and tutorials, with some
additional references.
References, algorithm details, and methodologies are found in the report.