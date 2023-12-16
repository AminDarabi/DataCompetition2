# This project is developed by Amin Darabi, and Seyed Armin Hosseini for
# second data competition of the course IFT6390: Fundamentals of Machine
# Learning at University of Montreal.
'''
This module contains implementation of an ada boosting using decision trees
for sign language classification.
'''

import argparse

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def read_sign_mnist_train(
        file: str = 'Data/sign_mnist_train.csv',
        shuffle: bool = True,
        valid_split: float = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    read the sign language train set from csv file.

    Parameters
    ----------
    file : str
        the path to the csv file.
    shuffle : bool
        shuffle the data.
    valid_split : float
        the ratio of the validation data.
    augment : bool
        augment training data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]
        images and labels of the train set and validation set.
        return None for validation data if valid_split is 0.
        shape of images is (n, 28, 28)
        shape of labels is (n,)
        shape of validation images and labels is (m, 28, 28) and (m,)
    """

    df = pd.read_csv(file)

    labels = np.array(df['label'].values).astype(np.int8)
    images = np.array(
        df.drop('label', axis=1).values.reshape(-1, 28, 28)
    ).astype(np.float32)

    if shuffle:
        indices = np.arange(0, len(labels))
        np.random.shuffle(indices)

        labels = labels[indices]
        images = images[indices]

    if valid_split != 0:

        split_index = int(len(labels) * valid_split)

        train_images = images[split_index:]
        train_labels = labels[split_index:]
        valid_images = images[:split_index]
        valid_labels = labels[:split_index]

    else:

        train_images = images
        train_labels = labels
        valid_images = None
        valid_labels = None

    return (
        train_images,
        train_labels,
        valid_images,
        valid_labels
    )


def read_sign_mnist_test(
        file: str = 'Data/test.csv'
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the sign language test set from csv file.

    Parameters
    ----------
    file : str
        the path to the csv file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        images a adn b of the test set.
        shape of images is (n, 28, 28)
        shape of labels is (n,)
    """

    df = pd.read_csv(file)

    images = np.array(
        df.drop('id', axis=1).values.reshape(-1, 2, 28, 28)
    ).astype(np.float32)
    images_a = images[:, 0, :, :]
    images_b = images[:, 1, :, :]

    return images_a, images_b


def ascii_task_output(
        predict_a: np.ndarray,
        predict_b: np.ndarray,
        file: str = 'result/output.csv'
) -> pd.DataFrame:
    """
    Create the ascii output for the test.

    Parameters
    ----------
    predict_a : torch.Tensor
        the predicted classes of the first test images.
        shape is (3000,)
    predict_b : torch.Tensor
        the predicted classes of the second test images.
        shape is (3000,)
    file : str
        the path to the output file.

    Returns
    -------
    pd.DataFrame
        the output dataframe.
    """

    predict_a = np.array(predict_a)
    predict_b = np.array(predict_b)

    predict = predict_a + predict_b + 65

    df = pd.DataFrame(
        data={
            'id': np.arange(0, 3000),
            'label': [chr(x) for x in predict]
        }
    )

    df.to_csv(file, header=True, index=False)
    return df


def pre_process(
        images: np.ndarray
) -> np.ndarray:
    """
    Pre process the images.

    Parameters
    ----------
    images : np.ndarray
        the images to pre process.
        shape is (n, 28, 28)

    Returns
    -------
    np.ndarray
        the pre processed images.
        shape is (n, 7 * 6 * 6)
    """

    filters = np.array([
        [  # Vertical Edge Detection
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ],
        [  # Horizontal Edge Detection
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ],
        [  # Blur
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9]
        ],
        [  # Sharpen
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ],
        [  # digonal
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ],
        [  # digonal Inverse
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ],
        [  # Gaussian Blur
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625]
        ]
    ])

    images = images[:, None, :, :]

    # Apply filters
    filtered_images = np.zeros((images.shape[0], 7, 26, 26))

    for i in range(26):
        for j in range(26):
            filtered_images[:, :, i, j] = np.sum(
                images[:, :, i:i + 3, j:j + 3] * filters,
                axis=(2, 3)
            )

    # Max Pooling
    images = np.zeros((images.shape[0], 7, 6, 6))

    for i in range(6):
        for j in range(6):
            images[:, :, i, j] = np.max(
                filtered_images[:, :, i * 4:i * 4 + 6, j * 4:j * 4 + 6],
                axis=(2, 3)
            )

    return images.reshape(-1, 7 * 36)


def main(args: argparse.Namespace):
    """
    Main function.

    Parameters
    ----------
    args : argparse.Namespace
        the arguments of the program.
    """

    print(
        "Program is running and reading the data."
    )

    if args.valid_data is None:
        train_data, train_labels, valid_data, valid_labels =\
            read_sign_mnist_train(
                args.train_data,
                args.shuffle,
                args.valid_split
            )
        train_data = pre_process(train_data)
        if valid_data is not None:
            valid_data = pre_process(valid_data)
    else:
        valid_data, valid_labels, _, _ = read_sign_mnist_train(
            args.valid_data,
            args.shuffle,
            0
        )
        valid_data = pre_process(valid_data)
        train_data, train_labels, _, _ = read_sign_mnist_train(
            args.train_data,
            args.shuffle,
            0
        )
        train_data = pre_process(train_data)

    print("Program is training the model.")

    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=args.tree_depth
        ),
        n_estimators=args.nestimators,
        learning_rate=args.learning_rate
    ).fit(
        train_data,
        train_labels
    )

    if args.print_loss or valid_data is not None:
        print(
            f"Training accuracy: "
            f"{model.score(train_data, train_labels)}"
        )

    if valid_data is not None:
        print(
            f"Validation accuracy: "
            f"{model.score(valid_data, valid_labels)}"
        )

    print(
        "Program is predicting the test data, writing results "
        f"to '{args.output}'."
    )

    test_data_a, test_data_b = read_sign_mnist_test(args.test_data)
    test_data_a = pre_process(test_data_a)
    test_data_b = pre_process(test_data_b)

    predict_a = model.predict(test_data_a)
    predict_b = model.predict(test_data_b)

    ascii_task_output(predict_a, predict_b, args.output)
    exit(0)


if __name__ == '__main__':
    # Parse the arguments and call the main function.

    parser = argparse.ArgumentParser(
        prog='AdaBoost Classifier using Decision Trees for Sign Language ',
        description="This project developed by Amin Darabi and Seyed Armin "
                    "Hossieni for the second data competition of the "
                    "Introduction to Machine Learning course(IFT6390_UdeM). "
                    "This program trains a AdaBoost Classifier using Decision "
                    "Trees to classify the sign language images. "
                    "After training the model, it predicts the test data and "
                    "performs and ascii trick to create the output file."
    )

    parser.add_argument(
        'train_data', type=str,
        help='path to the train data.'
    )
    parser.add_argument(
        'test_data', type=str,
        help='path to the test data.'
    )
    parser.add_argument(
        '-o', '--output', type=str, default='output.csv',
        help='path to the output file.'
    )
    parser.add_argument(
        '--valid_data', type=str, default=None,
        help='path to the validation data.'
    )
    parser.add_argument(
        '-n', '--nestimators', type=int, default=30,
        help='number of estimators for the model.'
    )
    parser.add_argument(
        '--shuffle', type=bool, default=True,
        help='shuffle the training data.'
    )
    parser.add_argument(
        '-p', '--print_loss', action='store_true',
        help='print the loss of the model.'
    )
    parser.add_argument(
        '--valid_split', type=float, default=0,
        help='split the train data into train and validation data(0 for None).'
    )
    parser.add_argument(
        '-t', '--tree_depth', type=int, default=15,
        help='the depth of the decision tree.'
    )
    parser.add_argument(
        '-lr', '--learning_rate', type=float, default=1,
        help='the learning rate of the model.'
    )

    main(parser.parse_args())
