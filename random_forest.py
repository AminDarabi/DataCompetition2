# This project is developed by Amin Darabi, and Seyed Armin Hosseini for
# second data competition of the course IFT6390: Fundamentals of Machine
# Learning at University of Montreal.
'''
This module contains the implementation of a Random Forest Classifier.
Applying on the Sign Language Dataset.
'''

import argparse

import pandas as pd
import numpy as np


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


class RandomForestClassifier:
    """
    A random forest classifier.

    """

    def __init__(
            self,
            n_estimators: int = 10,
            max_depth: int = 10,
            max_samples: int | float = 0.5,
            min_samples: int = 5,
            quantiles: np.ndarray | list[float] | None = None,
            n_features: int | float | str = 'sqrt'
    ):
        """
        Initialize a random forest classifier.

        Parameters
        ----------
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of each tree.
        max_samples : int | float
            The number of samples to use for each tree.
        n_features : int | float | str
            The number of features to use for each tree.
            could be 'sqrt',  'log2', or 'all'

        Raises
        ------
        ValueError
            If n_features is not 'sqrt', 'log2', 'all', or an int or float.

        """

        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.max_samples = max_samples

        if (
            isinstance(n_features, str) and
            n_features not in ('sqrt', 'log2', 'all')
        ):
            raise ValueError(
                "n_features must be 'sqrt', 'log2', or 'all', not "
                f"{n_features}"
            )
        elif not isinstance(n_features, (int, float, str)):
            raise ValueError(
                f"n_features must be int, float, or str, not {n_features}"
            )
        self.n_features = n_features

        self.estimators = [
            DecisionTreeClassifier(
                self.max_depth,
                min_samples,
                quantiles
            )
            for _ in range(self.n_estimators)
        ]
        self.features: list[np.ndarray] = []

    def _calculate_n_features(
            self,
            n_cols: int
    ) -> int:
        # This function caculates the number of features to use for each tree.

        if self.n_features == 'sqrt':
            return max(1, int(np.sqrt(n_cols)))
        elif self.n_features == 'log2':
            return max(1, int(np.log2(n_cols)))
        elif self.n_features == 'all':
            return n_cols
        elif isinstance(self.n_features, int):
            return max(1, min(self.n_features, n_cols))
        return max(1, min(int(n_cols * self.n_features), n_cols))

    def _calculate_n_samples(
            self,
            n_rows: int
    ) -> int:
        # This function caculates the number of samples to use for each tree.

        if isinstance(self.max_samples, int):
            return max(1, min(self.max_samples, n_rows))
        return max(1, min(int(n_rows * self.max_samples), n_rows))

    def fit(
            self,
            data: np.ndarray,
            labels: np.ndarray,
            print_progress: bool = False
    ) -> 'RandomForestClassifier':
        """
        Fit the random forest classifier.

        Parameters
        ----------
        data : np.ndarray
            The data to fit the model to.
        labels : np.ndarray
            The labels for the data.
        print_progress : bool
            Whether to print progress or not.

        Returns
        -------
        RandomForestClassifier
            The fitted model.
        """

        number_of_samples = self._calculate_n_samples(data.shape[0])
        number_of_features = self._calculate_n_features(data.shape[1])

        if print_progress:
            print(
                f"Training {self.n_estimators} trees on {number_of_samples} "
                f"samples with {number_of_features} features."
            )

        for estimator in range(self.n_estimators):

            samples_index = np.random.choice(
                data.shape[0],
                size=number_of_samples
            )
            features_index = np.random.choice(
                data.shape[1],
                size=number_of_features,
                replace=False
            )

            sample_data = data[samples_index, :][:, features_index]
            sample_labels = labels[samples_index]

            self.estimators[estimator].fit(
                sample_data,
                sample_labels
            )
            self.features.append(features_index)

            if print_progress:
                score = self.estimators[estimator].score(
                    sample_data,
                    sample_labels
                )
                print(
                    f"Estimator {estimator + 1} of {self.n_estimators} fitted "
                    f"with score %.4f." % score
                )

        return self

    def predict(
            self,
            data: np.ndarray
    ) -> np.ndarray:
        """
        Predict the class of each sample in the input array.

        Parameters
        ----------
        data : np.ndarray
            The input array.
            shape is (n_samples, n_features)

        Returns
        -------
        np.ndarray
            The predicted classes.
            shape is (n_samples,)
        """

        predictions = np.array([
            self.estimators[i].predict(data[:, self.features[i]])
            for i in range(self.n_estimators)
        ])
        labels = [i for i in range(np.max(predictions) + 1)]
        return np.argmax(
            np.array([
                predictions == label for label in labels
            ]).sum(axis=1),
            0
        )

    def score(
            self,
            data: np.ndarray,
            labels: np.ndarray
    ) -> float:
        """
        Calculate the accuracy of the model.

        Parameters
        ----------
        data : np.ndarray
            The data to score.
        labels : np.ndarray
            The labels for the data.

        Returns
        -------
        float
            The accuracy of the model.
        """

        predictions = self.predict(data)
        return (predictions == labels).sum() / len(labels)


class DecisionTreeClassifier:
    """
    A decision tree classifier.
    """

    def __init__(
            self,
            max_depth: int = 10,
            min_samples: int = 5,
            quantiles: np.ndarray | list[float] | None = None
    ):
        """
        Initialize a decision tree classifier.

        Parameters
        ----------
        max_depth : int
            The maximum depth of the tree.

        """

        self.max_depth = max_depth
        self.min_samples = min_samples

        self.quantiles = quantiles
        self.root: DecisionTreeClassifier.Node | None = None

    def fit(
            self,
            data: np.ndarray,
            labels: np.ndarray
    ) -> 'DecisionTreeClassifier':
        """
        Fit the decision tree to the input data.

        Parameters
        ----------
        data : np.ndarray
            The input data.
            shape is (n_samples, n_features)
        labels : np.ndarray
            The labels for each sample.
            shape is (n_samples,)

        Returns
        -------
        DecisionTreeClassifier
            The fitted decision tree.
        """

        self.root = DecisionTreeClassifier.Node(
            data,
            labels,
            self.max_depth,
            self.min_samples,
            self.quantiles
        )

        return self

    def predict(
            self,
            data: np.ndarray
    ) -> np.ndarray:
        """
        Predict the class of each sample in the input array.

        Parameters
        ----------
        data : np.ndarray
            The input array.
            shape is (n_samples, n_features)

        Returns
        -------
        np.ndarray
            The predicted classes.
            shape is (n_samples,)

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """

        if self.root is None:
            raise RuntimeError('The model has not been fitted yet.')
        return self.root.predict(data)

    def score(
            self,
            data: np.ndarray,
            labels: np.ndarray
    ) -> float:
        """
        Calculate the accuracy of the model.

        Parameters
        ----------
        data : np.ndarray
            The data to score.
        labels : np.ndarray
            The labels for the data.

        Returns
        -------
        float
            The accuracy of the model.
        """

        predictions = self.predict(data)
        return (predictions == labels).sum() / len(labels)

    class Node:
        """
        A node in a decision tree.
        """

        def __init__(
                self,
                data: np.ndarray,
                labels: np.ndarray,
                remaining_depth: int,
                min_samples_per_node: int = 5,
                quantiles: np.ndarray | list[float] | None = None
        ):
            """
            Initialize a node in a decision tree.
            """

            if quantiles is None and len(labels) > 5:
                quantile = [0.2, 0.4, 0.6, 0.8]
            elif quantiles is None:
                quantile = [i / len(labels) for i in range(1, len(labels))]

            self.output_class: int | None = None

            if remaining_depth == 0 or len(labels) <= min_samples_per_node:
                self.output_class = int(np.argmax(np.bincount(labels)))
                return
            elif len(np.unique(labels)) == 1:
                self.output_class = labels[0]
                return

            questions = np.concatenate([
                np.quantile(data, quantile, 0),
                np.mean(data, 0)[None, :]
            ])
            self.feature, self.threshold = self._find_best_question(
                data, labels, questions
            )

            if (
                DecisionTreeClassifier.Node._compute_entropy_over_question(
                    data, labels, self.feature, self.threshold
                ) >= DecisionTreeClassifier.Node._calculate_entropy(labels)
            ):
                self.output_class = int(np.argmax(np.bincount(labels)))
                return

            left = data[:, self.feature] > self.threshold
            right = np.invert(left)

            self.left_child = DecisionTreeClassifier.Node(
                data[left],
                labels[left],
                remaining_depth - 1,
                min_samples_per_node,
                quantiles
            )
            self.right_child = DecisionTreeClassifier.Node(
                data[right],
                labels[right],
                remaining_depth - 1,
                min_samples_per_node,
                quantiles
            )

        def _find_best_question(
                self,
                data: np.ndarray,
                labels: np.ndarray,
                questions: np.ndarray
        ) -> tuple[int, float]:
            # This function finds the best question to ask at this node.
            # It returns the feature and threshold that minimize the entropy.

            entropies = np.array([
                [
                    DecisionTreeClassifier.Node._compute_entropy_over_question(
                        data, labels, feature, threshold
                    ) for feature, threshold in enumerate(quant)
                ] for quant in questions
            ])

            best = np.unravel_index(np.argmin(entropies), entropies.shape)

            return best[1], questions[best]

        @staticmethod
        def _compute_entropy_over_question(
                data: np.ndarray,
                labels: np.ndarray,
                feature: int,
                threshold: float
        ) -> float:
            # This function computes the entropy of the labels

            left = data[:, feature] > threshold
            right = np.invert(left)
            return (
                    DecisionTreeClassifier.Node._calculate_entropy(
                        labels[left]) * np.sum(left) +
                    DecisionTreeClassifier.Node._calculate_entropy(
                        labels[right]) * np.sum(right)
            ) / len(labels)

        @staticmethod
        def _calculate_entropy(
                labels: np.ndarray
        ) -> float:
            # This function calculates the entropy of the labels
            # entropy = sum_l(-p_l log2 (p_l)) for each label l)

            frequency = np.unique(labels, return_counts=True)[1] / len(labels)
            return -np.sum(frequency * np.log2(frequency))

        def predict(self, data: np.ndarray) -> np.ndarray:
            """
            Predict the class of each sample in the input array.

            Parameters
            ----------
            data : np.ndarray
                The input array.
                shape is (n_samples, n_features)

            Returns
            -------
            np.ndarray
                The predicted classes.
                shape is (n_samples,)
            """

            if self.output_class is not None:
                return np.full(data.shape[0], self.output_class).astype('int8')

            left = data[:, self.feature] > self.threshold
            right = np.invert(left)

            predict = np.zeros(data.shape[0], dtype='int8')
            predict[left] = self.left_child.predict(data[left])
            predict[right] = self.right_child.predict(data[right])

            return predict


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

    if args.features_ratio.isdigit():
        args.features_ratio = float(args.features_ratio)
        if args.features_ratio > 1:
            args.features_ratio = int(args.features_ratio)
    elif args.features_ratio not in ('sqrt', 'log2', 'all'):
        raise ValueError(
            f"n_features must be 'sqrt', 'log2', or 'all', not "
            f"{args.features_ratio}"
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

    model = RandomForestClassifier(
        args.nestimators,
        args.tree_depth,
        min_samples=args.min_samples,
        n_features=args.features_ratio,
        max_samples=args.bootstrap,
    ).fit(
        train_data,
        train_labels,
        args.print_progress
    )

    if args.print_progress or valid_data is not None:
        print(
            f"Training accuracy: "
            f"{model.score(train_data, train_labels)}"
        )

    if valid_data is not None and valid_labels is not None:
        print(
            f"Validation accuracy: "
            f"{model.score(valid_data, valid_labels)}"
        )

    print(
        "Program is predicting the test data, writing results "
        f"to '{args.output}'."
    )

    test_data_a, test_data_b = read_sign_mnist_test(args.test_data)
    predict_a = model.predict(
        pre_process(test_data_a)
    )
    predict_b = model.predict(
        pre_process(test_data_b)
    )

    ascii_task_output(predict_a, predict_b, args.output)
    exit(0)


if __name__ == '__main__':
    # Parse the arguments and call the main function.

    parser = argparse.ArgumentParser(
        prog='Random Forest Classifier for Sign Language Dataset',
        description="This project developed by Amin Darabi and Seyed Armin "
                    "Hossieni for the second data competition of the "
                    "Introduction to Machine Learning course(IFT6390_UdeM). "
                    "This program trains a random forest classifier "
                    "to classify sign language MNIST. "
                    "Random Forest and Decision Tree are implemented from "
                    "scratch using only numpy. "
                    "The program reads the train data and test data then "
                    "applies some pre processing on the data. "
                    "After that, program trains the model and predicts the "
                    "test data. At the end, program performs an ascii trick "
                    "to create the output file."
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
        '-n', '--nestimators', type=int, default=300,
        help='number of estimators for the model.'
    )
    parser.add_argument(
        '--shuffle', type=bool, default=True,
        help='shuffle the training data.'
    )
    parser.add_argument(
        '-p', '--print_progress', action='store_true',
        help='print the progress of the training.'
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
        '-b', '--bootstrap', type=float, default=0.3,
        help='the ratio of the bootstrap.'
    )
    parser.add_argument(
        '-f', '--features_ratio', type=str, default='sqrt',
        help=(
            'the ratio of the features to use for each tree. '
            'could be sqrt, log2, all, or a float number between 0 and 1 '
            'or an integer.'
        )
    )
    parser.add_argument(
        '-m', '--min_samples', type=int, default=1,
        help='the minimum number of samples to use for each tree.'
    )

    main(parser.parse_args())
