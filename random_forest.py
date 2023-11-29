# This project is developed by Amin Darabi, and Seyed Armin Hosseini for
# second data competition of the course IFT6390: Fundamentals of Machine
# Learning at University of Montreal.
"""
This module contains the implementation of Random Forest Classifier.
Applying on the Sign Language Dataset.
"""

import numpy as np
import pandas as pd


def read_sing_mnist_train(
        file: str = 'Data/sign_mnist_train.csv'
) -> tuple[np.ndarray, np.ndarray]:
    """
    read the sign language train set from csv file.

    Parameters
    ----------
    file : str
        the path to the csv file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        the images and labels of the train set.
        shape of images is (27455, 28, 28)
        shape of labels is (27455,)
    """

    df = pd.read_csv(file)

    labels = np.array(df['label'].values).astype(np.int8)
    images = np.array(
        df.drop('label', axis=1).values.reshape(-1, 28, 28)
    ).astype(np.float32)

    return images, labels


def read_sing_mnist_test(
        file: str = 'Data/test.csv'
) -> tuple[np.ndarray, np.ndarray]:
    """
    read the sign language test set from csv file.

    Parameters
    ----------
    file : str
        the path to the csv file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        images a adn b of the test set.
        shape of images is (3000, 28, 28)
        shape of labels is (3000,)
    """

    df = pd.read_csv(file)

    images = np.array(
        df.drop('id', axis=1).values.reshape(-1, 2, 28, 28)
    ).astype(np.float32)
    images_a = images[:, 0, :, :]
    images_b = images[:, 1, :, :]

    return images_a, images_b


def ascii_task_output(
        predict_a: torch.Tensor | np.ndarray,
        predict_b: torch.Tensor | np.ndarray,
        file: str = 'result/output.csv'
) -> pd.DataFrame:
    """
    create the ascii output for the test.

    Parameters
    ----------
    predict_a : torch.Tensor
        the predicted classes of the first test images.
        shape is (3000,)
    predict_b : torch.Tensor
        the predicted classes of the second test images.
        shape is (3000,)

    Returns
    -------
    str
        the ascii output for the task.
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
