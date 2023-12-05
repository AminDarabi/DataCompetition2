# This project is developed by Amin Darabi, and Seyed Armin Hosseini for
# second data competition of the course IFT6390: Fundamentals of Machine
# Learning at University of Montreal.
"""
This module contains the implementation of Random Forest Classifier.
Applying on the Sign Language Dataset.
"""

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
from randomfores_class.randomforest import RandomForestClassifier
# from sklearn.ensemble import RandomForestClassifier


def filter_images(images: np.ndarray, filter_size: np.float32 = 4) -> np.ndarray:
    filter_height = int(images.shape[1]/filter_size)
    reshaped_images = images.reshape(images.shape[0], filter_height, filter_size, filter_height, filter_size)
    filtered_images = np.sum(reshaped_images, axis=(2, 4))
    return filtered_images.reshape(filtered_images.shape[0], -1) 

def convolve(images, kernel):
    image_count, image_height, image_width = images.shape
    images = images / 255
    kernel_height, kernel_width = kernel.shape
    new_image = np.zeros((image_count, image_height - kernel_height + 1, image_width - kernel_width + 1))

    for i in range(image_height - kernel_height + 1):
        for j in range(image_width - kernel_width + 1):
            # Extract the region of the image that corresponds to the current position of the kernel
            new_image[:, i, j] = np.sum(
                images[:, i:i+kernel_height, j:j+kernel_width] * kernel, axis=(1, 2)
            )

    return new_image


def read_sign_mnist_train(
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

    # images = np.array(df.drop('label', axis=1).values).astype(np.float32)

    return images, labels


def read_sign_mnist_test(
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

Kernels = {
    'vertical': np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ]),
    'horizontal': np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ]),
    'box_blur': np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ]),
    'sharpen': np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]

    ]),
    'diag1': np.array([
        [ 1,  0,  0],
        [ 0,  1,  0],
        [ 0,  0,  1]
    ]),
    'diag2': np.array([
        [ 0,  0,  1],
        [ 0,  1,  0],
        [ 1,  0,  0]
    ]),
    'gaussian_blur': np.array([
        [1/16, 1/8, 1/16],
        [1/8, 1/4, 1/8],
        [1/16, 1/8, 1/16]
    ])}




if __name__ == "__main__":
    images, labels = read_sign_mnist_train()
    test_images, test_labels = read_sign_mnist_train(file='Data/sign_mnist_test.csv')
    # filtered_train_images = filter_images(images)
    # filtered_test_images = filter_images(test_images)
    kernel_height, kernel_width = Kernels['vertical'].shape
    image_with_kernel = np.ndarray((images.shape[0], len(Kernels), images.shape[1] - kernel_height + 1,
                                    images.shape[2] - kernel_height + 1))
    test_image_with_kernel = np.ndarray((test_images.shape[0], len(Kernels), test_images.shape[1] - kernel_height + 1,
                                         test_images.shape[2] - kernel_height + 1))
    for i, kernel in tqdm(enumerate(Kernels), total=len(Kernels)):
        image_with_kernel[:, i, :, :] = convolve(images, Kernels[kernel])
        test_image_with_kernel[:, i, :, :] = convolve(test_images, Kernels[kernel])

    max_pool_image = np.ndarray((images.shape[0], len(Kernels), 6, 6))
    test_max_pool_image = np.ndarray((test_images.shape[0], len(Kernels), 6, 6))
    for i in range(6):
        for j in range(6):
            max_pool_image[:,:,i,j] = np.max(image_with_kernel[:, :, i*4:i*4+6, j*4:j*4+6], axis=(2, 3))
            test_max_pool_image[:,:,i,j] = np.max(test_image_with_kernel[:, :, i*4:i*4+6, j*4:j*4+6], axis=(2, 3))

    image_with_kernel = image_with_kernel.reshape(image_with_kernel.shape[0], -1)
    test_image_with_kernel = test_image_with_kernel.reshape(test_image_with_kernel.shape[0], -1)
    tree_counts = [30, 50, 100, 200, 300]
    depths = [10, 20, 30, 50]
    df = pd.DataFrame(columns=['tree_count', 'depth', 'train_accuracy', 'test_accuracy'])
    print()

    # for tree_count in tree_counts:
    #     for depth in depths:
    rf = RandomForestClassifier(n_estimators=50, max_depth=20)
    rf.fit(image_with_kernel, labels)
    train_predict = rf.predict(image_with_kernel)
    train_accuracy = (train_predict == labels).mean()
    test_predict = rf.predict(test_image_with_kernel)
    test_accuracy = (test_predict == test_labels).mean()
    df = pd.concat([df, pd.DataFrame([{'tree_count': 50, 'depth': 20, 'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy}])], ignore_index=True)
    os.system('cls' if os.name == 'nt' else 'clear')
    print(df)

