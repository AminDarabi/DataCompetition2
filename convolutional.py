# This project is developed by Amin Darabi, and Seyed Armin Hosseini for
# second data competition of the course IFT6390: Fundamentals of Machine
# Learning at University of Montreal.
'''
This module contains implementation of a convolutional neural network for
sign language classification.
'''

import argparse

import torch
import pandas as pd
import numpy as np
from scipy import ndimage


def read_sign_mnist_train(
        file: str = 'Data/sign_mnist_train.csv',
        shuffle: bool = True,
        valid_split: float = 0,
        augment: bool = False
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

    if augment:

        # Add rotated images
        rotated_images = np.array(
            [ndimage.rotate(
                img, np.random.randint(-20, 21), reshape=False
            ) for img in train_images]
        )

        # Add shifted images
        shifted_images = np.array(
            [
                ndimage.shift(img, np.random.randint(-3, 4, [2]))
                for img in train_images
            ]
        )

        train_labels = np.concatenate(
            [train_labels, train_labels, train_labels]
        )
        train_images = np.concatenate(
            [train_images, rotated_images, shifted_images]
        )

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
        predict_a: torch.Tensor | np.ndarray,
        predict_b: torch.Tensor | np.ndarray,
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


class Convolutional:
    """
    Simple convolutional neural network for sign language classification.

    Attributes
    ----------
    model : torch.nn.Module
        the convolutional neural network model.
    optimizer : torch.optim.Adam | torch.optim.SGD
        the optimizer for the model.
    loss_function : torch.nn.CrossEntropyLoss
        the loss function for the model.
    device : str
        the device for the model. (auto, cuda, mps, cpu)

    Methods
    -------
    predict(data: torch.Tensor | np.ndarray) -> np.ndarray
        predict the class of the given data.
    score(
        data: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray
    ) -> tuple[float, float]
        calculate the accuracy and loss of the given data.
    fit(
        data: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
        valid_data: torch.Tensor | np.ndarray | None = None,
        valid_labels: torch.Tensor | np.ndarray | None = None,
        epochs: int = 4,
        batch_size: int = 32,
        print_loss: bool = True
    ) -> 'Convolutional'
        fit the convolutional neural network to the given data.
    """

    def __init__(
            self,
            output_classes: int = 26,
            input_channels: int = 1,
            device: str = 'auto',
            optimizer: str = 'adam',
            loss_function: str = 'cross_entropy'
    ):
        """
        Initialize the convolutional neural network.

        Parameters
        ----------
        output_classes : int
            number of output classes.
        input_channels : int
            number of input channels.
        device : str
            device for the model. (auto, cuda, mps, cpu)
        optimizer : str
            optimizer for the model. (adam, sgd)
        loss_function : str
            loss function for the model. (cross_entropy)

        Raises
        ------
        ValueError
            if the device is not available.
        ValueError
            if the optimizer is not supported.
        ValueError
            if the loss function is not supported.
        """

        if device == 'auto':
            self.device = (
                'cuda' if torch.cuda.is_available() else
                'mps' if torch.backends.mps.is_available() else
                'cpu'
            )
        else:
            self.device = device

        self.model = self.CNN(
            input_channels,
            output_classes
        ).to(self.device)

        self.optimizer: torch.optim.Adam | torch.optim.SGD
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=0.0001,
            )
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=0.0001
            )
        else:
            raise ValueError('optimizer must be adam or sgd.')

        if loss_function == 'cross_entropy':
            self.loss_function = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError('loss_function must be cross_entropy.')

    def _handle_input_data(
            self, data: torch.Tensor | np.ndarray
    ) -> torch.Tensor:
        # This function handles the input data.
        return torch.Tensor(data).unsqueeze(1).to(self.device)

    def _handle_ouput_labels(
            self, labels: torch.Tensor | np.ndarray
    ) -> torch.Tensor:
        # This function handles the output labels.
        return torch.Tensor(labels).long().to(self.device)

    def predict(self, data: torch.Tensor | np.ndarray) -> np.ndarray:
        """
        predict the class of the given data.

        Parameters
        ----------
        data : torch.Tensor | np.ndarray
            the data to predict its class.

        Returns
        -------
        np.ndarray
            the predicted classes.
        """

        data = self._handle_input_data(data)
        with torch.no_grad():
            return self.model.forward(data).argmax(dim=1).to('cpu').numpy()

    def score(
            self,
            data: torch.Tensor | np.ndarray,
            labels: torch.Tensor | np.ndarray
    ) -> tuple[float, float]:
        """
        Calculate the accuracy and loss of the given data.

        Parameters
        ----------
        data : torch.Tensor | np.ndarray
            the data to calculate its accuracy and loss.
        labels : torch.Tensor | np.ndarray
            the labels of the given data.

        Returns
        -------
        tuple[float, float]
            the accuracy and loss of the given data.
        """

        data = self._handle_input_data(data)
        labels = self._handle_ouput_labels(labels)

        with torch.no_grad():
            predict = self.model.forward(data)
            loss = self.loss_function(predict, labels)
            accuracy = (predict.argmax(dim=1) == labels).float().mean()

        return (
            round(accuracy.item(), 4),
            round(loss.item(), 4)
        )

    def fit(
            self,
            data: torch.Tensor | np.ndarray,
            labels: torch.Tensor | np.ndarray,
            valid_data: torch.Tensor | np.ndarray | None = None,
            valid_labels: torch.Tensor | np.ndarray | None = None,
            epochs: int = 4,
            batch_size: int = 32,
            print_loss: bool = True
    ) -> 'Convolutional':
        """
        Fit the convolutional neural network to the given data.

        Parameters
        ----------
        data : torch.Tensor | np.ndarray
            the data to fit the model.
        labels : torch.Tensor | np.ndarray
            the labels of the given data.
        valid_data : torch.Tensor | np.ndarray | None
            the validation data.
        valid_labels : torch.Tensor | np.ndarray | None
            the labels of the validation data.
        epochs : int
            number of epochs to train the model.
        batch_size : int
            batch size for training the model.
        print_loss : bool
            print the loss of the model.

        Returns
        -------
        Convolutional
            the fitted model.
        """

        data_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(
                self._handle_input_data(data),
                self._handle_ouput_labels(labels)
            ),
            batch_size=batch_size
        )

        for epoch in range(epochs):
            for _, (data_x, data_y) in enumerate(data_loader):

                self.optimizer.zero_grad()

                predict = self.model.forward(data_x)
                loss = self.loss_function(predict, data_y)

                loss.backward()
                self.optimizer.step()

            if print_loss:
                print(
                    f'Epoch: {epoch + 1}, Training accuracy and loss: '
                    f'{self.score(data, labels)}'
                )
            if valid_data is not None and valid_labels is not None:
                print(
                    f'Epoch: {epoch + 1}, Validation accuracy and loss: '
                    f'{self.score(valid_data, valid_labels)}'
                )

        return self

    class CNN(torch.nn.Module):
        """
        torch CNN module.
        """

        def __init__(
                self,
                input_channels: int,
                output_classes: int,
        ):
            """
            Initialize the convolutional neural network.

            Parameters
            ----------
            input_channels : int
                number of input channels.
            output_classes : int
                number of output classes.
            """

            super().__init__()

            self.conv1 = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=20,
                    kernel_size=(3, 3),
                    padding='same'
                ),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=(2, 2)),
                torch.nn.Dropout(0.05)
            )
            self.conv2 = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=20,
                    out_channels=32,
                    kernel_size=(3, 3),
                    padding='same'
                ),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=(2, 2)),
                torch.nn.Dropout(0.1)
            )
            self.conv3 = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=32,
                    out_channels=20,
                    kernel_size=(2, 2),
                    padding='valid'
                ),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=(2, 2)),
                torch.nn.Dropout(0.05)
            )
            self.linear = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(
                    in_features=180,
                    out_features=output_classes
                )
            )

        def forward(self, data: torch.Tensor) -> torch.Tensor:
            """
            Forward pass of the convolutional neural network.

            Parameters
            ----------
            data : torch.Tensor
                the input data.

            Returns
            -------
            torch.Tensor
                the output of the model.
            """

            data = self.conv1(data)
            data = self.conv2(data)
            data = self.conv3(data)

            data = self.linear(data)
            return data


def main(args: argparse.Namespace):
    """
    Main function.

    Parameters
    ----------
    args : argparse.Namespace
        the arguments of the program.
    """

    print(
        "Program is running and reading the data with"
        f"{args.without_augment * 'out'} augmentation."
    )

    if args.valid_data is None:
        train_data, train_labels, valid_data, valid_labels =\
            read_sign_mnist_train(
                args.train_data,
                args.shuffle,
                args.valid_split,
                not args.without_augment
            )
    else:
        valid_data, valid_labels, _, _ = read_sign_mnist_train(
            args.valid_data,
            args.shuffle,
            0
        )
        train_data, train_labels, _, _ = read_sign_mnist_train(
            args.train_data,
            args.shuffle,
            0,
            not args.without_augment
        )

    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('mps is not available on this machine.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('cuda is not available on this machine.')

    if args.print_loss or args.device == 'auto':
        device = (
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else
            'cpu'
        )
        print(f"Program is running on {device}.")

    model = Convolutional(
        device=args.device,
        optimizer=args.optimizer
    ).fit(
        train_data, train_labels,
        valid_data, valid_labels,
        args.epochs, args.batch_size,
        args.print_loss
    )

    print(
        "Program is predicting the test data, writing results "
        f"'to {args.output}'."
    )

    test_data_a, test_data_b = read_sign_mnist_test(args.test_data)
    predict_a = model.predict(torch.Tensor(test_data_a))
    predict_b = model.predict(torch.Tensor(test_data_b))

    ascii_task_output(predict_a, predict_b, args.output)
    exit(0)


if __name__ == '__main__':
    # Parse the arguments and call the main function.

    parser = argparse.ArgumentParser(
        prog='Convolutional Neural Network for Sign Language',
        description="This project developed by Amin Darabi and Seyed Armin "
                    "Hossieni for the second data competition of the "
                    "Introduction to Machine Learning course(IFT6390_UdeM). "
                    "This program trains a convolutional neural network "
                    "to classify sign language MNIST."
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
        '-e', '--epochs', type=int, default=25,
        help='number of epochs to train the model.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='batch size for training the model.'
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
        '--optimizer', type=str, default='adam',
        help='optimizer for the model.'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='device for the model. (auto, cuda, mps, cpu)'
    )
    parser.add_argument(
        '--valid_split', type=float, default=0,
        help='split the train data into train and validation data(0 for None).'
    )
    parser.add_argument(
        '--without_augment', action='store_true',
        help='do not augment the train data.'
    )

    main(parser.parse_args())
