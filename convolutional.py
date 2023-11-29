'''

'''

import torch
import pandas as pd
import numpy as np


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


class Convolutional(torch.nn.Module):
    """
    simple convolutional neural network for sign language classification.

    """

    def __init__(
            self,
            optimizer: str = 'adam',
            loss_function: str = 'cross_entropy'
    ):
        """
        initialize the convolutional neural network.
        """

        super().__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(3, 3),
            padding='same'
        )
        self.act1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = torch.nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            padding='same'
        )
        self.act2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(7, 7),
            padding='valid'
        )
        self.act3 = torch.nn.ReLU()

        self.flat = torch.nn.Flatten()
        self.mlp = torch.nn.Linear(
            in_features=128,
            out_features=64
        )
        self.act4 = torch.nn.ReLU()

        self.out = torch.nn.Linear(
            in_features=64,
            out_features=26
        )

        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        else:
            raise ValueError('optimizer must be adam or sgd.')

        if loss_function == 'cross_entropy':
            self.loss_function = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError('loss_function must be cross_entropy.')

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        forward pass of the convolutional neural network.
        """

        data = self.conv1(data)
        data = self.act1(data)
        data = self.pool1(data)

        data = self.conv2(data)
        data = self.act2(data)
        data = self.pool2(data)

        data = self.conv3(data)
        data = self.act3(data)

        data = self.flat(data)
        data = self.mlp(data)
        data = self.act4(data)

        data = self.out(data)

        return data

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """
        predict the class of the given data.
        """

        with torch.no_grad():
            return self.forward(data).argmax(dim=1)

    def fit(
            self,
            data: torch.Tensor | np.ndarray,
            labels: torch.Tensor | np.ndarray,
            epochs: int = 10,
            batch_size: int = 64,
            shuffle: bool = True,
            print_loss: bool = True
    ) -> 'Convolutional':
        """
        fit the convolutional neural network to the given data.
        """

        data = torch.Tensor(data).unsqueeze(1)
        labels = torch.Tensor(labels).long()

        data_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(data, labels),
            batch_size=batch_size,
            shuffle=shuffle
        )

        for epoch in range(epochs):
            for batch, (data, labels) in enumerate(data_loader):

                self.optimizer.zero_grad()

                predict = self(data)
                loss = self.loss_function(predict, labels)

                loss.backward()
                self.optimizer.step()

                if print_loss and batch % 100 == 0:
                    print(
                        f'Epoch: {epoch}, Batch: {batch}, Loss: {loss.item()}'
                    )

        return self


def main():
    """
    Main function.
    """

    train_data, train_labels = read_sing_mnist_train()
    model = Convolutional().fit(train_data, train_labels)

    test_data_a, test_data_b = read_sing_mnist_test()
    predict_a = model.predict(torch.Tensor(test_data_a).unsqueeze(1))
    predict_b = model.predict(torch.Tensor(test_data_b).unsqueeze(1))

    ascii_task_output(predict_a, predict_b)
    exit(0)


if __name__ == '__main__':

    main()