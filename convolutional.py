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


class Convolutional(torch.nn.Module):
    """
    """

    def __init__(self):
        """
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

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
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


def main():
    """
    """

    images, labels = read_sing_mnist_train()

    images = torch.Tensor(images).unsqueeze(1)
    labels = torch.Tensor(labels).long()

    data_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(images, labels),
        batch_size=32,
        shuffle=True
    )

    model = Convolutional()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range(10):

        for batch, (images, labels) in enumerate(data_loader):

            optimizer.zero_grad()

            predictions = model(images)
            loss = loss_function(predictions, labels)

            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch}, Loss: {loss.item()}')


if __name__ == '__main__':
    """
    """

    main()