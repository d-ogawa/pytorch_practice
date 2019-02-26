import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from skimage import io
import numpy as np


class myDataset(Dataset):
    """
    Dataset for Handmade dataset
    """

    def __init__(self, csv_file_path, transform=None):

        self.dataframe = pd.read_csv(csv_file_path)
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return (len(self.dataframe))

    def __getitem__(self, idx):

        label = self.dataframe.at[idx, "label"]
        # label = np.array([self.dataframe.at[idx, "label"]])

        img_path = self.dataframe.at[idx, "path_to_image"]

        img = io.imread(img_path)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)

        if self.transform:
            img = self.transform(img)

        return (img, label)



if __name__ == "__main__":

    # datasets
    dataset = datasets.MNIST("../data",
                   train=True,
                   download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307, ), (0.3081, ))
                   ]))
    for (i, sample) in enumerate(dataset):
        print(sample[0].shape, sample[1])

        if i == 10:
            break

    train_loader = DataLoader(
                    datasets.MNIST("../data",
                                   train=True,
                                   download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307, ), (0.3081, ))
                                   ])),
                    batch_size=10,
                    shuffle=False,
                    )

    for i, (image, label) in enumerate(train_loader):
        print(i, image.shape, label)
        if i == 10:
            break


    # my dataset
    dataset = myDataset(csv_file_path="../data/mnist_train.csv", #),
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307, ), (0.3081, ))
                        ]))
    for i, (image, label) in enumerate(dataset):
        print(image.shape, label)

        if i == 10:
            break

    train_loader = DataLoader(
                        myDataset(csv_file_path="../data/mnist_train.csv",
                                  transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307, ), (0.3081, ))
                                    ])),
                        batch_size=10,
                        shuffle=False,
                        )
    for i, (image, label) in enumerate(train_loader):
        print(i, image.shape, label)
        if i == 10:
            break
