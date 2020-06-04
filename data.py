from pathlib import Path

import torch
from torch.utils.data.dataset import random_split
import torchvision.transforms as T
from torchvision import datasets as D


class PyTorchDatasets:
    def __init__(self, train_split=.8, seed=23,):
        self.train_split = .8

    def __call__(self, dataset='MNIST'):
        return getattr(self, dataset.lower())()

    def _nb_train_val(self, _len):
        nb_train = int(self.train_split * _len)
        nb_valid = _len - nb_train
        return nb_train, nb_valid

    def mnist(self,):
        torch.manual_seed(self.seed)
        train_valid_dataset = D.MNIST(
            root=Path("~/datasets/MNIST").expanduser(),
            train=True,
            transform=T.ToTensor(),
            download=True,)
        nb_train, nb_valid = self._nb_train_val(len(train_valid_dataset))
        return random_split(train_valid_dataset, [nb_train, nb_valid])

    def cifar10(self,):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_valid_dataset = D.CIFAR10(
            root=Path("~/datasets/CIFAR10").expanduser(),
            train=True,
            transform=transform,
            download=True,)
        nb_train, nb_valid = self._nb_train_val(len(train_valid_dataset))
        return random_split(train_valid_dataset, [nb_train, nb_valid])

    def cifar100(self,):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_valid_dataset = D.CIFAR100(
            "/prj/neo_lv/user/ybhalgat/LSQ-KD/",
            train=True,
            transform=transform,
            download=True,)
        nb_train, nb_valid = self._nb_train_val(len(train_valid_dataset))
        return random_split(train_valid_dataset, [nb_train, nb_valid])


def get_datasets(dataset, train_split=.8, seed=23):
    return PyTorchDatasets(train_split, seed)(dataset)
