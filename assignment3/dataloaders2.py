from torchvision import transforms, datasets
from torchvision.transforms import v2
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import ConcatDataset
import torch
import typing
import numpy as np
import pathlib
np.random.seed(0)

# Given:
# mean = (0.5, 0.5, 0.5)
# std = (.25, .25, .25)

# CIFAR-10 actual:
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

# Task 4:
# mean = (0.485, 0.456, 0.406)
# std = (0.229, 0.224, 0.225)

def get_data_dir():
    server_dir = pathlib.Path("/work/datasets/cifar10")
    if server_dir.is_dir():
        return str(server_dir)
    return "data/cifar10"


def load_cifar10(batch_size: int, validation_fraction: float = 0.1, shrink: bool = False
                 ) -> typing.List[torch.utils.data.DataLoader]:
    # Note that transform train will apply the same transform for
    # validation!

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_train_augmented = transforms.Compose([
        # transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0), ratio=(1, 1), antialias=True),
        transforms.RandomHorizontalFlip(1.0), 
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    data_train = datasets.CIFAR10(get_data_dir(),
                                  train=True,
                                  download=True,
                                  transform=transform_train)
    
    data_train_augmented = datasets.CIFAR10(get_data_dir(),
                                  train=True,
                                  download=True,
                                  transform=transform_train_augmented)

    data_test = datasets.CIFAR10(get_data_dir(),
                                 train=False,
                                 download=True,
                                 transform=transform_test)
    
    # Combining original and augmented training samples. Total n_samples = 2 * 50 000
    data_train_combined = ConcatDataset([data_train, data_train_augmented])

    indices_original = list(range(len(data_train)))
    indices = list(range(len(data_train_combined)))
    # print("Number of indices: ", len(indices))
    split_idx = int(np.floor(validation_fraction * len(data_train)))

    # Validation indices are only pulled from the original 50 000
    val_indices = np.random.choice(indices_original, size=split_idx, replace=False)
    train_indices = list(set(indices) - set(val_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    dataloader_train = torch.utils.data.DataLoader(data_train_combined,
                                                   sampler=train_sampler,
                                                   batch_size=batch_size,
                                                   num_workers=2,
                                                   drop_last=True)

    dataloader_val = torch.utils.data.DataLoader(data_train,
                                                 sampler=validation_sampler,
                                                 batch_size=batch_size,
                                                 num_workers=2)

    dataloader_test = torch.utils.data.DataLoader(data_test,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2)

    return dataloader_train, dataloader_val, dataloader_test
