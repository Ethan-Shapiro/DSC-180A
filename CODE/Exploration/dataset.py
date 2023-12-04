import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm
import os


def one_hot_data(dataset: Dataset, num_classes: int, num_samples: int = -1, shift_label: bool = False) -> list:
    """
    Converts dataset labels to one-hot encoded format.

    Parameters:
    dataset (Dataset): The dataset to be processed.
    num_classes (int): The number of classes for one-hot encoding.
    num_samples (int): Number of samples to process (-1 for all samples).
    shift_label (bool): Whether to shift the label values (for zero-indexing).

    Returns:
    list: A list of tuples containing flattened images and one-hot encoded labels.
    """
    labelset = {}
    for i in range(num_classes):
        one_hot = torch.zeros(num_classes)
        one_hot[i] = 1
        labelset[i] = one_hot

    offset = 0
    if shift_label:
        offset = -1

    subset = [(ex.flatten(), labelset[label + offset])
              for idx, (ex, label) in enumerate(dataset) if idx < num_samples]

    return subset


def split(trainset: list, p: float = .8) -> tuple:
    """
    Splits the dataset into training and validation sets.

    Parameters:
    dataset (list): The dataset to be split.
    p (float): Proportion of the dataset to be used as the training set.

    Returns:
    tuple: A tuple containing the training and validation sets.
    """
    train, val = train_test_split(trainset, train_size=p)
    return train, val


def get_cifar(split_percentage: float = .8, num_train: int = float('inf'), num_test: int = float('inf')) -> tuple:
    """
    Prepares the CIFAR-10 dataset.

    Parameters:
    split_percentage (float): The percentage of the dataset to be used for training.
    num_train (int): The number of training samples to use.
    num_test (int): The number of test samples to use.

    Returns:
    tuple: A tuple containing DataLoaders for the training, validation, and test sets.
    """
    NUM_CLASSES = 10
    transform = transforms.Compose([transforms.ToTensor()])
    path = os.path.join(os.getcwd(), 'datasets/')

    trainset = torchvision.datasets.CIFAR10(root=path,
                                            train=True,
                                            transform=transform,
                                            download=False)

    trainset = one_hot_data(trainset, NUM_CLASSES, num_samples=num_train)
    trainset, valset = split(trainset, p=split_percentage)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=1)

    valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                            shuffle=False, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root=path,
                                           train=False,
                                           transform=transform,
                                           download=False)

    testset = one_hot_data(testset, NUM_CLASSES, num_samples=num_test)

    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=1)

    print("Num Train: ", len(trainset), "Num Val: ", len(valset),
          "Num Test: ", len(testset))
    return trainloader, valloader, testloader


def sample_data(num: int, d: int) -> tuple:
    """
    Generates random data for a regression problem.

    Parameters:
    num (int): The number of samples to generate.
    d (int): The dimension of each sample.

    Returns:
    tuple: A tuple of tensors containing the input data and corresponding labels.
    """
    X = np.random.normal(size=(num, d))
    y = X[:, 0] * X[:, 1]
    y = y.reshape(-1, 1)
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()


def get_two_coordinates(split_percentage: float = .8, num_train: int = 2000, num_test: int = 1000, d: int = 100) -> tuple:
    """
    Prepares a synthetic dataset for a regression problem.

    Parameters:
    split_percentage (float): The percentage of the dataset to be used for training.
    num_train (int): The number of training samples to generate.
    num_test (int): The number of test samples to generate.
    d (int): The dimension of each sample.

    Returns:
    tuple: A tuple containing DataLoaders for the training, validation, and test sets.
    """
    X, y = sample_data(num_train, d)
    trainset = list(zip(X, y))
    trainset, valset = split(trainset, p=split_percentage)
    X_test, y_test = sample_data(num_test, d)
    testset = list(zip(X_test, y_test))

    train_loader = DataLoader(trainset, batch_size=128,
                              shuffle=True, num_workers=1)
    val_loader = DataLoader(valset, batch_size=128,
                            shuffle=False, num_workers=1)
    test_loader = DataLoader(testset, batch_size=128,
                             shuffle=False, num_workers=1)
    print("Num Train: ", len(trainset), "Num Val: ", len(valset),
          "Num Test: ", len(testset))

    return train_loader, val_loader, test_loader


def celeba_subset(dataset: Dataset, feature_idx: int, num_samples: int = -1) -> list:
    """
    Creates a subset of the CelebA dataset based on a specific feature.

    Parameters:
    dataset (Dataset): The CelebA dataset.
    feature_idx (int): The index of the feature to be used for subsetting.
    num_samples (int): The number of samples to include in the subset.

    Returns:
    list: A list of tuples containing the data and corresponding labels.
    """
    NUM_CLASSES = 2
    labelset = {}
    for i in range(NUM_CLASSES):
        one_hot = torch.zeros(NUM_CLASSES)
        one_hot[i] = 1
        labelset[i] = one_hot

    by_class = {}
    features = []
    for idx in tqdm(range(len(dataset))):
        ex, label = dataset[idx]
        features.append(label[feature_idx])
        g = label[feature_idx].numpy().item()
        # ex = torch.mean(ex, dim=0)
        ex = ex.flatten()
        ex = ex / norm(ex)
        if g in by_class:
            by_class[g].append((ex, labelset[g]))
        else:
            by_class[g] = [(ex, labelset[g])]
        if idx > num_samples:
            break
    data = []
    if 1 in by_class:
        max_len = min(25000, len(by_class[1]))
        data.extend(by_class[1][:max_len])
        data.extend(by_class[0][:max_len])
    else:
        max_len = 1
        data.extend(by_class[0][:max_len])
    return data


def get_celeba(feature_idx: int, split_percentage: float = .8, num_train: int = float('inf'), num_test: int = float('inf')) -> tuple:
    """
    Prepares the CelebA dataset for a specific feature.

    Parameters:
    feature_idx (int): The index of the feature to be used.
    split_percentage (float): The percentage of the dataset to be used for training.
    num_train (int): The number of training samples to use.
    num_test (int): The number of test samples to use.

    Returns:
    tuple: A tuple containing DataLoaders for the training, validation, and test sets.
    """
    celeba_path = os.path.join(os.getcwd(), 'datasets/')
    SIZE = 96
    transform = transforms.Compose(
        [transforms.Resize([SIZE, SIZE]),
         transforms.ToTensor()
         ])

    trainset = torchvision.datasets.CelebA(root=celeba_path,
                                           split='train',
                                           transform=transform,
                                           download=True)
    trainset = celeba_subset(trainset, feature_idx, num_samples=num_train)
    trainset, valset = split(trainset, p=split_percentage)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=1)

    valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                            shuffle=False, num_workers=1)

    testset = torchvision.datasets.CelebA(root=celeba_path,
                                          split='test',
                                          transform=transform,
                                          download=True)
    testset = celeba_subset(testset, feature_idx, num_samples=num_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=1)

    print("Train Size: ", len(trainset), "Val Size: ",
          len(valset), "Test Size: ", len(testset))
    return trainloader, valloader, testloader


def group_by_class(dataset: Dataset) -> dict:
    """
    Groups images in the dataset by class.

    Parameters:
    dataset (Dataset): The dataset to be grouped.

    Returns:
    dict: A dictionary with class labels as keys and lists of images as values.
    """
    labelset = {}
    for i in range(10):
        labelset[i] = []
    for i, batch in enumerate(dataset):
        img, label = batch
        labelset[label].append(img.view(1, 3, 32, 32))
    return labelset


def merge_data(cifar: Dataset, mnist: Dataset, n: int) -> list:
    """
    Merges CIFAR and MNIST datasets.

    Parameters:
    cifar (Dataset): The CIFAR dataset.
    mnist (Dataset): The MNIST dataset.
    n (int): The number of samples to merge from each class.

    Returns:
    list: A list of tuples containing the merged data and corresponding labels.
    """
    cifar_by_label = group_by_class(cifar)
    mnist_by_label = group_by_class(mnist)

    merged_data = []
    merged_labels = []

    labelset = {}

    for i in range(10):
        one_hot = torch.zeros(1, 10)
        one_hot[0, i] = 1
        labelset[i] = one_hot

    for l in cifar_by_label:

        cifar_data = torch.cat(cifar_by_label[l])
        mnist_data = torch.cat(mnist_by_label[l])
        min_len = min(len(mnist_data), len(cifar_data))
        m = min(n, min_len)
        cifar_data = cifar_data[:m]
        mnist_data = mnist_data[:m]

        merged = torch.cat([cifar_data, mnist_data], axis=-1)
        # for i in range(3):
        #    vis.image(merged[i])
        merged_data.append(merged.reshape(m, -1))
        # print(merged.shape)
        merged_labels.append(np.repeat(labelset[l], m, axis=0))
    merged_data = torch.cat(merged_data, axis=0)

    merged_labels = np.concatenate(merged_labels, axis=0)
    merged_labels = torch.from_numpy(merged_labels)

    return list(zip(merged_data, merged_labels))


def draw_star(ex: torch.Tensor, v: float, c: int = 3) -> torch.Tensor:
    """
    Draws a star shape on an image tensor.

    Parameters:
    ex (torch.Tensor): The image tensor.
    v (float): The value to use for drawing the star.
    c (int): The number of channels to draw on.

    Returns:
    torch.Tensor: The image tensor with a star drawn on it.
    """
    ex[:c, 5:6, 7:14] = v
    ex[:c, 4, 9:12] = v
    ex[:c, 3, 10] = v
    ex[:c, 6, 8:13] = v
    ex[:c, 7, 9:12] = v
    ex[:c, 8, 8:13] = v
    ex[:c, 9, 8:10] = v
    ex[:c, 9, 11:13] = v
    return ex


def one_hot_stl_toy(dataset: Dataset, num_samples: int = -1) -> list:
    """
    Prepares a toy STL dataset with one-hot encoding.

    Parameters:
    dataset (Dataset): The STL dataset.
    num_samples (int): The number of samples to process.

    Returns:
    list: A list of tuples containing the data and corresponding one-hot encoded labels.
    """
    labelset = {}
    for i in range(2):
        one_hot = torch.zeros(2)
        one_hot[i] = 1
        labelset[i] = one_hot

    subset = [(ex, label) for idx, (ex, label) in enumerate(dataset)
              if idx < num_samples and (label == 0 or label == 9)]

    adjusted = []
    for idx, (ex, label) in enumerate(subset):
        if label == 9:
            ex = draw_star(ex, 1, c=2)
            y = 1
        else:
            ex = draw_star(ex, 0)
            y = 0
        ex = ex.flatten()
        adjusted.append((ex, labelset[y]))
    return adjusted


def get_stl_star(split_percentage: float = .8, num_train: int = float('inf'), num_test: int = float('inf')) -> tuple:
    """
    Prepares the STL dataset with star shapes added for a binary classification task.

    Parameters:
    split_percentage (float): The percentage of the dataset to be used for training.
    num_train (int): The number of training samples to use.
    num_test (int): The number of test samples to use.

    Returns:
    tuple: A tuple containing DataLoaders for the training, validation, and test sets.
    """
    SIZE = 96
    transform = transforms.Compose(
        [transforms.Resize([SIZE, SIZE]),
         transforms.ToTensor()
         ])

    path = os.path.join(os.getcwd(), 'datasets/')
    trainset = torchvision.datasets.STL10(root=path,
                                          split='train',
                                          # train=True,
                                          transform=transform,
                                          download=True)
    trainset = one_hot_stl_toy(trainset, num_samples=num_train)
    trainset, valset = split(trainset, p=split_percentage)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                            shuffle=False, num_workers=1)
    testset = torchvision.datasets.STL10(root=path,
                                         split='test',
                                         transform=transform,
                                         download=True)
    testset = one_hot_stl_toy(testset, num_samples=num_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=2)
    print("Num Train: ", len(trainset), "Num Val: ", len(valset),
          "Num Test: ", len(testset))
    return trainloader, valloader, testloader


# get_celeba(feature_idx=0, num_train=10_000, num_test=2_000)
# get_stl_star()
