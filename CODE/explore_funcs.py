import os
from scipy import linalg as LA
from scipy.sparse.linalg import eigsh
import numpy as np
import matplotlib.pyplot as plt

from verify_deep_NFA import *


def get_NFA_correlation(path: str, init_path: str, layer_idx: int = 0, feature_idx: int = 0) -> list:
    """
    Calculates Neural Feature Attribution (NFA) correlations for a given network layer and feature index.

    Parameters:
    path (str): The file path to the neural network model.
    init_path (str): The file path to the initial neural network model.
    layer_idx (int): The index of the network layer.
    feature_idx (int): The index of the feature for NFA calculation.

    Returns:
    list: A list containing centered NFA correlations.
    """
    results = verify_NFA(
        path, init_path, 'celeba', layer_idx=layer_idx, feature_idx=feature_idx)
    init, centered, uncentered = results
    return init, centered, uncentered


def calculate_NFM_GOP(path: str, dataset_name: str, feature_idx: int = None, layer_idx: int = 0) -> tuple:
    """
    Calculates the Neural Feature Matrix (NFM) and Gradient Outer Product (GOP) for a specified layer and dataset.

    Parameters:
    path (str): The file path to the neural network model.
    dataset_name (str): The name of the dataset to use.
    feature_idx (int, optional): The index of the feature for NFM and GOP calculation.
    layer_idx (int): The index of the network layer.

    Returns:
    tuple: A tuple containing the NFM and GOP matrices.
    """
    remove_init = False
    random_net = False

    if dataset_name == 'celeba':
        NUM_CLASSES = 2
        FEATURE_IDX = feature_idx
        SIZE = 96
        c = 3
        dim = c * SIZE * SIZE
    elif dataset_name == 'stl_star':
        NUM_CLASSES = 2
        c = 3
        SIZE = 96
        dim = c * SIZE * SIZE

    width, depth, act_name = read_configs(path)

    net, M = load_nn(path, width, depth, dim, NUM_CLASSES, layer_idx=layer_idx,
                     remove_init=remove_init, act_name=act_name)
    subnet = build_subnetwork(net, M.shape[0], width, depth, NUM_CLASSES, layer_idx=layer_idx,
                              random_net=random_net, act_name=act_name)

    if dataset_name == 'celeba':
        trainloader, valloader, testloader = dataset.get_celeba(FEATURE_IDX,
                                                                num_train=20000,
                                                                num_test=1)
    elif dataset_name == 'cifar':
        trainloader, valloader, testloader = dataset.get_cifar(num_train=1000,
                                                               num_test=1)
    elif dataset_name == 'stl_star':
        trainloader, valloader, testloader = dataset.get_stl_star(num_train=1000,
                                                                  num_test=1)
    out = get_layer_output(net, trainloader, layer_idx=layer_idx)
    G = egop(subnet, out, centering=True)
    G2 = egop(subnet, out, centering=False)

    print(
        f"Calulated Neural Feature Matrix and Gradient Outer Product for layer: {layer_idx}")

    return M, G


def plot_NFM_GOP(Mat: np.ndarray, is_gop: bool = False, feature_name: str = "Placeholder", dataset: str = 'default', save: bool = False):
    """
    Plots the Neural Feature Matrix (NFM) or Gradient Outer Product (GOP) of a layer.

    Parameters:
    Mat (np.ndarray): The matrix (NFM or GOP) to be plotted.
    feature_name (str): The name of the feature or layer for titling the plot.
    dataset (str): The name of the dataset for file naming.
    save (bool): Flag to determine if the plot should be saved as an image.

    Returns:
    None
    """
    if is_gop:
        save_path = os.path.join(os.getcwd(), 'image_outputs/NFM_GOP/GOP')
    else:
        save_path = os.path.join(os.getcwd(), 'image_outputs/NFM_GOP/NFM')
    # Extract the diagonal
    diagonal = np.diag(Mat)

    # Split the diagonal into three parts for RGB channels
    length = len(diagonal) // 3
    diagonal_r = diagonal[:length]
    diagonal_g = diagonal[length:2*length]
    diagonal_b = diagonal[2*length:]

    # Reshape each part into a 96x96 matrix
    image_r = diagonal_r.reshape(96, 96)
    image_g = diagonal_g.reshape(96, 96)
    image_b = diagonal_b.reshape(96, 96)

    # Stack the reshaped parts to form a 96x96x3 image
    image_rgb = np.stack((image_r, image_g, image_b), axis=-1)

    # Scale the values to be in the range [0, 1] for visualization
    image_rgb_scaled = (image_rgb - np.min(image_rgb)) / \
        (np.max(image_rgb) - np.min(image_rgb))

    # Display the image
    plt.imshow(image_rgb_scaled)
    plt.title(feature_name)
    plt.axis('off')  # Turn off axis numbers
    if save:
        plt.savefig(f"{save_path}/{feature_name}_{dataset}.png",
                    bbox_inches='tight')
    plt.show()

    # matrix becomes immutable so we have to copy to make it mutable
    diagonal = np.diag(Mat).copy()  # Make a mutable copy

    # Split the diagonal into three parts for RGB channels
    length = len(diagonal) // 3
    diagonal_r = diagonal[:length].copy()  # Make a mutable copy
    diagonal_g = diagonal[length:2*length].copy()  # Make a mutable copy
    diagonal_b = diagonal[2*length:].copy()  # Make a mutable copy

    # Reshape each part into a 96x96 matrix
    image_r = diagonal_r.reshape(96, 96)
    image_g = diagonal_g.reshape(96, 96)
    image_b = diagonal_b.reshape(96, 96)

    # filter the 2% of pixels to black
    for image in [image_r, image_g, image_b]:
        threshold = np.percentile(image, 98)  # Find the 98th percentile
        # Replace pixels above the threshold with 1
        image[image > threshold] = 1

    # Stack the reshaped parts to form a 96x96x3 image
    image_rgb = np.stack((image_r, image_g, image_b), axis=-1)

    # Display the image
    plt.imshow(image_rgb)
    plt.axis('off')  # Turn off axis numbers
    if save:
        plt.savefig(f"{save_path}/{feature_name}_{dataset}_Thresh.png",
                    bbox_inches='tight')
    plt.show()


def plot_top_eigenvector(Mat: np.ndarray, is_gop: bool = False, feature_name: str = "TopEigenvector", dataset: str = 'placeholder', save: bool = False):
    """
    Plots the top eigenvector of the Neural Feature Matrix (NFM) or Gradient Outer Product (GOP).

    Parameters:
    Mat (np.ndarray): The matrix (NFM or GOP) whose top eigenvector is to be plotted.
    feature_name (str): The name of the feature or layer for titling the plot.
    dataset (str): The name of the dataset for file naming.
    save (bool): Flag to determine if the plot should be saved as an image.

    Returns:
    None
    """
    # Compute eigenvectors and eigenvalues of the square matrix
    eigenvalues, eigenvectors = eigsh(Mat, k=1)

    # Extract the top eigenvector and normalize it
    top_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
    if not is_gop:
        top_eigenvector /= np.linalg.norm(top_eigenvector)

    # Split the top eigenvector into three parts for RGB channels
    length = len(top_eigenvector) // 3
    top_eigenvector_r = top_eigenvector[:length]
    top_eigenvector_g = top_eigenvector[length:2*length]
    top_eigenvector_b = top_eigenvector[2*length:]

    # Reshape each part into a 96x96 matrix for visualization
    image_r = top_eigenvector_r.reshape(96, 96)
    image_g = top_eigenvector_g.reshape(96, 96)
    image_b = top_eigenvector_b.reshape(96, 96)

    # Normalize each channel to be in the range [0, 1]
    image_r = (image_r - np.min(image_r)) / (np.max(image_r) - np.min(image_r))
    image_g = (image_g - np.min(image_g)) / (np.max(image_g) - np.min(image_g))
    image_b = (image_b - np.min(image_b)) / (np.max(image_b) - np.min(image_b))

    # Combine into a single RGB image
    top_image_rgb = np.stack((image_r, image_g, image_b), axis=-1)

    # Plot the top eigenvector
    plt.imshow(top_image_rgb)
    plt.title(f"{feature_name}")
    plt.axis('off')

    # Save the plot if required
    if save:
        if is_gop:
            save_path = os.path.join(
                os.getcwd(), 'image_outputs/EigenVector_NFM_GOP/GOP')
        else:
            save_path = os.path.join(
                os.getcwd(), 'image_outputs/EigenVector_NFM_GOP/NFM')

        plt.savefig(os.path.join(
            save_path, f"{feature_name}_{dataset}.png"), bbox_inches='tight')
    plt.show()
