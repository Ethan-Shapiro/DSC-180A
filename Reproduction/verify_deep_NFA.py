import numpy as np
import torch
import random
import dataset
import neural_model
from torch.linalg import norm
from torch.utils.data import Dataset, DataLoader
from torch import vmap
from torch.func import jacrev

SEED = 1717

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)


def get_name(dataset_name: str, configs: dict) -> str:
    """
    Generates a string name based on dataset name and configuration parameters.

    Parameters:
    dataset_name (str): Name of the dataset.
    configs (dict): Dictionary containing configuration parameters.

    Returns:
    str: A concatenated string of dataset name and configuration parameters.
    """
    name_str = dataset_name
    for key in configs:
        name_str += key + '_' + str(configs[key]) + '_'
    name_str += 'nn'
    return name_str


def load_nn(path: str, width: int, depth: int, dim: int, num_classes: int, layer_idx: int = 0,
            remove_init: bool = False, act_name: str = 'relu') -> tuple:
    """
    Loads a neural network model from a given path and processes a specific layer.

    Parameters:
    path (str): Path to the saved neural network model.
    width (int): Width of the network.
    depth (int): Depth of the network.
    dim (int): Dimension of the input.
    num_classes (int): Number of output classes.
    layer_idx (int): Index of the layer to process.
    remove_init (bool): Flag to remove initial parameters.
    act_name (str): Name of the activation function used in the network.

    Returns:
    tuple: A tuple containing the neural network model and processed matrix M.
    """
    if remove_init:
        suffix = path.split('/')[-1]
        prefix = ''

        init_net = neural_model.Net(dim, width=width, depth=depth,
                                    num_classes=num_classes, act_name=act_name)
        d = torch.load(prefix + 'init_' + suffix)
        init_net.load_state_dict(d['state_dict'])
        init_params = [p for idx, p in enumerate(init_net.parameters())]

    net = neural_model.Net(dim, width=width, depth=depth,
                           num_classes=num_classes, act_name=act_name)

    d = torch.load(path)
    net.load_state_dict(d['state_dict'])

    for idx, p in enumerate(net.parameters()):
        if idx == layer_idx:
            M = p.data.numpy()
            print(M.shape)
            if remove_init:
                M0 = init_params[idx].data.numpy()
                M -= M0
            break

    M = M.T @ M * 1/len(M)

    return net, M


def load_init_nn(path: str, width: int, depth: int, dim: int, num_classes: int, layer_idx: int = 0, act_name: str = 'relu') -> tuple:
    """
    Loads the initial neural network model from a given path and processes a specific layer.

    Parameters:
    path (str): Path to the saved neural network model.
    width (int): Width of the network.
    depth (int): Depth of the network.
    dim (int): Dimension of the input.
    num_classes (int): Number of output classes.
    layer_idx (int): Index of the layer to process.
    remove_init (bool): Flag to remove initial parameters.
    act_name (str): Name of the activation function used in the network.

    Returns:
    tuple: A tuple containing the neural network model and processed matrix M.
    """
    suffix = path.split('/')[-1]
    prefix = '/Users/ethanshapiro/Repository/UCSD_Courses/dsc-180a/saved_nns/'

    net = neural_model.Net(dim, width=width, depth=depth,
                           num_classes=num_classes, act_name=act_name)
    d = torch.load(prefix + 'init_' + suffix)
    net.load_state_dict(d['state_dict'])

    for idx, p in enumerate(net.parameters()):
        if idx == layer_idx:
            M = p.data.numpy()
            print(M.shape)
            break

    M = M.T @ M * 1/len(M)
    return net, M


def get_layer_output(net: neural_model.Net, trainloader: DataLoader, layer_idx: int = 0) -> torch.Tensor:
    """
    Retrieves the output of a specific layer from a neural network for the given data.

    Parameters:
    net (neural_model.Net): The neural network model.
    trainloader (DataLoader): DataLoader containing the training data.
    layer_idx (int): The index of the layer from which to retrieve the output.

    Returns:
    torch.Tensor: The output of the specified layer for the entire dataset.
    """
    net.eval()
    out = []
    for idx, batch in enumerate(trainloader):
        data, _ = batch
        if layer_idx == 0:
            out.append(data.cpu())
        elif layer_idx == 1:
            o = neural_model.Nonlinearity()(net.first(data))
            out.append(o.cpu())
        elif layer_idx > 1:
            o = net.first(data)
            for l_idx, m in enumerate(net.middle):
                o = m(o)
                if l_idx + 1 == layer_idx:
                    o = neural_model.Nonlinearity()(o)
                    out.append(o.cpu())
                    break
    out = torch.cat(out, dim=0)
    net.cpu()
    return out


def build_subnetwork(net: neural_model.Net, dim: int, width: int, depth: int, num_classes: int,
                     layer_idx: int = 0, random_net: bool = False, act_name: str = 'relu') -> neural_model.Net:
    """
    Builds a subnetwork from a given neural network model.

    Parameters:
    net (neural_model.Net): The original neural network model.
    dim (int): Dimension of the input.
    width (int): Width of the network.
    depth (int): Depth of the network.
    num_classes (int): Number of output classes.
    layer_idx (int): Index of the layer from which the subnetwork starts.
    random_net (bool): Flag to determine if the subnetwork should be randomized.
    act_name (str): Name of the activation function used in the network.

    Returns:
    neural_model.Net: The subnetwork model.
    """
    net_ = neural_model.Net(dim, depth=depth - layer_idx,
                            width=width, num_classes=num_classes,
                            act_name=act_name)

    params = [p for idx, p in enumerate(net.parameters())]
    if not random_net:
        for idx, p_ in enumerate(net_.parameters()):
            p_.data = params[idx + layer_idx].data

    return net_


def get_jacobian(net: neural_model.Net, data: torch.Tensor) -> torch.Tensor:
    """
    Computes the Jacobian matrix of the neural network with respect to the input data.

    Parameters:
    net (neural_model.Net): The neural network model.
    data (torch.Tensor): Input data.

    Returns:
    torch.Tensor: The Jacobian matrix.
    """
    with torch.no_grad():
        return vmap(jacrev(net))(data).transpose(0, 2).transpose(0, 1)


def egop(net: neural_model.Net, dataset: torch.Tensor, centering: bool = False) -> torch.Tensor:
    """
    Computes the Empirical Gram-Operator of Perturbations (EGOP) for the network.

    Parameters:
    net (neural_model.Net): The neural network model.
    dataset (torch.Tensor): The dataset.
    centering (bool): Flag to indicate if centering should be applied.

    Returns:
    torch.Tensor: The EGOP matrix.
    """
    device = torch.device('cuda')
    bs = 1000
    batches = torch.split(dataset, bs)
    net = net.cuda()
    G = 0

    Js = []
    for batch_idx, data in enumerate(batches):
        data = data.to(device)
        print("Computing Jacobian for batch: ", batch_idx, len(batches))
        J = get_jacobian(net, data)
        Js.append(J.cpu())

        # Optional for stopping EGOP computation early
        # if batch_idx > 30:
        #    break
    Js = torch.cat(Js, dim=-1)
    if centering:
        J_mean = torch.mean(Js, dim=-1).unsqueeze(-1)
        Js = Js - J_mean

    Js = torch.transpose(Js, 2, 0)
    Js = torch.transpose(Js, 1, 2)
    print(Js.shape)
    batches = torch.split(Js, bs)
    for batch_idx, J in enumerate(batches):
        print(batch_idx, len(batches))
        m, c, d = J.shape
        J = J.cuda()
        G += torch.einsum('mcd,mcD->dD', J, J).cpu()
        del J
    G = G * 1/len(Js)

    return G


def correlate(M: torch.Tensor, G: torch.Tensor) -> float:
    """
    Computes the correlation between two matrices.

    Parameters:
    M (torch.Tensor): First matrix.
    G (torch.Tensor): Second matrix.

    Returns:
    float: The correlation coefficient between the two matrices.
    """
    M = M.double()
    G = G.double()
    normM = norm(M.flatten())
    normG = norm(G.flatten())

    corr = torch.dot(M.flatten(), G.flatten()) / (normM * normG)
    return corr.item()  # Convert to Python float


def read_configs(path: str) -> tuple:
    """
    Reads configuration settings from a file path string.

    Parameters:
    path (str): The file path containing configuration settings.

    Returns:
    tuple: A tuple containing width, depth, and activation name from the configuration.
    """
    tokens = path.strip().split('_')
    print(tokens)
    act_name = 'relu'
    for idx, t in enumerate(tokens):
        if t == 'width':
            width = int(tokens[idx+1])
        if t == 'depth':
            depth = int(tokens[idx+1])
        if t == 'act':
            act_name = tokens[idx+1]

    return width, depth, act_name


def verify_NFA(path: str, dataset_name: str, feature_idx: int = None, layer_idx: int = 0) -> tuple:
    """
    Verifies the Neural Feature Attribution (NFA) for a given neural network model and dataset.

    Parameters:
    path (str): Path to the neural network model.
    dataset_name (str): Name of the dataset to be used.
    feature_idx (int, optional): Index of the feature for which NFA is to be verified.
    layer_idx (int): Index of the layer in the neural network.

    Returns:
    tuple: A tuple containing initial correlation, centered correlation, and uncentered correlation.
    """
    remove_init = False
    random_net = False

    # Define network and data dimensions based on dataset
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

    # Read network configurations from path
    width, depth, act_name = read_configs(path)

    # Load the network and initial network
    net, M = load_nn(path, width, depth, dim, NUM_CLASSES, layer_idx=layer_idx,
                     remove_init=remove_init, act_name=act_name)
    net0, M0 = load_init_nn(path, width, depth, dim, NUM_CLASSES, layer_idx=layer_idx,
                            act_name=act_name)

    # Build subnetwork
    subnet = build_subnetwork(net, M.shape[0], width, depth, NUM_CLASSES, layer_idx=layer_idx,
                              random_net=random_net, act_name=act_name)

    # Calculate initial correlation
    init_correlation = correlate(torch.from_numpy(M),
                                 torch.from_numpy(M0))
    print("Init Net Feature Matrix Correlation: ", init_correlation)

    # Load data based on dataset name
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
    # Additional dataset loaders can be added here

    # Get layer output and compute EGOP
    out = get_layer_output(net, trainloader, layer_idx=layer_idx)
    G = egop(subnet, out, centering=True)
    G2 = egop(subnet, out, centering=False)

    # Calculate correlations
    centered_correlation = correlate(torch.from_numpy(M), G)
    uncentered_correlation = correlate(torch.from_numpy(M), G2)
    print("Full Matrix Correlation Centered: ", centered_correlation)
    print("Full Matrix Correlation Uncentered: ", uncentered_correlation)

    return init_correlation, centered_correlation, uncentered_correlation


def main():

    # Path to saved neural net model
    path = '/Users/ethanshapiro/Repository/UCSD_Courses/dsc-180a/saved_nns/stl_star_num_epochs_200_learning_rate_0.1_weight_decay_0_init_default_optimizer_sgd_freeze_False_width_512_depth_5_act_relu_nn.pth'
    idxs = [0, 1]  # Layers for which to compute EGOP
    init, centered, uncentered = [], [], []
    for idx in idxs:
        results = verify_NFA(path, 'stl_star', layer_idx=idx)
        i, c, u = results
        init.append(i.numpy().item())
        centered.append(c.numpy().item())
        uncentered.append(u.numpy().item())
    for idx in idxs:
        print("Layer " + str(idx), init[idx], centered[idx], uncentered[idx])


if __name__ == "__main__":
    main()

# def main():

#     # Path to saved neural net model
#     path = 'C:\Repository\DSC-180A\saved_nns\celeba_num_epochs_500_learning_rate_0.1_weight_decay_0_init_default_optimizer_sgd_freeze_False_width_1024_depth_2_act_relu_nn.pth'
#     idxs = [0, 1]  # Layers for which to compute EGOP
#     init, centered, uncentered = [], [], []
#     for idx in idxs:
#         results = verify_NFA(path, 'celeba', layer_idx=idx, feature_idx=16)
#         i, c, u = results
#         init.append(i.numpy().item())
#         centered.append(c.numpy().item())
#         uncentered.append(u.numpy().item())
#     for idx in idxs:
#         print("Layer " + str(idx), init[idx], centered[idx], uncentered[idx])


# if __name__ == "__main__":
#     main()
