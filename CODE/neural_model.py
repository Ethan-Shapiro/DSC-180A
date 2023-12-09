import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class Nonlinearity(nn.Module):
    """
    A custom nonlinearity module that supports multiple activation functions.

    Parameters:
    name (str): The name of the activation function to use.
                Supported values are 'relu', 'sigmoid', 'leaky_relu', 'sine', 'tanh'.
    """

    def __init__(self, name: str = 'relu'):
        super(Nonlinearity, self).__init__()
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the specified activation function to the input tensor.

        Parameters:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The output tensor after applying the activation function.
        """
        if self.name == 'relu':
            return F.relu(x)
        if self.name == 'sigmoid':
            return torch.sigmoid(x)
        if self.name == 'leaky_relu':
            return F.leaky_relu(x)
        if self.name == 'sine':
            return torch.sin(x)
        if self.name == 'tanh':
            return torch.tanh(x)


class Net(nn.Module):
    """
    A custom neural network module with configurable depth and width.

    Parameters:
    dim (int): The size of the input feature vector.
    depth (int): The number of layers in the network.
    width (int): The number of neurons in each hidden layer.
    num_classes (int): The number of output classes (for the final layer).
    act_name (str): The name of the activation function to use.
    """

    def __init__(self, dim: int, depth: int = 1, width: int = 1024, num_classes: int = 2, act_name: str = 'relu'):
        super(Net, self).__init__()
        bias = False
        self.dim = dim
        self.width = width
        self.depth = depth
        self.name = act_name

        if depth == 1:
            self.first = nn.Linear(dim, width, bias=bias)
            self.fc = nn.Sequential(Nonlinearity(name=self.name),
                                    nn.Linear(width, num_classes, bias=bias))
        else:
            module = nn.Sequential(Nonlinearity(name=self.name),
                                   nn.Linear(width, width, bias=bias))
            num_layers = depth - 1
            self.first = nn.Sequential(nn.Linear(dim, width, bias=bias))
            self.middle = nn.ModuleList(
                [deepcopy(module) for idx in range(num_layers)])
            self.last = nn.Sequential(Nonlinearity(name=self.name),
                                      nn.Linear(width, num_classes, bias=bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Parameters:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The output tensor of the network.
        """
        if self.depth == 1:
            return self.fc(self.first(x))
        else:
            o = self.first(x)
            for idx, m in enumerate(self.middle):
                o = m(o)
            o = self.last(o)
            return o
