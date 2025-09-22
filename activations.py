import numpy as np


def identity(x):
    return x


def identity_grad(x):
    return np.ones_like(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_grad(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(float)


def get_activation(name):
    """
    Returns activation function and its gradient.
    """
    if name.lower() == "identity":
        return identity, identity_grad
    elif name.lower() == "sigmoid":
        return sigmoid, sigmoid_grad
    elif name.lower() == "tanh":
        return tanh, tanh_grad
    elif name.lower() == "relu":
        return relu, relu_grad
    else:
        raise ValueError(f"Unknown activation function: {name}")
