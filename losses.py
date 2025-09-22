import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """
    Cross-entropy loss for classification.
    y_true: one-hot or integer labels
    y_pred: probabilities from softmax
    """
    m = y_true.shape[0]
    if y_true.ndim == 1:  # convert to one-hot
        one_hot = np.zeros_like(y_pred)
        one_hot[np.arange(m), y_true] = 1
        y_true = one_hot

    # numerical stability
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.sum(y_true * np.log(y_pred)) / m
    return loss


def cross_entropy_grad(y_pred, y_true):
    """
    Gradient of cross-entropy w.r.t logits.
    """
    m = y_true.shape[0]
    if y_true.ndim == 1:  # convert to one-hot
        one_hot = np.zeros_like(y_pred)
        one_hot[np.arange(m), y_true] = 1
        y_true = one_hot

    return (y_pred - y_true) / m


def mse_loss(y_pred, y_true):
    """
    Mean Squared Error (alternative loss).
    """
    if y_true.ndim == 1:
        y_true = np.eye(y_pred.shape[1])[y_true]
    return np.mean((y_pred - y_true) ** 2)


def mse_grad(y_pred, y_true):
    if y_true.ndim == 1:
        y_true = np.eye(y_pred.shape[1])[y_true]
    return 2 * (y_pred - y_true) / y_true.shape[0]


def get_loss(name):
    """
    Returns loss function and its gradient.
    """
    if name.lower() in ["cross_entropy", "ce"]:
        return cross_entropy_loss, cross_entropy_grad
    elif name.lower() in ["mse", "mean_squared_error"]:
        return mse_loss, mse_grad
    else:
        raise ValueError(f"Unknown loss function: {name}")
