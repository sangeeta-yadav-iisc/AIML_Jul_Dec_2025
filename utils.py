import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist, mnist


def load_data(dataset="fashion_mnist"):
    """
    Load dataset: Fashion-MNIST or MNIST.
    Normalize and flatten.
    """
    if dataset.lower() == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    elif dataset.lower() == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        raise ValueError("Dataset not supported")

    # Normalize
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Flatten
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Train/Validation/Test split
    X_val, y_val = X_train[-10000:], y_train[-10000:]
    X_train, y_train = X_train[:-10000], y_train[:-10000]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def plot_metrics(train_losses, val_losses, train_accs, val_accs, save_path="training_curves.png"):
    """
    Plot loss and accuracy curves.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
