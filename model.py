import numpy as np
from activations import get_activation
from losses import get_loss


class FeedForwardNN:
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 activation="ReLU", loss="cross_entropy", seed=42):
        np.random.seed(seed)

        self.layers = num_layers
        self.loss_name = loss
        self.loss_fn, self.loss_grad = get_loss(loss)

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.activations = []
        self.activation_grads = []

        # Input → Hidden
        self.weights.append(np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size))
        self.biases.append(np.zeros((1, hidden_size)))
        act, act_grad = get_activation(activation)
        self.activations.append(act)
        self.activation_grads.append(act_grad)

        # Hidden → Hidden
        for _ in range(num_layers - 2):
            self.weights.append(np.random.randn(hidden_size, hidden_size) * np.sqrt(2. / hidden_size))
            self.biases.append(np.zeros((1, hidden_size)))
            act, act_grad = get_activation(activation)
            self.activations.append(act)
            self.activation_grads.append(act_grad)

        # Hidden → Output
        self.weights.append(np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size))
        self.biases.append(np.zeros((1, output_size)))
        # Output layer uses softmax (for classification)
        self.activations.append(lambda x: np.exp(x - np.max(x, axis=1, keepdims=True)) /
                                         np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True))
        self.activation_grads.append(None)

        # For optimizers
        self.grads = {"dW": [np.zeros_like(W) for W in self.weights],
                      "dB": [np.zeros_like(b) for b in self.biases]}

    def parameters(self):
        return {"weights": self.weights, "biases": self.biases, "grads": self.grads}

    def forward(self, X):
        """
        Forward pass through the network.
        """
        activations = [X]
        for i in range(self.layers):
            Z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            A = self.activations[i](Z)
            activations.append(A)
        return activations

    def backward(self, activations, y_true):
        """
        Backward pass: compute gradients.
        """
        m = y_true.shape[0]
        y_onehot = np.zeros_like(activations[-1])
        y_onehot[np.arange(m), y_true] = 1

        dA = self.loss_grad(activations[-1], y_onehot)

        for i in reversed(range(self.layers)):
            Z = np.dot(activations[i], self.weights[i]) + self.biases[i]
            if i != self.layers - 1:  # hidden layers
                dZ = dA * self.activation_grads[i](Z)
            else:  # output layer
                dZ = dA

            self.grads["dW"][i] = np.dot(activations[i].T, dZ) / m
            self.grads["dB"][i] = np.sum(dZ, axis=0, keepdims=True) / m
            dA = np.dot(dZ, self.weights[i].T)

    def train_one_epoch(self, X, y, optimizer, batch_size=64):
        """
        Train model for one epoch (mini-batch gradient descent).
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        total_loss, correct = 0, 0

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            X_batch, y_batch = X[batch_idx], y[batch_idx]

            # Forward
            activations = self.forward(X_batch)
            loss = self.loss_fn(activations[-1], y_batch)
            total_loss += loss * len(X_batch)

            # Accuracy
            preds = np.argmax(activations[-1], axis=1)
            correct += np.sum(preds == y_batch)

            # Backward
            self.backward(activations, y_batch)

            # Update
            optimizer.step(self.weights, self.biases, self.grads)

        avg_loss = total_loss / n_samples
        acc = correct / n_samples
        return avg_loss, acc

    def evaluate(self, X, y):
        """
        Evaluate model on validation/test set.
        """
        activations = self.forward(X)
        loss = self.loss_fn(activations[-1], y)
        preds = np.argmax(activations[-1], axis=1)
        acc = np.mean(preds == y)
        return loss, acc
