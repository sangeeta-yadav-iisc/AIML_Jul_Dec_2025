import numpy as np


class Optimizer:
    def step(self, weights, biases, grads):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr=0.01, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, weights, biases, grads):
        for i in range(len(weights)):
            weights[i] -= self.lr * (grads["dW"][i] + self.weight_decay * weights[i])
            biases[i] -= self.lr * grads["dB"][i]


class Momentum(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.lr, self.momentum, self.weight_decay = lr, momentum, weight_decay
        self.vW, self.vB = None, None

    def step(self, weights, biases, grads):
        if self.vW is None:
            self.vW = [np.zeros_like(w) for w in weights]
            self.vB = [np.zeros_like(b) for b in biases]

        for i in range(len(weights)):
            self.vW[i] = self.momentum * self.vW[i] + self.lr * (
                grads["dW"][i] + self.weight_decay * weights[i]
            )
            self.vB[i] = self.momentum * self.vB[i] + self.lr * grads["dB"][i]

            weights[i] -= self.vW[i]
            biases[i] -= self.vB[i]


class NAG(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.lr, self.momentum, self.weight_decay = lr, momentum, weight_decay
        self.vW, self.vB = None, None

    def step(self, weights, biases, grads):
        if self.vW is None:
            self.vW = [np.zeros_like(w) for w in weights]
            self.vB = [np.zeros_like(b) for b in biases]

        for i in range(len(weights)):
            lookahead_w = weights[i] - self.momentum * self.vW[i]
            lookahead_b = biases[i] - self.momentum * self.vB[i]

            self.vW[i] = self.momentum * self.vW[i] + self.lr * (
                grads["dW"][i] + self.weight_decay * lookahead_w
            )
            self.vB[i] = self.momentum * self.vB[i] + self.lr * grads["dB"][i]

            weights[i] -= self.vW[i]
            biases[i] -= self.vB[i]


class RMSProp(Optimizer):
    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8, weight_decay=0.0):
        self.lr, self.beta, self.epsilon, self.weight_decay = lr, beta, epsilon, weight_decay
        self.sW, self.sB = None, None

    def step(self, weights, biases, grads):
        if self.sW is None:
            self.sW = [np.zeros_like(w) for w in weights]
            self.sB = [np.zeros_like(b) for b in biases]

        for i in range(len(weights)):
            self.sW[i] = self.beta * self.sW[i] + (1 - self.beta) * grads["dW"][i] ** 2
            self.sB[i] = self.beta * self.sB[i] + (1 - self.beta) * grads["dB"][i] ** 2

            weights[i] -= self.lr * (grads["dW"][i] / (np.sqrt(self.sW[i]) + self.epsilon)
                                     + self.weight_decay * weights[i])
            biases[i] -= self.lr * (grads["dB"][i] / (np.sqrt(self.sB[i]) + self.epsilon))


class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        self.lr, self.beta1, self.beta2, self.epsilon, self.weight_decay = lr, beta1, beta2, epsilon, weight_decay
        self.mW, self.vW, self.mB, self.vB = None, None, None, None
        self.t = 0

    def step(self, weights, biases, grads):
        if self.mW is None:
            self.mW = [np.zeros_like(w) for w in weights]
            self.vW = [np.zeros_like(w) for w in weights]
            self.mB = [np.zeros_like(b) for b in biases]
            self.vB = [np.zeros_like(b) for b in biases]

        self.t += 1
        for i in range(len(weights)):
            self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * grads["dW"][i]
            self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * (grads["dW"][i] ** 2)
            self.mB[i] = self.beta1 * self.mB[i] + (1 - self.beta1) * grads["dB"][i]
            self.vB[i] = self.beta2 * self.vB[i] + (1 - self.beta2) * (grads["dB"][i] ** 2)

            mW_hat = self.mW[i] / (1 - self.beta1 ** self.t)
            vW_hat = self.vW[i] / (1 - self.beta2 ** self.t)
            mB_hat = self.mB[i] / (1 - self.beta1 ** self.t)
            vB_hat = self.vB[i] / (1 - self.beta2 ** self.t)

            weights[i] -= self.lr * (mW_hat / (np.sqrt(vW_hat) + self.epsilon)
                                     + self.weight_decay * weights[i])
            biases[i] -= self.lr * (mB_hat / (np.sqrt(vB_hat) + self.epsilon))


class Nadam(Adam):
    def step(self, weights, biases, grads):
        if self.mW is None:
            self.mW = [np.zeros_like(w) for w in weights]
            self.vW = [np.zeros_like(w) for w in weights]
            self.mB = [np.zeros_like(b) for b in biases]
            self.vB = [np.zeros_like(b) for b in biases]

        self.t += 1
        for i in range(len(weights)):
            self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * grads["dW"][i]
            self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * (grads["dW"][i] ** 2)
            self.mB[i] = self.beta1 * self.mB[i] + (1 - self.beta1) * grads["dB"][i]
            self.vB[i] = self.beta2 * self.vB[i] + (1 - self.beta2) * (grads["dB"][i] ** 2)

            mW_hat = self.mW[i] / (1 - self.beta1 ** self.t)
            vW_hat = self.vW[i] / (1 - self.beta2 ** self.t)
            mB_hat = self.mB[i] / (1 - self.beta1 ** self.t)
            vB_hat = self.vB[i] / (1 - self.beta2 ** self.t)

            nW = (self.beta1 * mW_hat + (1 - self.beta1) * grads["dW"][i] / (1 - self.beta1 ** self.t))
            nB = (self.beta1 * mB_hat + (1 - self.beta1) * grads["dB"][i] / (1 - self.beta1 ** self.t))

            weights[i] -= self.lr * (nW / (np.sqrt(vW_hat) + self.epsilon)
                                     + self.weight_decay * weights[i])
            biases[i] -= self.lr * (nB / (np.sqrt(vB_hat) + self.epsilon))


def get_optimizer(name, params, config):
    """
    Factory method to return optimizer instance.
    """
    if name.lower() == "sgd":
        return SGD(lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif name.lower() == "momentum":
        return Momentum(lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    elif name.lower() == "nag":
        return NAG(lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    elif name.lower() == "rmsprop":
        return RMSProp(lr=config["learning_rate"], beta=config["beta1"], epsilon=config["epsilon"], weight_decay=config["weight_decay"])
    elif name.lower() == "adam":
        return Adam(lr=config["learning_rate"], beta1=config["beta1"], beta2=config["beta2"], epsilon=config["epsilon"], weight_decay=config["weight_decay"])
    elif name.lower() == "nadam":
        return Nadam(lr=config["learning_rate"], beta1=config["beta1"], beta2=config["beta2"], epsilon=config["epsilon"], weight_decay=config["weight_decay"])
    else:
        raise ValueError(f"Unknown optimizer: {name}")
