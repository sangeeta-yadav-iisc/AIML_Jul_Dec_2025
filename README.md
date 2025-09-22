# AIML_Jul_Dec_2025 Assignment 1: Feedforward Neural Network with Backpropagation


### Goal
1. Implement feedforward neural network and backpropagation.
2. Implement multiple optimizers: SGD, Momentum, NAG, RMSProp, Adam, Nadam.
3. Run hyperparameter sweeps using Weights & Biases (wandb).
4. Compare cross-entropy and MSE loss.
5. Report results and insights.

---

### Usage
```bash
python train.py --wandb_entity yourname --wandb_project cs6910_a1 \
  --dataset fashion_mnist --epochs 10 --batch_size 64 \
  --optimizer adam --learning_rate 0.001 --num_layers 3 \
  --hidden_size 128 --activation ReLU


This repository contains the code for **AIML - Assignment 6
**. The goal of this assignment is to implement a feedforward neural network from scratch using **NumPy** and train it on the **Fashion-MNIST** dataset using various optimization algorithms.
We also use **Weights & Biases (wandb)** to track experiments and hyperparameter sweeps.

---



