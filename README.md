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
python train.py --wandb_entity yourname --wandb_project AIML_Jul_Dec_2025 \
  --dataset fashion_mnist --epochs 10 --batch_size 64 \
  --optimizer adam --learning_rate 0.001 --num_layers 3 \
  --hidden_size 128 --activation ReLU

---



