import argparse
import wandb
import numpy as np
from utils import load_data, accuracy_score, plot_metrics
from model import FeedForwardNN
from optim import get_optimizer


def train(config=None):
    """
    Main training loop with wandb integration.
    """

    # Initialize wandb
    with wandb.init(config=config):
        config = wandb.config

        # Load dataset
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(config.dataset)

        # Model init
        model = FeedForwardNN(
            input_size=X_train.shape[1],
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_size=len(np.unique(y_train)),
            activation=config.activation,
            loss=config.loss,
        )

        # Optimizer init
        optimizer = get_optimizer(
            config.optimizer, model.parameters(), config
        )

        # Training loop
        train_losses, val_losses, train_accs, val_accs = [], [], [], []

        for epoch in range(config.epochs):
            epoch_loss, epoch_acc = model.train_one_epoch(
                X_train, y_train, optimizer, config.batch_size
            )

            val_loss, val_acc = model.evaluate(X_val, y_val)

            # Store metrics
            train_losses.append(epoch_loss)
            val_losses.append(val_loss)
            train_accs.append(epoch_acc)
            val_accs.append(val_acc)

            # Log to wandb
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                    "val_loss": val_loss,
                    "train_acc": epoch_acc,
                    "val_acc": val_acc,
                }
            )

            print(
                f"Epoch {epoch+1}/{config.epochs} | "
                f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

        # Final test accuracy
        test_loss, test_acc = model.evaluate(X_test, y_test)
        wandb.log({"test_loss": test_loss, "test_acc": test_acc})
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        # Save plots
        plot_metrics(train_losses, val_losses, train_accs, val_accs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Feedforward NN")

    # General args
    parser.add_argument("--wandb_project", type=str, default="cs6910_a1")
    parser.add_argument("--wandb_entity", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="fashion_mnist")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--loss", type=str, default="cross_entropy")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
    )
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument(
        "--activation",
        type=str,
        default="ReLU",
        choices=["identity", "sigmoid", "tanh", "ReLU"],
    )

    # Optimizer-specific
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    args = parser.parse_args()

    wandb.login()
    train(vars(args))
