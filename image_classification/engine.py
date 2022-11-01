"""Pytorch trainer for training model."""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import Union, Callable


class Engine:
    """
    Pytorch engine to train neural network.

    Attributes
    ----------
    model : AlexNet
        Custom model in use.
    loss_fn : BCEWithLogitsLoss
        Type of loss function in use. Currently binary cross entropy with logits.
    optimizer : Union[RMSprop, Adam]
        Optimizer to use. Either RMSprop or Adam.
    device : str
        Device on where to perform computations (either 'cuda' or 'cpu').
    loss_history : Dict[str, List[float]]
        Dictionary with ("train", "val") as keys and list with loss per epoch as values.
    acc_history : Dict[str, List[float]]
        Dictionary with ("val") as keys and list with accuracy per epoch as values.
    """

    def __init__(
        self,
        model: Callable,
        loss_fn: nn.BCEWithLogitsLoss,
        optimizer: Union[optim.RMSprop, optim.Adam],
        device: str,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.loss_history = {"train": [], "val": []}
        self.acc_history = {"val": []}

    def _train_step(self, batch: DataLoader):
        """Run backprop, update model and return loss."""
        self.model.train()
        # Get tensors.
        image = batch["image"].to(self.device)
        targets = batch["target"].to(self.device)
        # Clean gradients and perform backwards pass.
        self.optimizer.zero_grad()
        logits = self.model(image)
        loss = self.loss_fn(logits, targets)
        loss.backward()
        # Update model.
        self.optimizer.step()

        return loss.item()

    def _validation_step(self, batch: DataLoader):
        """Compute loss for val data."""
        self.model.eval()
        with torch.no_grad():
            # Get tensors.
            image = batch["image"].to(self.device)
            targets = batch["target"].to(self.device)
            # Get logits and compute loss.
            logits = self.model(image)
            loss = self.loss_fn(logits, targets).item()
            total = targets.size(0)
            correct = (logits.sigmoid().round() == targets).sum().item()

            return loss, total, correct

    def train(self, train_loader: DataLoader, val_loader: DataLoader, n_epochs: int):
        """
        Train neural network and display training and validation loss.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader with training data.
        val_loader : DataLoader
            DataLoader with validation data.
        n_epochs : int
            Number of epochs to train.
        """
        for epoch in range(1, n_epochs + 1):
            losses = []

            for batch in train_loader:
                loss = self._train_step(batch)
                losses.append(loss)

            epoch_train_loss = np.mean(losses)
            self.loss_history["train"].append(epoch_train_loss)

            with torch.no_grad():
                losses, accuracies = [], []
                correct = 0
                total = 0

                for batch in val_loader:
                    loss, tot, corr = self._validation_step(batch)
                    losses.append(loss)
                    total += tot
                    correct += corr

                epoch_val_loss = np.mean(losses)
                epoch_val_acc = np.mean(accuracies)
                self.loss_history["val"].append(epoch_val_loss)

                self.acc_history["val"].append(epoch_val_acc)

            print(
                f"[{epoch}/{n_epochs}] Training loss: {epoch_train_loss:.4f}\t Validation loss: {epoch_val_loss:.4f}\t Validation accuracy: {correct/total:.4f}"
            )

    def plot_losses(self):
        """Plot losses accross epochs."""
        plt.plot(self.loss_history["train"], label="Training loss")
        plt.plot(self.loss_history["val"], label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()
