"""Training utilities for the portfolio MLP experiments."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class TrainingReport:
    epochs_run: int
    best_epoch: int
    validation_loss: float
    validation_auc: float


def train_binary_classifier(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    epochs: int = 80,
    learning_rate: float = 1e-3,
    early_stopping: bool = True,
    patience: int = 8,
    min_delta: float = 0.0,
    seed: int = 0,
) -> TrainingReport:
    """Train an MLP with full-batch Adam and optional validation stopping."""
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs must be positive and learning_rate must be positive")
    if patience < 1:
        raise ValueError("patience must be at least 1")

    torch.manual_seed(seed)
    train_features = torch.as_tensor(x_train, dtype=torch.float32)
    train_labels = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    validation_features = torch.as_tensor(x_validation, dtype=torch.float32)
    validation_labels = torch.as_tensor(y_validation, dtype=torch.float32).reshape(-1, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCELoss()
    best_loss = float("inf")
    best_epoch = 0
    best_state: Optional[dict[str, torch.Tensor]] = None
    wait = 0
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss_function(model(train_features), train_labels).backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = float(
                loss_function(model(validation_features), validation_labels)
            )
        epochs_run = epoch

        if validation_loss < best_loss - min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if early_stopping and wait >= patience:
                break

    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return TrainingReport(
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        validation_loss=best_loss,
        validation_auc=_roc_auc(y_validation, model, validation_features),
    )


def _roc_auc(
    labels: np.ndarray,
    model: nn.Module,
    features: torch.Tensor,
) -> float:
    from sklearn.metrics import roc_auc_score

    with torch.no_grad():
        probabilities = model(features).cpu().numpy().ravel()
    return float(roc_auc_score(labels, probabilities))
