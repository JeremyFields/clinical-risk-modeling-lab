"""Configurable PyTorch feed-forward model builders."""

from typing import List

from torch import nn


def build_feedforward_model(
    input_dim: int,
    hidden_units: List[int],
    activation: str = "relu",
    dropout_rate: float = 0.0,
) -> nn.Module:
    """Build an untrained binary classifier from an architecture specification."""
    layers = []
    current_dim = input_dim

    for hidden_dim in hidden_units:
        layers.append(nn.Linear(current_dim, hidden_dim))
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "tanh":
            layers.append(nn.Tanh())
        else:
            raise ValueError("activation must be 'relu' or 'tanh'")
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        current_dim = hidden_dim

    layers.extend([nn.Linear(current_dim, 1), nn.Sigmoid()])
    return nn.Sequential(*layers)


def build_shape_comparison(input_dim: int):
    """Return deep/narrow and wide/shallow models with similar capacity."""
    deep = build_feedforward_model(input_dim, [16, 16, 16])
    wide = build_feedforward_model(input_dim, [48])
    return deep, wide


def build_activation_comparison(input_dim: int):
    """Return identical networks differing only in hidden activation."""
    relu = build_feedforward_model(input_dim, [16, 16, 16], activation="relu")
    tanh = build_feedforward_model(input_dim, [16, 16, 16], activation="tanh")
    return relu, tanh


def build_dropout_comparison(input_dim: int, dropout_rate: float):
    """Return identical networks differing only in dropout."""
    dropped = build_feedforward_model(
        input_dim, [16, 16, 16], dropout_rate=dropout_rate
    )
    plain = build_feedforward_model(input_dim, [16, 16, 16])
    return dropped, plain
