"""End-to-end comparison of independent tabular model families."""

from dataclasses import asdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .data import TabularDataset, stratified_split
from .forest import train_random_forest
from .imbalance import random_oversample
from .neural_networks import build_feedforward_model
from .training import train_binary_classifier


def run_comparison(
    dataset: TabularDataset,
    *,
    seed: int = 0,
    validation_fraction: float = 0.2,
    epochs: int = 80,
) -> dict:
    """Train and evaluate a forest and an MLP on one deterministic split."""
    validation_mask = stratified_split(dataset.labels, validation_fraction, seed)
    x_train = dataset.features[~validation_mask]
    y_train = dataset.labels[~validation_mask]
    x_validation = dataset.features[validation_mask]
    y_validation = dataset.labels[validation_mask]

    balanced_features, balanced_labels = random_oversample(x_train, y_train, seed)
    forest = train_random_forest(
        balanced_features, balanced_labels, n_estimators=200, max_depth=6, seed=seed
    )
    forest_probabilities = forest.predict_proba(x_validation)[:, 1]

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(balanced_features).astype(np.float32)
    scaled_validation = scaler.transform(x_validation).astype(np.float32)
    mlp = build_feedforward_model(
        dataset.features.shape[1], [32, 16], activation="relu", dropout_rate=0.2
    )
    training_report = train_binary_classifier(
        mlp,
        scaled_train,
        balanced_labels,
        scaled_validation,
        y_validation,
        epochs=epochs,
        early_stopping=True,
        seed=seed,
    )
    mlp.eval()
    with torch.no_grad():
        mlp_probabilities = mlp(torch.as_tensor(scaled_validation)).numpy().ravel()

    return {
        "split": {"train_rows": int(len(x_train)), "validation_rows": int(len(x_validation))},
        "forest": _classification_summary(y_validation, forest_probabilities),
        "mlp": {
            **_classification_summary(y_validation, mlp_probabilities),
            "training": asdict(training_report),
        },
    }


def _classification_summary(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }
