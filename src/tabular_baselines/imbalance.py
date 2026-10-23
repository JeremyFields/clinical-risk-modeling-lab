"""Strategies for handling imbalanced classification labels."""

from typing import Dict, Tuple

import numpy as np


def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """Return inverse-frequency weights for every class present."""
    classes, counts = np.unique(labels, return_counts=True)
    weights: Dict[int, float] = {}
    for class_label, count in zip(classes, counts):
        weights[int(class_label)] = float(len(labels) / (len(classes) * count))
    return weights


def random_oversample(
    features: np.ndarray, labels: np.ndarray, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Duplicate minority rows until every class reaches the majority count."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    majority = int(counts.max())
    extra_features = []
    extra_labels = []

    for class_label, count in zip(classes, counts):
        if count < majority:
            class_indices = np.where(labels == class_label)[0]
            deficit = majority - int(count)
            sampled_indices = rng.choice(class_indices, size=deficit, replace=True)
            extra_features.append(features[sampled_indices])
            extra_labels.append(labels[sampled_indices])

    if not extra_features:
        return features, labels
    return (
        np.concatenate([features, *extra_features]),
        np.concatenate([labels, *extra_labels]),
    )
