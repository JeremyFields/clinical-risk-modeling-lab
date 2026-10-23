"""Random Forest baseline and model explanations."""

from typing import List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(
    features: np.ndarray,
    labels: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 6,
    seed: int = 0,
) -> RandomForestClassifier:
    """Fit a reproducible Random Forest on prepared training data."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
    )
    return model.fit(features, labels)


def rank_feature_importances(
    model: RandomForestClassifier, feature_names: List[str]
) -> List[Tuple[str, float]]:
    """Pair feature names with importances and rank them descending."""
    ranked = []
    for name, importance in zip(feature_names, model.feature_importances_):
        ranked.append((name, float(importance)))
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked
