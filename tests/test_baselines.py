import numpy as np
import torch

from tabular_baselines.forest import rank_feature_importances
from tabular_baselines.imbalance import compute_class_weights, random_oversample
from tabular_baselines.neural_networks import build_feedforward_model
from tabular_baselines.data import generate_synthetic_dataset, load_dataset


def test_class_weights_and_oversampling():
    labels = np.array([0] * 7 + [1] * 3)
    weights = compute_class_weights(labels)
    assert weights[1] > weights[0]

    features = np.arange(20).reshape(10, 2)
    balanced_features, balanced_labels = random_oversample(features, labels, seed=42)
    assert np.bincount(balanced_labels).tolist() == [7, 7]
    np.testing.assert_array_equal(balanced_features[:10], features)


def test_forest_importance_ranking():
    class FakeForest:
        feature_importances_ = np.array([0.1, 0.5, 0.05])

    assert rank_feature_importances(FakeForest(), ["a", "b", "c"])[0] == ("b", 0.5)


def test_feedforward_model_shapes():
    model = build_feedforward_model(19, [16, 8], activation="relu")
    output = model(torch.zeros(10, 19))
    assert output.shape == (10, 1)
    assert torch.all((output >= 0) & (output <= 1))


def test_synthetic_generator_is_reproducible(tmp_path):
    first_path = generate_synthetic_dataset(tmp_path / "first.csv", n_samples=40, seed=7)
    second_path = generate_synthetic_dataset(tmp_path / "second.csv", n_samples=40, seed=7)

    assert first_path.read_bytes() == second_path.read_bytes()
    dataset = load_dataset(first_path)
    assert dataset.features.shape == (40, 19)
    assert set(dataset.labels).issubset({0, 1})
    assert 0 < dataset.labels.sum() < 40
