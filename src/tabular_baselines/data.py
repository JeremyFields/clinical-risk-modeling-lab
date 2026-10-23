"""Dataset loading and deterministic train/validation splitting."""

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_FEATURES = (
    "age",
    "is_female",
    "number_of_unique_meds",
    "number_of_encounters",
    "number_of_procedures",
    "unique_procedures",
    "pain_severity",
    "body_height",
    "body_weight",
    "body_mass_index",
    "systolic_blood_pressure",
    "diastolic_blood_pressure",
    "heart_rate",
    "respiratory_rate",
    "qaly",
    "daly",
    "qols",
    "healthcare_expenses",
    "healthcare_coverage",
)

PAIN_TERMS = (
    "chronic", "pain", "arthritis", "osteoarthritis", "rheumatoid",
    "fibromyalgia", "migraine", "neuropathy", "neuralgia", "sciatica",
    "spinal", "fracture", "injury", "burn", "wound", "trauma", "sprain",
    "strain", "tendon", "ligament", "joint", "osteoporosis", "gout",
    "lupus", "paralysis", "amputation", "surgery", "postoperative", "whiplash",
)


@dataclass(frozen=True)
class TabularDataset:
    features: np.ndarray
    labels: np.ndarray
    feature_names: tuple[str, ...]


def load_dataset(
    path: str | Path,
    feature_names: Sequence[str] = DEFAULT_FEATURES,
) -> TabularDataset:
    """Load numeric features and derive the binary target from condition text."""
    frame = pd.read_csv(path)
    pattern = r"\b(?:" + "|".join(re.escape(term) for term in PAIN_TERMS) + r")\b"
    matcher = re.compile(pattern, flags=re.IGNORECASE)
    labels = np.array(
        [int(bool(matcher.search(str(text)))) for text in frame["condition_text"]],
        dtype=np.int64,
    )
    features = frame[list(feature_names)].to_numpy(dtype=np.float32)
    return TabularDataset(features, labels, tuple(feature_names))


def generate_synthetic_dataset(
    path: str | Path,
    *,
    n_samples: int = 320,
    positive_rate: float = 0.4,
    seed: int = 0,
) -> Path:
    """Write a reproducible, clearly synthetic clinical-style CSV for demos."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if not 0.0 < positive_rate < 1.0:
        raise ValueError("positive_rate must be between 0 and 1")

    rng = np.random.default_rng(seed)
    labels = (rng.random(n_samples) < positive_rate).astype(np.int64)
    age = np.clip(rng.normal(58, 17, n_samples), 18, 95)
    height = np.clip(rng.normal(168, 10, n_samples), 140, 205)
    bmi = np.clip(rng.normal(27, 6, n_samples) + labels * 1.5, 15, 48)
    weight = bmi * (height / 100) ** 2
    encounters = rng.poisson(4 + labels * 2, n_samples)
    medications = rng.poisson(3 + labels * 2, n_samples)
    procedures = rng.poisson(2 + labels, n_samples)
    pain_severity = np.clip(rng.normal(2.5 + labels * 3.0, 1.4, n_samples), 0, 10)
    systolic = np.clip(rng.normal(128 + age * 0.08, 14, n_samples), 90, 190)
    diastolic = np.clip(rng.normal(78, 9, n_samples), 45, 120)
    qaly = np.clip(rng.normal(0.82 - labels * 0.12, 0.1, n_samples), 0, 1)
    qols = np.clip(rng.normal(78 - labels * 18, 12, n_samples), 0, 100)

    frame = pd.DataFrame({
        "id": [f"SYN-{index:04d}" for index in range(n_samples)],
        "condition_text": np.where(
            labels == 1,
            "synthetic chronic pain cohort; essential hypertension",
            "synthetic seasonal allergy cohort; essential hypertension",
        ),
        "age": age.round(1),
        "is_female": rng.integers(0, 2, n_samples),
        "number_of_unique_meds": medications,
        "number_of_encounters": encounters,
        "number_of_procedures": procedures,
        "unique_procedures": np.maximum(procedures - rng.integers(0, 2, n_samples), 0),
        "pain_severity": pain_severity.round(1),
        "body_height": height.round(1),
        "body_weight": weight.round(1),
        "body_mass_index": bmi.round(1),
        "systolic_blood_pressure": systolic.round(1),
        "diastolic_blood_pressure": diastolic.round(1),
        "heart_rate": np.clip(rng.normal(74, 10, n_samples), 45, 130).round(1),
        "respiratory_rate": np.clip(rng.normal(16, 3, n_samples), 8, 30).round(1),
        "qaly": qaly.round(3),
        "daly": np.clip(rng.normal(0.1 + labels * 0.08, 0.05, n_samples), 0, 1).round(3),
        "qols": qols.round(1),
        "healthcare_expenses": np.clip(rng.normal(6500 + labels * 2500, 2500, n_samples), 500, None).round(2),
        "healthcare_coverage": np.clip(rng.normal(14000, 3000, n_samples), 1000, None).round(2),
    })
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def stratified_split(
    labels: np.ndarray,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> np.ndarray:
    """Return a boolean mask whose true entries belong to validation data."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    validation_mask = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        shuffled = rng.permutation(indices)
        validation_count = round(validation_fraction * len(indices))
        validation_mask[shuffled[:validation_count]] = True
    return validation_mask
