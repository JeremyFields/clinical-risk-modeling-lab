# Clinical Risk Modeling Lab

An independent, reproducible study of binary risk modeling on tabular clinical
data. The project asks a practical question: how do data imbalance, model
capacity, and regularization change the quality of positive-case ranking?

The study compares two **independent** model families:

- **Random Forest**: a nonlinear tree ensemble used as a strong tabular
  baseline and a source of feature-ranking diagnostics.
- **PyTorch MLP**: a configurable feed-forward network used to study shape,
  activation, dropout, and training control.

The forest is not stacked on top of the MLP and is not a neural-network layer.
Both models consume the same prepared feature matrix, and their predictions
are evaluated with the same classification metrics.

## Design

```text
src/tabular_baselines/
  imbalance.py       reusable resampling and weighting strategies
  forest.py          tree baseline and feature-ranking diagnostics
  neural_networks.py MLP construction and architecture experiments
  metrics.py         threshold and ranking metrics
```

The package is intentionally separate from the submitted coursework in the
parent `pipeline/` directory. It uses a new public API and a more general
experiment framing rather than reproducing the assignment file structure.

## Data

The project does not require a downloaded clinical dataset. It generates a
clearly synthetic clinical-style CSV with a fixed seed:

```powershell
python -m tabular_baselines --generate-demo-data demo_data\synthetic_clinical.csv --epochs 80
```

The generated records are artificial and are not patient records. They are
useful for demonstrating the pipeline, not for clinical conclusions. The
generator is deterministic for a given seed, so results can be reproduced.

You can also provide a compatible CSV explicitly:

```powershell
python -m tabular_baselines --data path\to\your_dataset.csv --epochs 80
```

It must contain `condition_text` plus the numeric feature columns defined in
`data.py`. The loader derives the binary target from the condition text.

## Development

From this folder:

```powershell
python -m pytest
python -m pip install -e .
python -m tabular_baselines --generate-demo-data demo_data\synthetic_clinical.csv --epochs 80
```


## Interpreting The Demo

With the default seed, one local run produced approximately:

```text
Random Forest ROC-AUC: 0.997
MLP ROC-AUC:           0.917
```

These are demonstration results, not clinical findings. The synthetic
generator makes positive examples more likely to have higher pain severity,
more encounters and medications, lower quality-of-life measures, and related
feature shifts. That built-in relationship makes the target learnable. The
MLP then benefits from balanced training rows, training-only standardization,
and 20 epochs of Adam optimization. ROC-AUC measures ranking quality across
thresholds, so `0.917` means the MLP generally ranks positive examples above
negative examples; it does not mean 91.7% of classifications are correct.
