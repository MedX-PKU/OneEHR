# Training

`oneehr train` fits one or more models, optionally runs grid search, evaluates on each split, and writes structured metrics and predictions.

---

## Basic usage

```bash
uv run oneehr train --config experiment.toml
```

Use `--force` to overwrite an existing run directory:

```bash
uv run oneehr train --config experiment.toml --force
```

---

## Multi-model training

Configure multiple models in a single experiment using `[[models]]` (TOML array of tables):

```toml
[[models]]
name = "gru"

[[models]]
name = "xgboost"

[[models]]
name = "lstm"
```

Each model is trained independently on the same splits. Results are organized per model under `models/{model_name}/`.

---

## Training flow

For each model and each split:

1. **Mask patients** into train/val/test by patient_id
2. **Apply post-split pipeline** (standardize, impute, etc.) -- fit on train only
3. **Train the model**:
    - Tabular models (XGBoost, CatBoost, etc.): fit directly on the 2D feature matrix
    - DL models: build padded 3D tensors `(batch, time, features)`, train with the unified trainer
4. **Evaluate** on val and test sets, compute metrics
5. **Calibrate** probabilities if `[calibration]` is enabled (binary tasks only)
6. **Save** model checkpoint, metrics, and optionally predictions

---

## The unified DL trainer

Deep learning models are trained by a shared trainer that handles:

- **AdamW optimizer** with configurable learning rate, weight decay, and gradient clipping
- **Early stopping** on a monitored metric (`val_loss` by default)
- **Best-state checkpointing** in memory (restores best weights after training)
- **Masked loss** for time-level (N-N) prediction: `sum(loss * mask) / sum(mask)`
- **Final refit** on train+val after split-level evaluation (`final_refit = "train_val"`)

```toml
[trainer]
max_epochs = 30
batch_size = 64
lr = 1e-3
early_stopping = true
early_stopping_patience = 5
monitor = "val_loss"
monitor_mode = "min"
final_refit = "train_val"
```

---

## Metrics

After training, per-split metrics are saved to `models/{model}/{split}/metrics.json`.

### Binary classification

| Metric | Description |
|--------|-------------|
| `auroc` | Area under the ROC curve |
| `auprc` | Area under the precision-recall curve |
| `f1` | F1 score at optimal threshold |
| `accuracy` | Accuracy at optimal threshold |
| `precision` | Precision at optimal threshold |
| `recall` | Recall at optimal threshold |

### Regression

| Metric | Description |
|--------|-------------|
| `rmse` | Root mean squared error |
| `mae` | Mean absolute error |
| `r2` | R-squared |

---

## Summary outputs

After all models and splits complete:

| File | Description |
|------|-------------|
| `summary.json` | Per-model, per-split metrics |
| `hpo_best.csv` | Best HPO config per model (when HPO is enabled) |

---

## Custom loss function

Provide a custom loss factory via Python callable reference:

```toml
[trainer]
loss_fn = "path/to/custom_loss.py:make_loss"
```

The function signature: `make_loss(task: TaskConfig) -> torch.nn.Module`

---

## Bootstrap test evaluation

Enable bootstrap evaluation for confidence intervals:

```toml
[trainer]
bootstrap_test = true
bootstrap_n = 200
```
