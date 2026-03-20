# Models Reference

OneEHR ships 6 models: 2 tabular and 4 deep learning. All models are configured via `[[models]]` entries with a `name` and `params` dict.

---

## Model overview

| Model | Config name | Type | Patient (N-1) | Time (N-N) |
|-------|-----------|------|:---:|:---:|
| XGBoost | `xgboost` | Tabular | Yes | Yes |
| CatBoost | `catboost` | Tabular | Yes | Yes |
| GRU | `gru` | DL | Yes | Yes |
| LSTM | `lstm` | DL | Yes | Yes |
| TCN | `tcn` | DL | Yes | Yes |
| Transformer | `transformer` | DL | Yes | Yes |

---

## Tabular models

Tabular models flatten the time dimension and operate on a 2D feature matrix. They work with both `patient` and `time` prediction modes.

### XGBoost

```toml
[[models]]
name = "xgboost"
[models.params]
max_depth = 6
n_estimators = 500
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
reg_lambda = 1.0
min_child_weight = 1.0
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_depth` | `int` | `6` | Maximum tree depth |
| `n_estimators` | `int` | `500` | Number of boosting rounds |
| `learning_rate` | `float` | `0.05` | Step size shrinkage |
| `subsample` | `float` | `0.8` | Row subsampling ratio |
| `colsample_bytree` | `float` | `0.8` | Column subsampling ratio per tree |
| `reg_lambda` | `float` | `1.0` | L2 regularization |
| `min_child_weight` | `float` | `1.0` | Minimum sum of instance weight in a child |

### CatBoost

```toml
[[models]]
name = "catboost"
[models.params]
depth = 6
n_estimators = 500
learning_rate = 0.05
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `depth` | `int` | `6` | Maximum tree depth |
| `n_estimators` | `int` | `500` | Number of boosting iterations |
| `learning_rate` | `float` | `0.05` | Step size shrinkage |

---

## Deep learning models

DL models consume 3D tensors `(batch, time_steps, features)` and are trained via the unified trainer with AdamW, early stopping, and gradient clipping.

### GRU

```toml
[[models]]
name = "gru"
[models.params]
hidden_dim = 128
num_layers = 1
dropout = 0.0
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | `128` | Hidden state size |
| `num_layers` | `int` | `1` | Number of stacked GRU layers |
| `dropout` | `float` | `0.0` | Dropout between layers |

### LSTM

```toml
[[models]]
name = "lstm"
[models.params]
hidden_dim = 128
num_layers = 1
dropout = 0.0
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | `128` | Hidden state size |
| `num_layers` | `int` | `1` | Number of stacked LSTM layers |
| `dropout` | `float` | `0.0` | Dropout between layers |

### TCN

Temporal Convolutional Network.

```toml
[[models]]
name = "tcn"
[models.params]
hidden_dim = 128
num_layers = 2
kernel_size = 3
dropout = 0.1
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | `128` | Channel dimension |
| `num_layers` | `int` | `2` | Number of TCN blocks |
| `kernel_size` | `int` | `3` | Convolutional kernel size |
| `dropout` | `float` | `0.1` | Dropout rate |

### Transformer

```toml
[[models]]
name = "transformer"
[models.params]
d_model = 128
nhead = 4
num_layers = 2
dim_feedforward = 256
dropout = 0.1
pooling = "last"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | `int` | `128` | Model dimension |
| `nhead` | `int` | `4` | Number of attention heads |
| `num_layers` | `int` | `2` | Number of encoder layers |
| `dim_feedforward` | `int` | `256` | FFN inner dimension |
| `dropout` | `float` | `0.1` | Dropout rate |
| `pooling` | `str` | `"last"` | Pooling for patient mode: `last` or `mean` |
