# Models Reference

OneEHR ships 15 models: 2 tabular and 13 deep learning. All models are configured via `[[models]]` entries with a `name` and `params` dict.

---

## Model overview

| Model | Config name | Type | Patient (N-1) | Time (N-N) | Static branch |
|-------|-----------|------|:---:|:---:|:---:|
| XGBoost | `xgboost` | Tabular | Yes | Yes | N/A |
| CatBoost | `catboost` | Tabular | Yes | Yes | N/A |
| GRU | `gru` | DL | Yes | Yes | No |
| LSTM | `lstm` | DL | Yes | Yes | No |
| RNN | `rnn` | DL | Yes | Yes | No |
| TCN | `tcn` | DL | Yes | Yes | No |
| Transformer | `transformer` | DL | Yes | Yes | No |
| MLP | `mlp` | DL | Yes | Yes | No |
| AdaCare | `adacare` | DL | Yes | Yes | No |
| StageNet | `stagenet` | DL | Yes | Yes | No |
| RETAIN | `retain` | DL | Yes | Yes | No |
| ConCare | `concare` | DL | Yes | Yes | Yes |
| GRASP | `grasp` | DL | Yes | Yes | Yes |
| MCGRU | `mcgru` | DL | Yes | Yes | Yes |
| DrAgent | `dragent` | DL | Yes | Yes | Yes |

Models with a **static branch** automatically receive patient-level static features as a separate input tensor when `static.csv` is provided. The `static_dim` parameter is auto-detected from the static feature count.

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

## Recurrent models

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

### RNN

Vanilla (Elman) recurrent network.

```toml
[[models]]
name = "rnn"
[models.params]
hidden_dim = 128
num_layers = 1
dropout = 0.0
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | `128` | Hidden state size |
| `num_layers` | `int` | `1` | Number of stacked RNN layers |
| `dropout` | `float` | `0.0` | Dropout between layers |

---

## Non-recurrent models

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

### MLP

Feedforward network operating on the last time step.

```toml
[[models]]
name = "mlp"
[models.params]
hidden_dim = 128
dropout = 0.0
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | `128` | Hidden layer size |
| `dropout` | `float` | `0.0` | Dropout rate |

---

## EHR-specialised models

### AdaCare

Adaptive clinical feature calibration with dilated convolutions.

> Liantao Ma et al. AdaCare: Explainable Clinical Health Status Representation Learning via Scale-Adaptive Feature Extraction and Recalibration. AAAI 2020.

```toml
[[models]]
name = "adacare"
[models.params]
hidden_dim = 128
kernel_size = 2
kernel_num = 64
dropout = 0.5
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | `128` | Hidden state size |
| `kernel_size` | `int` | `2` | Dilated convolution kernel size |
| `kernel_num` | `int` | `64` | Number of convolution channels |
| `dropout` | `float` | `0.5` | Dropout rate |

### StageNet

Stage-aware LSTM with stage-adaptive convolution.

> Junyi Gao et al. StageNet: Stage-Aware Neural Network for Health Risk Prediction. WWW 2020.

```toml
[[models]]
name = "stagenet"
[models.params]
chunk_size = 128
levels = 3
conv_size = 10
dropout = 0.3
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | `int` | `128` | Stage-aware hidden chunk size |
| `levels` | `int` | `3` | Number of hierarchical levels |
| `conv_size` | `int` | `10` | Convolution window size |
| `dropout` | `float` | `0.3` | Dropout rate |

### RETAIN

Reverse Time Attention Network with interpretable alpha and beta attention.

> Edward Choi et al. RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism. NIPS 2016.

```toml
[[models]]
name = "retain"
[models.params]
hidden_dim = 128
dropout = 0.5
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | `128` | Hidden state size for both alpha and beta GRUs |
| `dropout` | `float` | `0.5` | Dropout rate |

### ConCare

Context-aware temporal attention with self-attention over feature embeddings. Supports a dedicated static branch.

> Liantao Ma et al. ConCare: Personalized Clinical Feature Embedding via Capturing the Healthcare Context. AAAI 2020.

```toml
[[models]]
name = "concare"
[models.params]
hidden_dim = 128
num_heads = 4
dropout = 0.5
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | `128` | Hidden state size |
| `num_heads` | `int` | `4` | Multi-head attention heads |
| `dropout` | `float` | `0.5` | Dropout rate |
| `static_dim` | `int` | auto | Auto-detected from static features |

### GRASP

GRU with K-means clustering and graph convolutional layers. Supports a dedicated static branch.

> Chaohe Zhang et al. GRASP: Generic Framework for Health Status Representation Learning Based on Incorporating Knowledge from Similar Patients. AAAI 2021.

```toml
[[models]]
name = "grasp"
[models.params]
hidden_dim = 128
cluster_num = 12
dropout = 0.5
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | `128` | Hidden state size |
| `cluster_num` | `int` | `12` | Number of patient clusters |
| `dropout` | `float` | `0.5` | Dropout rate |
| `static_dim` | `int` | auto | Auto-detected from static features |

### MCGRU

Multi-Channel GRU with per-feature GRU cells. Supports a dedicated static branch.

```toml
[[models]]
name = "mcgru"
[models.params]
hidden_dim = 32
feat_dim = 8
dropout = 0.0
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | `32` | Per-channel GRU hidden size |
| `feat_dim` | `int` | `8` | Feature embedding dimension |
| `dropout` | `float` | `0.0` | Dropout rate |
| `static_dim` | `int` | auto | Auto-detected from static features |

### DrAgent

Dual-agent reinforcement learning action selection for clinical prediction. Supports a dedicated static branch.

> Junyi Gao et al. Dr. Agent: Clinical predictive model via mimicked second opinions. JAMIA.

```toml
[[models]]
name = "dragent"
[models.params]
hidden_dim = 128
n_actions = 10
n_units = 64
dropout = 0.5
lamda = 0.5
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | `128` | GRU hidden state size |
| `n_actions` | `int` | `10` | Number of agent actions |
| `n_units` | `int` | `64` | Agent MLP hidden size |
| `dropout` | `float` | `0.5` | Dropout rate |
| `lamda` | `float` | `0.5` | Mixing weight for agent-selected vs current hidden state |
| `static_dim` | `int` | auto | Auto-detected from static features |
