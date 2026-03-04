# Models Reference

OneEHR ships 18 models: 5 tabular and 13 deep learning. All models are configured via TOML sub-tables under `[model.<name>]`.

---

## Model overview

| Model | Config key | Type | Patient (N-1) | Time (N-N) | Static branch |
|-------|-----------|------|:---:|:---:|:---:|
| XGBoost | `xgboost` | Tabular | Yes | Yes | -- |
| CatBoost | `catboost` | Tabular | Yes | Yes | -- |
| Random Forest | `rf` | Tabular | Yes | Yes | -- |
| Decision Tree | `dt` | Tabular | Yes | Yes | -- |
| GBDT | `gbdt` | Tabular | Yes | Yes | -- |
| GRU | `gru` | DL | Yes | Yes | No |
| LSTM | `lstm` | DL | Yes | Yes | No |
| RNN | `rnn` | DL | Yes | Yes | No |
| Transformer | `transformer` | DL | Yes | Yes | No |
| TCN | `tcn` | DL | Yes | Yes | No |
| MLP | `mlp` | DL | Yes | Yes | No |
| AdaCare | `adacare` | DL | Yes | Yes | No |
| StageNet | `stagenet` | DL | Yes | Yes | No |
| RETAIN | `retain` | DL | Yes | Yes | No |
| ConCare | `concare` | DL | Yes | Yes | Yes |
| GRASP | `grasp` | DL | Yes | Yes | Yes |
| MCGRU | `mcgru` | DL | Yes | Yes | Yes |
| DrAgent | `dragent` | DL | Yes | Yes | Yes |

!!! info "Static branch"
    ConCare, GRASP, MCGRU, and DrAgent accept a separate static feature vector alongside the dynamic sequence. The `static_dim` is derived automatically from the run manifest -- no user configuration needed.

---

## Tabular models

Tabular models flatten the time dimension and operate on a 2D feature matrix. They work with both `patient` and `time` prediction modes.

### XGBoost

```toml
[model.xgboost]
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
[model.catboost]
depth = 6
n_estimators = 500
learning_rate = 0.05
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `depth` | `int` | `6` | Maximum tree depth |
| `n_estimators` | `int` | `500` | Number of boosting iterations |
| `learning_rate` | `float` | `0.05` | Step size shrinkage |

### Random Forest

```toml
[model.rf]
n_estimators = 500
max_depth = null  # unlimited
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_estimators` | `int` | `500` | Number of trees |
| `max_depth` | `int` | `None` | Maximum tree depth (unlimited if not set) |

### Decision Tree

```toml
[model.dt]
max_depth = null  # unlimited
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_depth` | `int` | `None` | Maximum tree depth (unlimited if not set) |

### GBDT

Scikit-learn's `GradientBoostingClassifier` / `GradientBoostingRegressor`.

```toml
[model.gbdt]
n_estimators = 500
learning_rate = 0.05
max_depth = 3
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_estimators` | `int` | `500` | Number of boosting stages |
| `learning_rate` | `float` | `0.05` | Step size shrinkage |
| `max_depth` | `int` | `3` | Maximum tree depth |

---

## Deep learning models

DL models consume 3D tensors `(batch, time_steps, features)` and are trained via the unified trainer with AdamW, early stopping, and gradient clipping.

### GRU

```toml
[model.gru]
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
[model.lstm]
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

Vanilla RNN with configurable nonlinearity and optional bidirectional mode.

```toml
[model.rnn]
hidden_dim = 128
num_layers = 1
dropout = 0.0
bidirectional = false
nonlinearity = "tanh"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | `128` | Hidden state size |
| `num_layers` | `int` | `1` | Number of stacked RNN layers |
| `dropout` | `float` | `0.0` | Dropout between layers |
| `bidirectional` | `bool` | `false` | Enable bidirectional processing |
| `nonlinearity` | `str` | `"tanh"` | Activation: `tanh` or `relu` |

### Transformer

```toml
[model.transformer]
d_model = 128
nhead = 4
num_layers = 2
dim_feedforward = 256
dropout = 0.1
pooling = "last"  # patient mode only
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | `int` | `128` | Model dimension |
| `nhead` | `int` | `4` | Number of attention heads |
| `num_layers` | `int` | `2` | Number of encoder layers |
| `dim_feedforward` | `int` | `256` | FFN inner dimension |
| `dropout` | `float` | `0.1` | Dropout rate |
| `pooling` | `str` | `"last"` | Pooling for patient mode: `last` or `mean` |

### TCN

Temporal Convolutional Network.

```toml
[model.tcn]
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

### MLP

Multi-layer perceptron. Works with both patient and time modes.

```toml
[model.mlp]
hidden_dim = 128
num_layers = 2
dropout = 0.1
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | `128` | Hidden layer size |
| `num_layers` | `int` | `2` | Number of hidden layers |
| `dropout` | `float` | `0.1` | Dropout rate |

### AdaCare

Adaptive feature recalibration with convolutional context.

```toml
[model.adacare]
hidden_dim = 128
kernel_size = 2
kernel_num = 64
r_v = 4
r_c = 4
dropout = 0.5
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | `128` | Hidden dimension |
| `kernel_size` | `int` | `2` | Convolutional kernel size |
| `kernel_num` | `int` | `64` | Number of convolutional kernels |
| `r_v` | `int` | `4` | Value attention reduction ratio |
| `r_c` | `int` | `4` | Context attention reduction ratio |
| `dropout` | `float` | `0.5` | Dropout rate |

### StageNet

Stage-aware recurrent network with dilated convolutions.

```toml
[model.stagenet]
hidden_dim = 384
conv_size = 10
levels = 3
dropconnect = 0.3
dropout = 0.3
dropres = 0.3
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | `384` | Hidden state size |
| `conv_size` | `int` | `10` | Convolutional filter size |
| `levels` | `int` | `3` | Number of dilated conv levels |
| `dropconnect` | `float` | `0.3` | DropConnect rate |
| `dropout` | `float` | `0.3` | Dropout rate |
| `dropres` | `float` | `0.3` | Residual dropout rate |

### RETAIN

Reverse time attention network with two attention mechanisms (visit-level and variable-level).

```toml
[model.retain]
hidden_dim = 128
dropout = 0.1
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | `128` | Hidden state size |
| `dropout` | `float` | `0.1` | Dropout rate |

### ConCare

Context-aware multi-head self-attention with time-aware key-query decay.

```toml
[model.concare]
hidden_dim = 128
num_heads = 4
dropout = 0.1
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | `128` | Hidden dimension |
| `num_heads` | `int` | `4` | Number of attention heads |
| `dropout` | `float` | `0.1` | Dropout rate |

### GRASP

Graph-based representation with cluster-aware attention and learnable prototypes.

```toml
[model.grasp]
hidden_dim = 128
cluster_num = 12
dropout = 0.1
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | `128` | Hidden dimension |
| `cluster_num` | `int` | `12` | Number of learnable prototypes |
| `dropout` | `float` | `0.1` | Dropout rate |

### MCGRU

Multi-channel GRU with a dedicated static feature branch.

```toml
[model.mcgru]
hidden_dim = 128
num_layers = 1
dropout = 0.0
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | `128` | Hidden state size |
| `num_layers` | `int` | `1` | Number of GRU layers |
| `dropout` | `float` | `0.0` | Dropout rate |

!!! info
    MCGRU automatically receives `static_dim` from the run manifest. Provide a `static.csv` in your dataset config to use the static branch.

### DrAgent

Doctor-agent network with separate dynamic and static processing paths.

```toml
[model.dragent]
hidden_dim = 128
dropout = 0.1
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | `int` | `128` | Hidden dimension |
| `dropout` | `float` | `0.1` | Dropout rate |

!!! info
    Like MCGRU, DrAgent automatically receives `static_dim` from the run manifest.
