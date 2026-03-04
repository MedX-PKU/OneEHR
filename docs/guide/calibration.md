# Calibration

Post-hoc probability calibration adjusts model outputs so predicted probabilities better reflect true event rates. OneEHR supports calibration for **binary classification** tasks only.

---

## Enabling calibration

```toml
[calibration]
enabled = true
method = "temperature"
source = "val"
threshold_strategy = "f1"
use_calibrated = true
```

---

## Methods

### Temperature scaling

Learns a single temperature parameter `T` such that calibrated logits = `logits / T`. Optimized via gradient descent on negative log-likelihood (max 200 iterations).

```toml
[calibration]
method = "temperature"
```

Temperature scaling preserves the ranking of predictions (same AUC) while improving calibration (lower Brier score, lower log-loss).

### Platt scaling

Fits two parameters `a` and `b`: calibrated probability = `sigmoid(a * logit + b)`. Optimized via gradient descent with optional L2 regularization.

```toml
[calibration]
method = "platt"
```

Platt scaling is more flexible than temperature scaling but has a higher risk of overfitting on small validation sets.

---

## Threshold selection

After calibration, OneEHR selects an optimal classification threshold. The `threshold_strategy` parameter controls how:

| Strategy | Description |
|----------|-------------|
| `f1` | Maximize F1 score on the calibration source (default) |

The threshold is found via a cumulative TP/FP scan over the sorted probabilities.

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable calibration |
| `method` | `str` | `"temperature"` | Calibration method: `temperature` or `platt` |
| `source` | `str` | `"val"` | Data source for fitting the calibrator |
| `threshold_strategy` | `str` | `"f1"` | Threshold selection strategy |
| `use_calibrated` | `bool` | `true` | Use calibrated probabilities for threshold selection and downstream outputs |

---

## Calibration metrics

When calibration is enabled, additional metrics are included in `metrics.json`:

| Metric | Description |
|--------|-------------|
| `cal_log_loss` | Log-loss after calibration |
| `cal_brier` | Brier score after calibration |
| `threshold` | Selected classification threshold |
