"""Model checkpointing: torch.save for ALL models (DL + ML wrappers)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from oneehr.utils import ensure_dir, write_json


def save_checkpoint(
    *,
    out_dir: Path,
    model: object,
    model_name: str,
    params: dict,
    train_metrics: dict,
    feature_columns: list[str],
    extra_meta: dict | None = None,
) -> None:
    """Save checkpoint.ckpt + meta.json for any model type."""
    out_dir = ensure_dir(out_dir)

    # torch.save works for DL models (nn.Module) and wrapped ML models
    torch.save(model, out_dir / "checkpoint.ckpt")

    meta = {
        "model_name": model_name,
        "params": params,
        "train_metrics": train_metrics,
        "feature_columns": feature_columns,
    }
    if extra_meta:
        meta["extra"] = extra_meta
    write_json(out_dir / "meta.json", meta)


def load_checkpoint(model_dir: Path) -> tuple[object, dict]:
    """Load checkpoint.ckpt and meta.json. Returns (model, meta_dict)."""
    model_dir = Path(model_dir)
    model = torch.load(model_dir / "checkpoint.ckpt", map_location="cpu", weights_only=False)
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    return model, meta
