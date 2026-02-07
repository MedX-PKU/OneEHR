from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from oneehr.config.schema import ExperimentConfig
from oneehr.utils.imports import require_torch
from oneehr.utils.io import as_jsonable, ensure_dir, sha256_lines, write_json


def write_dl_artifacts(
    *,
    out_dir: Path,
    model,
    cfg: ExperimentConfig,
    feature_columns: list[str],
    code_vocab: list[str] | None,
) -> None:
    """Persist a trained DL model in a reproducible way.

    Writes:
      - state_dict.ckpt: torch checkpoint with model weights
      - model_meta.json: metadata required to rebuild the model + align inputs
    """

    torch = require_torch()
    out_dir = ensure_dir(out_dir)

    ckpt_path = out_dir / "state_dict.ckpt"
    torch.save(model.state_dict(), ckpt_path)

    model_cfg = getattr(cfg.model, cfg.model.name)

    meta = {
        "schema_version": 1,
        "model": {
            "name": cfg.model.name,
            "hyperparams": as_jsonable(asdict(model_cfg)) if hasattr(model_cfg, "__dataclass_fields__") else {},
        },
        "task": {"kind": str(cfg.task.kind), "prediction_mode": str(cfg.task.prediction_mode)},
        "dataset": {
            "time_col": "event_time",
            "patient_id_col": "patient_id",
            "event_time_col": "event_time",
        },
        "input": {
            "input_dim": int(len(feature_columns)),
            "static_dim": 0,
            "static_feature_columns": None,
            "static_feature_columns_sha256": None,
            "feature_columns": list(feature_columns),
            "code_vocab": None if code_vocab is None else list(code_vocab),
            "feature_columns_sha256": sha256_lines(list(feature_columns)),
            "code_vocab_sha256": None if code_vocab is None else sha256_lines(list(code_vocab)),
        },
        "preprocess": {
            "bin_size": str(cfg.preprocess.bin_size),
            "top_k_codes": None if cfg.preprocess.top_k_codes is None else int(cfg.preprocess.top_k_codes),
        },
    }

    write_json(out_dir / "model_meta.json", meta)
