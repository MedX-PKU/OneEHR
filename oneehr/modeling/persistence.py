from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from oneehr.config.schema import ExperimentConfig
from oneehr.utils.imports import optional_import
from oneehr.utils.io import ensure_dir, write_json


def _torch():
    torch = optional_import("torch")
    if torch is None:
        raise ModuleNotFoundError("torch")
    return torch


def _as_jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_as_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _as_jsonable(x) for k, x in v.items()}
    return str(v)


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_lines(lines: list[str]) -> str:
    # Normalize newlines + whitespace to keep hashes stable.
    norm = "\n".join([ln.strip() for ln in lines]) + "\n"
    return _sha256_text(norm)


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

    torch = _torch()
    out_dir = ensure_dir(out_dir)

    ckpt_path = out_dir / "state_dict.ckpt"
    torch.save(model.state_dict(), ckpt_path)

    model_cfg = getattr(cfg.model, cfg.model.name)

    meta = {
        "schema_version": 1,
        "model": {
            "name": cfg.model.name,
            "hyperparams": _as_jsonable(asdict(model_cfg)) if hasattr(model_cfg, "__dataclass_fields__") else {},
        },
        "task": {"kind": str(cfg.task.kind), "prediction_mode": str(cfg.task.prediction_mode)},
        "dataset": {
            "time_col": str(cfg.dataset.dynamic.time_col),
            "patient_id_col": str(cfg.dataset.dynamic.patient_id_col),
            "event_time_col": str(cfg.dataset.dynamic.time_col),
        },
        "input": {
            "input_dim": int(cfg.preprocess.top_k_codes or len(feature_columns)),
            "static_dim": 0,
            "static_feature_columns": None,
            "static_feature_columns_sha256": None,
            "feature_columns": list(feature_columns),
            "code_vocab": None if code_vocab is None else list(code_vocab),
            "feature_columns_sha256": _sha256_lines(list(feature_columns)),
            "code_vocab_sha256": None if code_vocab is None else _sha256_lines(list(code_vocab)),
        },
        "preprocess": {
            "bin_size": str(cfg.preprocess.bin_size),
            "top_k_codes": None if cfg.preprocess.top_k_codes is None else int(cfg.preprocess.top_k_codes),
        },
    }

    write_json(out_dir / "model_meta.json", meta)


def write_static_artifacts(
    *,
    out_dir: Path,
    feature_columns: list[str],
    fitted_postprocess: dict[str, object] | None,
    raw_cols: list[str],
) -> None:
    out_dir = ensure_dir(out_dir)
    write_json(
        out_dir / "static_meta.json",
        {
            "schema_version": 1,
            "raw_cols": list(raw_cols),
            "feature_columns": list(feature_columns),
            "feature_columns_sha256": _sha256_lines(list(feature_columns)),
            "postprocess": None if fitted_postprocess is None else {"pipeline": fitted_postprocess},
        },
    )


def read_dl_meta(run_dir: Path, model_name: str, split_name: str) -> dict[str, Any]:
    meta_path = run_dir / "models" / model_name / split_name / "model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(str(meta_path))
    return json.loads(meta_path.read_text(encoding="utf-8"))
