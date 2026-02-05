from __future__ import annotations

from pathlib import Path


def test_cli_smoke_preprocess_and_train(tmp_path: Path) -> None:
    import subprocess

    cfg = Path(__file__).resolve().parents[1] / "examples" / "experiment.toml"
    out = tmp_path / "out"

    # Patch output root via env (simple override by copying config).
    cfg_text = cfg.read_text(encoding="utf-8").replace('root = "logs"', f'root = "{out}"')
    cfg2 = tmp_path / "cfg.toml"
    cfg2.write_text(cfg_text, encoding="utf-8")

    subprocess.check_call(["oneehr", "preprocess", "--config", str(cfg2)])
    subprocess.check_call(["oneehr", "train", "--config", str(cfg2), "--force"])
    # train delegates to benchmark, which may skip some models if optional deps are missing.
    subprocess.check_call(["oneehr", "preprocess", "--config", str(cfg2)])
    subprocess.check_call(["oneehr", "test", "--config", str(cfg2), "--force"])

    run_root = out / "example"
    assert (run_root / "summary.json").exists()
    assert (run_root / "hpo_best.csv").exists()
    assert (run_root / "test_runs").exists()
