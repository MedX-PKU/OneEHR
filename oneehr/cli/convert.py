"""CLI handler for the ``oneehr convert`` command."""

from __future__ import annotations


def run_convert(dataset: str, raw_dir: str, output_dir: str, *, task: str | None = None) -> None:
    """Convert a raw clinical dataset into OneEHR three-table format."""
    if dataset == "mimic3":
        from oneehr.datasets.mimic3 import MIMIC3Converter

        converter = MIMIC3Converter(raw_dir)
    elif dataset == "mimic4":
        from oneehr.datasets.mimic4 import MIMIC4Converter

        converter = MIMIC4Converter(raw_dir)
    elif dataset == "eicu":
        from oneehr.datasets.eicu import EICUConverter

        converter = EICUConverter(raw_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    paths = converter.save(output_dir, task=task)
    print(f"Converted {dataset} → {output_dir}")
    for name, path in paths.items():
        print(f"  {name}: {path}")
