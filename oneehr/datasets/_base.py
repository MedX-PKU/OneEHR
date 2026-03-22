"""Base class for dataset converters."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ConvertedDataset:
    """Result of a dataset conversion."""

    dynamic: pd.DataFrame  # patient_id, event_time, code, value
    static: pd.DataFrame  # patient_id, ...
    labels: dict[str, pd.DataFrame]  # task_name -> label DataFrame


class BaseConverter(abc.ABC):
    """Convert a raw clinical dataset into OneEHR's three-table format."""

    def __init__(self, raw_dir: str | Path) -> None:
        self.raw_dir = Path(raw_dir)
        if not self.raw_dir.is_dir():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_dir}")

    @abc.abstractmethod
    def convert(self) -> ConvertedDataset:
        """Run the conversion and return a ConvertedDataset."""

    def save(self, output_dir: str | Path, task: str | None = None) -> dict[str, Path]:
        """Convert and save to disk as CSV files.

        Parameters
        ----------
        output_dir : path
            Directory to write the output files.
        task : str, optional
            Which label task to save. If None, saves all tasks as
            ``label_{task}.csv``.

        Returns
        -------
        dict mapping file name to path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self.convert()
        paths: dict[str, Path] = {}

        # dynamic
        p = output_dir / "dynamic.csv"
        result.dynamic.to_csv(p, index=False)
        paths["dynamic"] = p

        # static
        p = output_dir / "static.csv"
        result.static.to_csv(p, index=False)
        paths["static"] = p

        # labels
        if task is not None:
            if task not in result.labels:
                available = list(result.labels.keys())
                raise ValueError(f"Task {task!r} not found. Available: {available}")
            p = output_dir / "label.csv"
            result.labels[task].to_csv(p, index=False)
            paths["label"] = p
        else:
            for tname, ldf in result.labels.items():
                p = output_dir / f"label_{tname}.csv"
                ldf.to_csv(p, index=False)
                paths[f"label_{tname}"] = p

        return paths

    def _read_csv(self, name: str, **kwargs) -> pd.DataFrame:
        """Read a CSV file from the raw directory."""
        path = self.raw_dir / name
        if not path.exists():
            # Try .csv.gz
            gz = self.raw_dir / f"{name}.gz"
            if gz.exists():
                path = gz
            else:
                raise FileNotFoundError(f"Expected file not found: {path}")
        return pd.read_csv(path, **kwargs)
