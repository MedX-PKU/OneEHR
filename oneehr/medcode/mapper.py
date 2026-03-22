"""CodeMapper: unified interface for mapping event codes in dynamic tables.

Integrates ICD grouping, CCS mapping, and ATC hierarchy into the OneEHR
preprocessing pipeline. Can be used to aggregate codes by ontology group
before binning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from oneehr.medcode.icd import ICD9, ICD10
from oneehr.medcode.ccs import CCSGrouper
from oneehr.medcode.atc import ATCHierarchy


@dataclass
class CodeMapper:
    """Map event codes to higher-level groups for dimensionality reduction.

    Example usage::

        mapper = CodeMapper()
        mapper.add_icd_chapter_mapping(version=9)
        mapped_events = mapper.apply(events_df)

    Parameters
    ----------
    mappings : dict
        Custom code → group mappings. Applied first.
    """

    mappings: dict[str, str] = field(default_factory=dict)
    _prefix_rules: list[tuple[str, object]] = field(default_factory=list)

    def add_mapping(self, from_code: str, to_group: str) -> None:
        """Add a single code → group mapping."""
        self.mappings[from_code] = to_group

    def add_icd_chapter_mapping(self, *, version: int = 9, prefix: str = "DX_") -> None:
        """Map ICD diagnosis codes to their chapter names.

        Expects codes in the dynamic table to be prefixed (e.g., ``DX_4019``).
        """
        icd = ICD9 if version == 9 else ICD10

        def _map_fn(code: str) -> str:
            if not code.startswith(prefix):
                return code
            raw = code[len(prefix):]
            chapter = icd.chapter(raw)
            return f"{prefix}CHAPTER_{chapter.replace(' ', '_').replace('/', '_')}"

        self._prefix_rules.append((prefix, _map_fn))

    def add_icd_category_mapping(self, *, version: int = 9, prefix: str = "DX_") -> None:
        """Map ICD diagnosis codes to their 3-digit categories."""
        icd = ICD9 if version == 9 else ICD10

        def _map_fn(code: str) -> str:
            if not code.startswith(prefix):
                return code
            raw = code[len(prefix):]
            cat = icd.category(raw)
            return f"{prefix}CAT_{cat}"

        self._prefix_rules.append((prefix, _map_fn))

    def add_ccs_mapping(self, grouper: CCSGrouper, *, prefix: str = "DX_") -> None:
        """Map ICD codes to CCS categories using a loaded CCSGrouper."""

        def _map_fn(code: str) -> str:
            if not code.startswith(prefix):
                return code
            raw = code[len(prefix):]
            ccs = grouper.group(raw)
            if ccs is not None:
                label = grouper.label(ccs) or ccs
                return f"{prefix}CCS_{ccs}_{label.replace(' ', '_')}"
            return code

        self._prefix_rules.append((prefix, _map_fn))

    def add_atc_mapping(
        self, hierarchy: ATCHierarchy, *, level: int = 1, prefix: str = "RX_"
    ) -> None:
        """Map drug codes to ATC groups at the given level."""

        def _map_fn(code: str) -> str:
            if not code.startswith(prefix):
                return code
            raw = code[len(prefix):]
            group = hierarchy.group(raw, level)
            name = hierarchy.group_name(raw, level) or group
            return f"{prefix}ATC{level}_{group}_{name.replace(' ', '_')}"

        self._prefix_rules.append((prefix, _map_fn))

    def map_code(self, code: str) -> str:
        """Map a single code through all registered rules."""
        # Exact match first
        if code in self.mappings:
            return self.mappings[code]
        # Prefix rules
        for prefix, fn in self._prefix_rules:
            if code.startswith(prefix):
                return fn(code)
        return code

    def apply(self, events: pd.DataFrame, code_col: str = "code") -> pd.DataFrame:
        """Apply all mappings to a dynamic events DataFrame.

        Returns a copy with the code column transformed.
        """
        result = events.copy()
        result[code_col] = result[code_col].map(self.map_code)
        return result
