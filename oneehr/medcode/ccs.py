"""Clinical Classifications Software (CCS) grouping for ICD codes.

CCS reduces thousands of ICD diagnosis/procedure codes into a manageable
number of clinically meaningful categories (~280 single-level categories).
"""

from __future__ import annotations

from pathlib import Path

from oneehr.medcode.icd import normalize_icd


class CCSGrouper:
    """Map ICD codes to CCS categories.

    The grouper must be initialized with a CCS mapping file. AHRQ provides
    these as downloadable CSV files.

    Parameters
    ----------
    mapping_file : str or Path
        Path to the CCS single-level mapping CSV.
        Expected columns: ``icd_code``, ``ccs_category``, ``ccs_label``.
        Alternatively, the AHRQ format with columns:
        ``'ICD-9-CM CODE'``, ``'CCS CATEGORY'``, ``'CCS CATEGORY DESCRIPTION'``.
    """

    def __init__(self, mapping_file: str | Path | None = None) -> None:
        self._code_to_ccs: dict[str, str] = {}
        self._ccs_labels: dict[str, str] = {}
        if mapping_file is not None:
            self.load(mapping_file)

    def load(self, mapping_file: str | Path) -> None:
        """Load a CCS mapping from a CSV file."""
        import pandas as pd

        df = pd.read_csv(mapping_file)
        # Normalize column names
        cols = {c.strip().strip("'"): c for c in df.columns}

        # Detect format
        if "icd_code" in cols:
            code_col = cols["icd_code"]
            cat_col = cols["ccs_category"]
            label_col = cols.get("ccs_label", cols.get("ccs_category_description"))
        elif "ICD-9-CM CODE" in cols:
            code_col = cols["ICD-9-CM CODE"]
            cat_col = cols["CCS CATEGORY"]
            label_col = cols.get("CCS CATEGORY DESCRIPTION")
        else:
            # Generic: first 3 columns
            code_col, cat_col = df.columns[0], df.columns[1]
            label_col = df.columns[2] if len(df.columns) > 2 else None

        for _, row in df.iterrows():
            code = normalize_icd(str(row[code_col]).strip().strip("'"))
            cat = str(row[cat_col]).strip().strip("'")
            self._code_to_ccs[code] = cat
            if label_col is not None:
                label = str(row[label_col]).strip().strip("'")
                self._ccs_labels[cat] = label

    def group(self, icd_code: str) -> str | None:
        """Return the CCS category for an ICD code, or None if not found."""
        return self._code_to_ccs.get(normalize_icd(icd_code))

    def label(self, ccs_category: str) -> str | None:
        """Return the human-readable label for a CCS category."""
        return self._ccs_labels.get(str(ccs_category))

    def group_with_label(self, icd_code: str) -> tuple[str | None, str | None]:
        """Return (ccs_category, ccs_label) for an ICD code."""
        cat = self.group(icd_code)
        if cat is None:
            return None, None
        return cat, self._ccs_labels.get(cat)

    @property
    def categories(self) -> list[str]:
        """Return all known CCS categories."""
        return sorted(set(self._code_to_ccs.values()))

    def __len__(self) -> int:
        return len(self._code_to_ccs)

    def __contains__(self, icd_code: str) -> bool:
        return normalize_icd(icd_code) in self._code_to_ccs
