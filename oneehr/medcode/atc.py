"""Anatomical Therapeutic Chemical (ATC) classification hierarchy.

ATC codes have 5 levels:
  Level 1: A        (Anatomical main group, 14 groups)
  Level 2: A02      (Therapeutic subgroup)
  Level 3: A02B     (Pharmacological subgroup)
  Level 4: A02BC    (Chemical subgroup)
  Level 5: A02BC01  (Chemical substance)
"""

from __future__ import annotations

from pathlib import Path


# Level 1 ATC main groups
_ATC_LEVEL1: dict[str, str] = {
    "A": "Alimentary tract and metabolism",
    "B": "Blood and blood forming organs",
    "C": "Cardiovascular system",
    "D": "Dermatologicals",
    "G": "Genito-urinary system and sex hormones",
    "H": "Systemic hormonal preparations",
    "J": "Anti-infectives for systemic use",
    "L": "Antineoplastic and immunomodulating agents",
    "M": "Musculo-skeletal system",
    "N": "Nervous system",
    "P": "Antiparasitic products",
    "R": "Respiratory system",
    "S": "Sensory organs",
    "V": "Various",
}


class ATCHierarchy:
    """ATC classification with hierarchical grouping.

    Can be used standalone with the built-in level-1 definitions,
    or loaded with a full ATC mapping file for deeper levels.

    Parameters
    ----------
    mapping_file : str or Path, optional
        CSV file with columns ``atc_code`` and ``atc_name``.
    """

    def __init__(self, mapping_file: str | Path | None = None) -> None:
        self._code_to_name: dict[str, str] = dict(_ATC_LEVEL1)
        if mapping_file is not None:
            self.load(mapping_file)

    def load(self, mapping_file: str | Path) -> None:
        """Load an ATC mapping from a CSV file."""
        import pandas as pd

        df = pd.read_csv(mapping_file)
        for _, row in df.iterrows():
            code = str(row.iloc[0]).strip().upper()
            name = str(row.iloc[1]).strip()
            self._code_to_name[code] = name

    @staticmethod
    def level(code: str) -> int:
        """Determine the ATC level (1-5) from code length."""
        code = code.strip().upper()
        length = len(code)
        if length == 1:
            return 1
        if length == 3:
            return 2
        if length == 4:
            return 3
        if length == 5:
            return 4
        if length >= 7:
            return 5
        return 0  # unknown

    @staticmethod
    def parent(code: str, target_level: int = 1) -> str:
        """Return the parent code at the given level."""
        code = code.strip().upper()
        cuts = {1: 1, 2: 3, 3: 4, 4: 5, 5: 7}
        n = cuts.get(target_level, 1)
        return code[:n]

    def name(self, code: str) -> str | None:
        """Return the name for an ATC code."""
        code = code.strip().upper()
        return self._code_to_name.get(code)

    def group(self, code: str, level: int = 1) -> str:
        """Return the parent code at the given level."""
        return self.parent(code, level)

    def group_name(self, code: str, level: int = 1) -> str | None:
        """Return the name of the parent group at the given level."""
        parent_code = self.parent(code, level)
        return self._code_to_name.get(parent_code)

    @property
    def main_groups(self) -> dict[str, str]:
        """Return the 14 ATC level-1 main groups."""
        return dict(_ATC_LEVEL1)

    def __contains__(self, code: str) -> bool:
        return code.strip().upper() in self._code_to_name
