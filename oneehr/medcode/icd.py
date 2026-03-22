"""ICD-9 and ICD-10 code utilities.

Provides code normalization, chapter-level grouping, and basic
cross-mapping between ICD-9-CM and ICD-10-CM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ICDCode:
    """A parsed ICD code with version and components."""

    version: int  # 9 or 10
    code: str  # normalized code (no dots)
    chapter: str  # high-level category


# ICD-9-CM chapter ranges
_ICD9_CHAPTERS: list[tuple[str, str, str]] = [
    ("001", "139", "Infectious diseases"),
    ("140", "239", "Neoplasms"),
    ("240", "279", "Endocrine/metabolic"),
    ("280", "289", "Blood diseases"),
    ("290", "319", "Mental disorders"),
    ("320", "389", "Nervous system"),
    ("390", "459", "Circulatory system"),
    ("460", "519", "Respiratory system"),
    ("520", "579", "Digestive system"),
    ("580", "629", "Genitourinary system"),
    ("630", "679", "Pregnancy"),
    ("680", "709", "Skin diseases"),
    ("710", "739", "Musculoskeletal"),
    ("740", "759", "Congenital anomalies"),
    ("760", "779", "Perinatal conditions"),
    ("780", "799", "Symptoms/signs"),
    ("800", "999", "Injury/poisoning"),
    ("E", "E", "External causes"),
    ("V", "V", "Supplementary factors"),
]

# ICD-10-CM chapter ranges (letter-based)
_ICD10_CHAPTERS: dict[str, str] = {
    "A": "Infectious diseases",
    "B": "Infectious diseases",
    "C": "Neoplasms",
    "D": "Blood/immune diseases",
    "E": "Endocrine/metabolic",
    "F": "Mental disorders",
    "G": "Nervous system",
    "H": "Eye and ear",
    "I": "Circulatory system",
    "J": "Respiratory system",
    "K": "Digestive system",
    "L": "Skin diseases",
    "M": "Musculoskeletal",
    "N": "Genitourinary system",
    "O": "Pregnancy",
    "P": "Perinatal conditions",
    "Q": "Congenital anomalies",
    "R": "Symptoms/signs",
    "S": "Injury",
    "T": "Injury/poisoning",
    "U": "Special purposes",
    "V": "External causes",
    "W": "External causes",
    "X": "External causes",
    "Y": "External causes",
    "Z": "Health status factors",
}


def normalize_icd(code: str) -> str:
    """Remove dots and whitespace, uppercase."""
    return re.sub(r"[\s.]", "", str(code)).upper()


def _icd9_chapter(code: str) -> str:
    """Determine ICD-9 chapter from normalized code."""
    if code.startswith("E"):
        return "External causes"
    if code.startswith("V"):
        return "Supplementary factors"
    # Numeric codes
    try:
        num = int(code[:3])
    except ValueError:
        return "Unknown"
    for lo, hi, name in _ICD9_CHAPTERS:
        if lo.isdigit() and hi.isdigit():
            if int(lo) <= num <= int(hi):
                return name
    return "Unknown"


def _icd10_chapter(code: str) -> str:
    """Determine ICD-10 chapter from normalized code."""
    if not code:
        return "Unknown"
    return _ICD10_CHAPTERS.get(code[0], "Unknown")


class ICD9:
    """ICD-9-CM code utilities."""

    @staticmethod
    def normalize(code: str) -> str:
        return normalize_icd(code)

    @staticmethod
    def chapter(code: str) -> str:
        return _icd9_chapter(normalize_icd(code))

    @staticmethod
    def category(code: str) -> str:
        """Return the 3-digit category (first 3 chars)."""
        norm = normalize_icd(code)
        if norm.startswith(("E", "V")):
            return norm[:4]  # E-codes use 4 chars
        return norm[:3]

    @staticmethod
    def parse(code: str) -> ICDCode:
        norm = normalize_icd(code)
        return ICDCode(version=9, code=norm, chapter=_icd9_chapter(norm))


class ICD10:
    """ICD-10-CM code utilities."""

    @staticmethod
    def normalize(code: str) -> str:
        return normalize_icd(code)

    @staticmethod
    def chapter(code: str) -> str:
        return _icd10_chapter(normalize_icd(code))

    @staticmethod
    def category(code: str) -> str:
        """Return the 3-character category."""
        return normalize_icd(code)[:3]

    @staticmethod
    def parse(code: str) -> ICDCode:
        norm = normalize_icd(code)
        return ICDCode(version=10, code=norm, chapter=_icd10_chapter(norm))


# --- Cross-mapping ---
# NOTE: Full ICD-9 ↔ ICD-10 mapping requires the CMS GEM files.
# This module provides a programmatic interface; users supply the mapping data.

_icd9_to_10: dict[str, list[str]] = {}
_icd10_to_9: dict[str, list[str]] = {}


def load_gem_mapping(filepath: str, *, direction: str = "9to10") -> None:
    """Load a CMS General Equivalence Mapping (GEM) file.

    Parameters
    ----------
    filepath : str
        Path to the GEM text file (space/tab separated: source target flags).
    direction : str
        "9to10" or "10to9".
    """
    mapping = _icd9_to_10 if direction == "9to10" else _icd10_to_9

    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                src, tgt = parts[0], parts[1]
                mapping.setdefault(src, []).append(tgt)


def icd9_to_icd10(code: str) -> list[str]:
    """Map an ICD-9 code to ICD-10 codes using loaded GEM data."""
    norm = normalize_icd(code)
    return _icd9_to_10.get(norm, [])


def icd10_to_icd9(code: str) -> list[str]:
    """Map an ICD-10 code to ICD-9 codes using loaded GEM data."""
    norm = normalize_icd(code)
    return _icd10_to_9.get(norm, [])
