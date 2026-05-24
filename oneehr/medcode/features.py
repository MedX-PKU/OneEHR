"""Feature-name medical code resolution.

These helpers normalize OneEHR feature names into stable medical-code
identities. They are shared by preprocessing, KG adapters, and model-side
artifact builders.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from oneehr.medcode.atc import ATCHierarchy
from oneehr.medcode.icd import ICD9, ICD10

_ATC_RE = re.compile(r"^[A-Z]\d{2}[A-Z]{0,2}\d{0,2}$")


@dataclass(frozen=True)
class ParsedFeatureCode:
    """Structured identity for a feature that represents a medical code."""

    namespace: str
    raw_code: str
    normalized_code: str
    system: str | None
    kind: str
    feature_name: str

    @property
    def is_medical_code(self) -> bool:
        return self.kind in {"diagnosis", "procedure", "drug"}


def feature_base_name(name: str) -> str:
    """Return the stable pre-encoding feature name."""

    raw = str(name)
    if raw.startswith("num__"):
        raw = raw[5:]
    elif raw.startswith("cat__"):
        raw = raw[5:]
    if "__" in raw:
        raw = raw.split("__", 1)[0]
    return raw


def parse_feature_code(name: str) -> ParsedFeatureCode:
    """Parse a OneEHR feature name into a stable medical-code identity."""

    feature = feature_base_name(name)
    parts = feature.split("_")
    namespace = parts[0].upper() if parts else ""

    if namespace in {"DX", "PROC"}:
        kind = "diagnosis" if namespace == "DX" else "procedure"
        if len(parts) >= 3 and parts[1].upper() in {"ICD9", "ICD10"}:
            system = parts[1].upper()
            raw_code = "_".join(parts[2:])
        else:
            raw_code = "_".join(parts[1:]) if len(parts) > 1 else ""
            system = _infer_icd_system(raw_code)
        normalized = _normalize_icd(system, raw_code)
        return ParsedFeatureCode(namespace, raw_code, normalized, system, kind, feature)

    if namespace == "RX":
        raw_code = "_".join(parts[1:]) if len(parts) > 1 else ""
        system = _infer_drug_system(raw_code)
        normalized = _normalize_drug_code(raw_code, system)
        return ParsedFeatureCode(namespace, raw_code, normalized, system, "drug", feature)

    return ParsedFeatureCode(namespace, "_".join(parts[1:]), feature.upper(), None, "other", feature)


def feature_code_aliases(name: str) -> set[str]:
    """Aliases accepted when matching external KG node IDs to feature groups."""

    parsed = parse_feature_code(name)
    aliases = {parsed.feature_name, parsed.feature_name.upper(), feature_base_name(name), str(name)}
    if parsed.normalized_code:
        aliases.add(parsed.normalized_code)
    if parsed.raw_code:
        aliases.add(parsed.raw_code)
        aliases.add(parsed.raw_code.upper())
    if parsed.system and parsed.normalized_code:
        aliases.add(f"{parsed.system}:{parsed.normalized_code}")
        aliases.add(f"{parsed.system}::{parsed.normalized_code}")
        aliases.add(f"{parsed.namespace}_{parsed.system}_{parsed.normalized_code}")
    if parsed.namespace and parsed.normalized_code:
        aliases.add(f"{parsed.namespace}_{parsed.normalized_code}")
    return {alias for alias in aliases if alias}


def ontology_bucket(name: str, ontology: str) -> str | None:
    """Return a coarse ontology bucket for KG adjacency construction."""

    if ontology == "none":
        return None

    parsed = parse_feature_code(name)
    if parsed.kind in {"diagnosis", "procedure"} and ontology in {"auto", "icd"}:
        if parsed.system == "ICD10":
            category = ICD10.category(parsed.normalized_code)
            chapter = ICD10.chapter(parsed.normalized_code)
        elif parsed.system == "ICD9":
            category = ICD9.category(parsed.normalized_code)
            chapter = ICD9.chapter(parsed.normalized_code)
        else:
            return f"{parsed.namespace}::{parsed.feature_name}"
        return f"{parsed.system}::{parsed.namespace}::{category}::{chapter}"

    if parsed.kind == "drug" and ontology in {"auto", "atc"}:
        if parsed.system != "ATC":
            return None if ontology == "atc" else f"RX::{parsed.system or 'RAW'}"
        atc = ATCHierarchy()
        group = atc.group(parsed.normalized_code, level=1)
        return f"ATC::{group}::{atc.group_name(parsed.normalized_code, level=1) or 'unknown'}"

    if ontology == "auto" and parsed.namespace:
        return f"LEX::{parsed.namespace}"
    return None


def _infer_icd_system(code: str) -> str | None:
    code = str(code).strip().upper()
    if not code:
        return None
    if code[:1].isalpha():
        return "ICD10"
    if code[:1].isdigit():
        return "ICD9"
    return None


def _normalize_icd(system: str | None, code: str) -> str:
    if system == "ICD10":
        return ICD10.normalize(code)
    if system == "ICD9":
        return ICD9.normalize(code)
    return str(code).strip().upper()


def _infer_drug_system(code: str) -> str | None:
    code = str(code).strip().upper()
    if not code:
        return None
    if code.startswith("ATC_"):
        return "ATC"
    if code.startswith("NDC_"):
        return "NDC"
    if code.startswith("RXNORM_"):
        return "RXNORM"
    if _ATC_RE.match(code) and ATCHierarchy.level(code) > 0:
        return "ATC"
    return None


def _normalize_drug_code(code: str, system: str | None) -> str:
    code = str(code).strip().upper()
    if system in {"ATC", "NDC", "RXNORM"} and "_" in code:
        return code.split("_", 1)[1]
    return code
