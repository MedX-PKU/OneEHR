"""Tests for medical code ontology module."""

import pytest


def test_icd9_normalize():
    from oneehr.medcode.icd import ICD9
    assert ICD9.normalize("401.9") == "4019"
    assert ICD9.normalize(" E880.0 ") == "E8800"


def test_icd9_chapter():
    from oneehr.medcode.icd import ICD9
    assert ICD9.chapter("401.9") == "Circulatory system"
    assert ICD9.chapter("250.00") == "Endocrine/metabolic"
    assert ICD9.chapter("E880") == "External causes"
    assert ICD9.chapter("V10") == "Supplementary factors"


def test_icd9_category():
    from oneehr.medcode.icd import ICD9
    assert ICD9.category("401.9") == "401"
    assert ICD9.category("E880.0") == "E880"


def test_icd10_chapter():
    from oneehr.medcode.icd import ICD10
    assert ICD10.chapter("I10") == "Circulatory system"
    assert ICD10.chapter("E11.9") == "Endocrine/metabolic"
    assert ICD10.chapter("Z00") == "Health status factors"


def test_icd10_category():
    from oneehr.medcode.icd import ICD10
    assert ICD10.category("I10.0") == "I10"
    assert ICD10.category("E11.9") == "E11"


def test_icd_parse():
    from oneehr.medcode.icd import ICD9, ICD10
    code9 = ICD9.parse("401.9")
    assert code9.version == 9
    assert code9.code == "4019"
    assert code9.chapter == "Circulatory system"

    code10 = ICD10.parse("I10")
    assert code10.version == 10
    assert code10.chapter == "Circulatory system"


def test_gem_mapping_roundtrip():
    from oneehr.medcode.icd import icd9_to_icd10, icd10_to_icd9, _icd9_to_10, _icd10_to_9

    # Before loading, mappings are empty
    assert icd9_to_icd10("4019") == []
    assert icd10_to_icd9("I10") == []


def test_atc_level():
    from oneehr.medcode.atc import ATCHierarchy
    assert ATCHierarchy.level("A") == 1
    assert ATCHierarchy.level("A02") == 2
    assert ATCHierarchy.level("A02B") == 3
    assert ATCHierarchy.level("A02BC") == 4
    assert ATCHierarchy.level("A02BC01") == 5


def test_atc_parent():
    from oneehr.medcode.atc import ATCHierarchy
    assert ATCHierarchy.parent("A02BC01", target_level=1) == "A"
    assert ATCHierarchy.parent("A02BC01", target_level=2) == "A02"
    assert ATCHierarchy.parent("A02BC01", target_level=3) == "A02B"


def test_atc_main_groups():
    from oneehr.medcode.atc import ATCHierarchy
    h = ATCHierarchy()
    groups = h.main_groups
    assert "A" in groups
    assert "N" in groups
    assert len(groups) == 14


def test_ccs_grouper_basic():
    from oneehr.medcode.ccs import CCSGrouper
    g = CCSGrouper()
    assert len(g) == 0
    assert g.group("4019") is None


def test_code_mapper_icd_chapter():
    from oneehr.medcode.mapper import CodeMapper

    mapper = CodeMapper()
    mapper.add_icd_chapter_mapping(version=9, prefix="DX_")

    result = mapper.map_code("DX_4019")
    assert "CHAPTER" in result
    assert "Circulatory" in result

    # Non-DX codes pass through
    assert mapper.map_code("LAB_50801") == "LAB_50801"


def test_code_mapper_icd_category():
    from oneehr.medcode.mapper import CodeMapper

    mapper = CodeMapper()
    mapper.add_icd_category_mapping(version=9, prefix="DX_")

    result = mapper.map_code("DX_4019")
    assert result == "DX_CAT_401"


def test_code_mapper_apply():
    import pandas as pd
    from oneehr.medcode.mapper import CodeMapper

    mapper = CodeMapper()
    mapper.add_icd_chapter_mapping(version=9, prefix="DX_")

    events = pd.DataFrame({
        "patient_id": ["p1", "p1", "p2"],
        "code": ["DX_4019", "LAB_50801", "DX_25000"],
        "value": [1, 80, 1],
    })
    result = mapper.apply(events)
    assert "CHAPTER" in result["code"].iloc[0]
    assert result["code"].iloc[1] == "LAB_50801"  # unchanged
