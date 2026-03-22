# Medical Code Ontologies

OneEHR provides utilities for working with medical code systems, including ICD diagnosis codes, CCS grouping, and ATC drug classification.

## ICD-9 / ICD-10

Normalize, parse, and group ICD diagnosis and procedure codes.

```python
from oneehr.medcode import ICD9, ICD10

# Normalize codes (remove dots, whitespace, uppercase)
ICD9.normalize("401.9")   # → "4019"
ICD10.normalize("I10.0")  # → "I100"

# Chapter-level grouping
ICD9.chapter("401.9")     # → "Circulatory system"
ICD10.chapter("E11.9")    # → "Endocrine/metabolic"

# 3-digit category
ICD9.category("401.9")    # → "401"
ICD10.category("I10.0")   # → "I10"

# Parse into structured ICDCode object
code = ICD9.parse("401.9")
# ICDCode(version=9, code='4019', chapter='Circulatory system')
```

### ICD-9 Chapters

| Range | Chapter |
|-------|---------|
| 001-139 | Infectious diseases |
| 140-239 | Neoplasms |
| 240-279 | Endocrine/metabolic |
| 280-289 | Blood diseases |
| 290-319 | Mental disorders |
| 320-389 | Nervous system |
| 390-459 | Circulatory system |
| 460-519 | Respiratory system |
| 520-579 | Digestive system |
| 580-629 | Genitourinary system |
| 780-799 | Symptoms/signs |
| 800-999 | Injury/poisoning |
| E-codes | External causes |
| V-codes | Supplementary factors |

### Cross-Mapping (GEM)

Load CMS General Equivalence Mapping files for ICD-9 to ICD-10 conversion:

```python
from oneehr.medcode.icd import load_gem_mapping, icd9_to_icd10

load_gem_mapping("2018_I9gem.txt", direction="9to10")
icd9_to_icd10("4019")  # → ["I10"]
```

---

## CCS Grouping

Clinical Classifications Software reduces thousands of ICD codes into ~280 clinically meaningful categories.

```python
from oneehr.medcode import CCSGrouper

grouper = CCSGrouper("ccs_single_level.csv")
grouper.group("4019")        # → "101"
grouper.label("101")         # → "Essential hypertension"
grouper.group_with_label("4019")  # → ("101", "Essential hypertension")
print(len(grouper))          # number of mapped codes
print(grouper.categories)    # all CCS category IDs
```

The CCS mapping file can be downloaded from [AHRQ](https://hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp).

---

## ATC Drug Classification

The Anatomical Therapeutic Chemical (ATC) system classifies drugs in a 5-level hierarchy.

```python
from oneehr.medcode import ATCHierarchy

atc = ATCHierarchy()

# Level detection
ATCHierarchy.level("A")        # → 1 (Anatomical main group)
ATCHierarchy.level("A02")      # → 2 (Therapeutic subgroup)
ATCHierarchy.level("A02BC01")  # → 5 (Chemical substance)

# Parent extraction
ATCHierarchy.parent("A02BC01", target_level=1)  # → "A"
ATCHierarchy.parent("A02BC01", target_level=2)  # → "A02"

# Name lookup (built-in for level 1)
atc.group_name("A02BC01", level=1)  # → "Alimentary tract and metabolism"

# 14 main groups
atc.main_groups  # {'A': 'Alimentary...', 'B': 'Blood...', ...}
```

Load a full ATC mapping for deeper levels:

```python
atc = ATCHierarchy("atc_codes.csv")  # columns: atc_code, atc_name
```

---

## CodeMapper

Unified interface for mapping event codes in dynamic tables, useful for dimensionality reduction before binning.

```python
from oneehr.medcode import CodeMapper, CCSGrouper

mapper = CodeMapper()

# Map ICD codes to chapter-level groups
mapper.add_icd_chapter_mapping(version=9, prefix="DX_")

# Or to 3-digit categories
mapper.add_icd_category_mapping(version=9, prefix="DX_")

# Or to CCS groups
grouper = CCSGrouper("ccs_single_level.csv")
mapper.add_ccs_mapping(grouper, prefix="DX_")

# Map drug codes to ATC level-1 groups
from oneehr.medcode import ATCHierarchy
atc = ATCHierarchy()
mapper.add_atc_mapping(atc, level=1, prefix="RX_")

# Apply to events DataFrame
mapped_events = mapper.apply(events_df)
```

### Integration with Preprocessing

Apply code mapping before running `oneehr preprocess`:

```python
import pandas as pd
from oneehr.medcode import CodeMapper

# Load and map
events = pd.read_csv("data/dynamic.csv")
mapper = CodeMapper()
mapper.add_icd_chapter_mapping(version=9)
mapped = mapper.apply(events)
mapped.to_csv("data/dynamic_mapped.csv", index=False)
```

Then point your TOML config to the mapped file:

```toml
[dataset]
dynamic = "data/dynamic_mapped.csv"
```
