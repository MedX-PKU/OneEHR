"""Dataset converters: transform raw clinical datasets into OneEHR's three-table format.

Each converter reads source tables and produces:
  - dynamic.csv: (patient_id, event_time, code, value)
  - static.csv:  (patient_id, age, sex, ...)
  - label.csv:   (patient_id, label_time, label_code, label_value)
"""

from oneehr.data.converters.eicu import EICUConverter
from oneehr.data.converters.mimic3 import MIMIC3Converter
from oneehr.data.converters.mimic4 import MIMIC4Converter

__all__ = ["MIMIC3Converter", "MIMIC4Converter", "EICUConverter"]
