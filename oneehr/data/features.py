"""Re-export feature utilities from tabular module for backward compatibility."""

from oneehr.data.tabular import dynamic_feature_columns, has_static_branch

__all__ = ["dynamic_feature_columns", "has_static_branch"]
