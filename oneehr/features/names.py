from __future__ import annotations

import re


_STATIC_CAT_RE = re.compile(r"^cat__([^_].*?)__(.+)$")
_DYNAMIC_CAT_RE = re.compile(r"^cat__(.+?)__(.+)$")


def display_name(col: str) -> str:
    """Convert an internal feature column name into a clinician-friendly name.

    Internal conventions:
      - num__{name}
      - cat__{name}__{level}

    Output conventions (display):
      - numeric: {name}
      - categorical: {name}={level}
    """

    if col.startswith("num__"):
        return col[len("num__") :]
    if col.startswith("cat__"):
        rest = col[len("cat__") :]
        if "__" in rest:
            left, right = rest.split("__", 1)
            return f"{left}={right}"
        return rest
    return col


def display_names(cols: list[str]) -> list[str]:
    return [display_name(c) for c in cols]

