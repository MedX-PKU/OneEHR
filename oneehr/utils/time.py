from __future__ import annotations

import re
from datetime import timedelta


_BIN_RE = re.compile(r"^(\d+)([smhdw])$")


def parse_bin_size(bin_size: str) -> str:
    """Convert human-friendly bin size like `6h` to a pandas-compatible offset alias.

    Supported units:
    - s: seconds
    - m: minutes
    - h: hours
    - d: days
    - w: weeks
    """

    m = _BIN_RE.match(bin_size.strip().lower())
    if not m:
        raise ValueError(f"Invalid bin_size: {bin_size!r}. Expected like '1h', '6h', '1d'.")
    n = int(m.group(1))
    unit = m.group(2)
    unit_map = {"s": "S", "m": "min", "h": "H", "d": "D", "w": "W"}
    return f"{n}{unit_map[unit]}"


def bin_to_timedelta(bin_size: str) -> timedelta:
    m = _BIN_RE.match(bin_size.strip().lower())
    if not m:
        raise ValueError(f"Invalid bin_size: {bin_size!r}")
    n = int(m.group(1))
    unit = m.group(2)
    if unit == "s":
        return timedelta(seconds=n)
    if unit == "m":
        return timedelta(minutes=n)
    if unit == "h":
        return timedelta(hours=n)
    if unit == "d":
        return timedelta(days=n)
    if unit == "w":
        return timedelta(weeks=n)
    raise AssertionError("unreachable")

