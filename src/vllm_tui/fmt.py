"""Number formats (SPEC.md §5): speeds, durations, token counts, percents."""

from __future__ import annotations

import math

NO_DATA = "no data"


def missing(v: float) -> bool:
    return v is None or math.isnan(v) or math.isinf(v)


def speed(v: float) -> str:
    if missing(v):
        return NO_DATA
    a = abs(v)
    if a < 999.95:
        return f"{v:.1f}"
    if a < 99999.5:
        return f"{round(v):,}"
    if a < 999500:
        return f"{v / 1e3:.0f}K"
    if a < 999.5e6:
        return f"{v / 1e6:.2f}M"
    return f"{v / 1e9:.2f}B"


def rate(v: float) -> str:
    return speed(v) if missing(v) else speed(v) + "/s"


def pct(v: float) -> str:
    return NO_DATA if missing(v) else f"{v:.1f}%"


def tokens(v: float) -> str:
    if missing(v):
        return NO_DATA
    a = abs(v)
    if a < 999.5:
        return f"{v:.0f}"
    if a < 999500:
        return f"{v / 1e3:.0f}K"
    if a < 999.5e6:
        return f"{v / 1e6:.1f}M"
    return f"{v / 1e9:.2f}B"


def dur(s: float) -> str:
    if missing(s):
        return NO_DATA
    if s < 0.9995:
        return f"{s * 1e3:.0f}ms"
    if s < 9.995:
        return f"{s:.2f}s"
    return f"{s:.1f}s"


def dur_short(s: float) -> str:
    if missing(s):
        return NO_DATA
    if s < 0.9995:
        return f"{s * 1e3:.0f}ms"
    return f"{s:.1f}s"


def count(v: float) -> str:
    if missing(v):
        return NO_DATA
    if abs(v) < 10000:
        return f"{v:.0f}"
    return tokens(v)


def span(seconds: float) -> str:
    s = round(seconds)
    if s <= 120 or s % 60:
        return f"{s}s" if s < 600 else f"{s / 60:.0f}m"
    return f"{s // 60}m"
