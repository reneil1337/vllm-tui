"""Omarchy theme → colour roles.

Reads the active Omarchy theme (~/.local/state/omarchy/current/theme/
colors.toml) and maps it onto the roles the pages draw with, the same way
Omarchy's own btop template maps colors.toml onto btop's keys. The app polls
the file, so `omarchy theme set …` recolours a running dashboard.
"""

from __future__ import annotations

import colorsys
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

HOME = Path.home()
STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", HOME / ".local/state")) / "omarchy/current"
CURRENT_COLORS = STATE_DIR / "theme/colors.toml"
CURRENT_NAME = STATE_DIR / "theme.name"
OMARCHY_PATH = Path(os.environ.get("OMARCHY_PATH", "/usr/share/omarchy"))
THEME_DIRS = [OMARCHY_PATH / "themes", HOME / ".config/omarchy/themes"]

FOLLOW = "omarchy"  # pseudo theme: follow whatever Omarchy has active

# Used when there's no Omarchy (or no colors.toml): Tokyo Night.
FALLBACK = {
    "mode": "dark",
    "accent": "#7aa2f7",
    "selection": "#292e42",
    "muted": "#414868",
    "background": "#1a1b26",
    "foreground": "#a9b1d6",
    "light_foreground": "#b4bee6",
    "bright_foreground": "#c0caf5",
    "red": "#f7768e",
    "yellow": "#e0af68",
    "orange": "#eb927b",
    "green": "#9ece6a",
    "cyan": "#449dab",
    "blue": "#7aa2f7",
    "magenta": "#ad8ee6",
}

# Fixed state colours from the device, used only when the theme has nothing
# close enough in hue (e.g. monochrome themes).
FIXED_OK, FIXED_WAIT, FIXED_BAD = "#34c759", "#ff9500", "#ff3b30"

RGB = tuple[int, int, int]


def rgb(v: str | None) -> RGB | None:
    if not isinstance(v, str):
        return None
    v = v.strip().lstrip("#")
    if len(v) == 2:
        return (int(v, 16),) * 3
    if len(v) in (6, 8):
        try:
            return tuple(int(v[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
        except ValueError:
            return None
    return None


def hexc(c: RGB) -> str:
    return "#{:02x}{:02x}{:02x}".format(*c)


def dist(a: RGB, b: RGB) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5


def luminance(c: RGB) -> float:
    def lin(x: float) -> float:
        x /= 255
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4

    r, g, b = map(lin, c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: RGB, b: RGB) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def mix(a: RGB, b: RGB, t: float) -> RGB:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b, strict=True))  # type: ignore[return-value]


def lift(c: RGB, towards: RGB, bg: RGB, target: float) -> RGB:
    """Mix c towards `towards` just enough to reach `target` contrast on bg."""
    if contrast(c, bg) >= target:
        return c
    lo, hi = 0.0, 1.0
    for _ in range(20):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if contrast(mix(c, towards, mid), bg) >= target else (mid, hi)
    return mix(c, towards, hi)


def _hue_ok(c: RGB, hue: float, spread: float, bg: RGB) -> float | None:
    h, _l, s = colorsys.rgb_to_hls(*(x / 255 for x in c))
    if s < 0.25 or contrast(c, bg) < 2.2:
        return None
    d = abs(h * 360 - hue)
    d = min(d, 360 - d)
    return d if d <= spread else None


def _state_pick(
    colors: dict,
    keys: tuple[str, ...],
    palette: list[RGB],
    hue: float,
    spread: float,
    bg: RGB,
    fallback: str,
) -> RGB:
    """The theme's own colour for a state: its named key if that's roughly
    the right hue, else the closest hue in the palette, else the fixed one."""
    for k in keys:
        c = rgb(colors.get(k))
        if c and _hue_ok(c, hue, spread * 1.6, bg) is not None:
            return c
    return _hue_pick(palette, hue, bg, spread, fallback)


def _hue_pick(colors: list[RGB], hue: float, bg: RGB, spread: float, fallback: str) -> RGB:
    best, best_d = None, spread
    for c in colors:
        h, _l, s = colorsys.rgb_to_hls(*(x / 255 for x in c))
        if s < 0.35 or contrast(c, bg) < 2.2:
            continue
        d = abs(h * 360 - hue)
        d = min(d, 360 - d)
        if d < best_d:
            best, best_d = c, d
    return best or rgb(fallback)  # type: ignore[return-value]


@dataclass
class Theme:
    name: str
    slug: str
    dark: bool
    roles: dict[str, str]

    def __getitem__(self, role: str) -> str:
        return self.roles[role]


def resolve(colors: dict, name: str, slug: str) -> Theme:
    def get(*keys: str) -> RGB:
        for k in keys:
            c = rgb(colors.get(k))
            if c:
                return c
        for k in keys:
            c = rgb(FALLBACK.get(k))
            if c:
                return c
        return (128, 128, 128)

    bg = get("background")
    fg = get("foreground")
    out: dict[str, RGB] = {
        "BG": bg,
        "VALUE": fg,
        "TITLE": get("bright_foreground", "foreground")
        if contrast(get("bright_foreground"), bg) > contrast(fg, bg)
        else fg,
        "LABEL": get("light_foreground", "foreground"),
        "DIM": get("muted", "dark_foreground"),
        # Box colours as Omarchy's btop template: cpu/mem/net/proc box.
        "BORDER": get("magenta"),
        "BOX2": get("green"),
        "BOX3": get("red"),
        "BOX4": get("accent", "blue"),
        "GOOD": get("green"),
        "WARN": get("yellow"),
        "BAD": get("red"),
        "DECODE": get("accent", "blue"),
        "PREFILL": get("yellow"),
        "SELECT": get("selection"),
        "METER": get("selection", "muted"),
    }
    # Secondary text and borders readable on the background (the device lifts
    # btop's near-invisible inactive_fg the same way).
    for role, target in {
        "DIM": 3.6,
        "LABEL": 3.5,
        "BORDER": 1.9,
        "BOX2": 1.9,
        "BOX3": 1.9,
        "BOX4": 1.9,
        "GOOD": 2.6,
        "WARN": 2.6,
        "BAD": 2.6,
        "DECODE": 3.0,
        "PREFILL": 3.0,
    }.items():
        out[role] = lift(out[role], fg, bg, target)
    if dist(out["PREFILL"], out["DECODE"]) < 80:
        cands = [out["PREFILL"]] + [
            lift(get(k), fg, bg, 3.0) for k in ("orange", "magenta", "bright_yellow", "red", "cyan", "green")
        ]
        out["PREFILL"] = max(cands, key=lambda c: min(dist(c, out["DECODE"]), dist(c, bg)))

    palette = [
        c
        for k, v in colors.items()
        if isinstance(v, str)
        and not k.startswith(
            (
                "background",
                "dark_back",
                "darker_back",
                "lighter_back",
                "selection",
                "muted",
                "hyprland",
                "active_",
            )
        )
        for c in [rgb(v)]
        if c
    ]
    out["STATE_OK"] = _state_pick(colors, ("green", "bright_green"), palette, 120, 40, bg, FIXED_OK)
    out["STATE_WAIT"] = _state_pick(colors, ("orange", "yellow", "bright_yellow"), palette, 36, 14, bg, FIXED_WAIT)
    out["STATE_BAD"] = _state_pick(colors, ("red", "bright_red"), palette, 0, 20, bg, FIXED_BAD)
    out["BRAND"] = out["STATE_BAD"]
    dark = str(colors.get("mode", "dark" if luminance(bg) < 0.4 else "light")) != "light"
    return Theme(name=name, slug=slug, dark=dark, roles={k: hexc(v) for k, v in out.items()})


def _read(path: Path) -> dict | None:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None


def pretty(slug: str) -> str:
    return " ".join(w.capitalize() for w in slug.replace("_", "-").split("-"))


def current_name() -> str:
    try:
        return CURRENT_NAME.read_text().strip() or "omarchy"
    except OSError:
        return ""


def current_signature() -> tuple:
    """Changes whenever Omarchy switches theme (polled once a second)."""
    sig = []
    for p in (CURRENT_COLORS, CURRENT_NAME):
        try:
            st = p.stat()
            sig.append((st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append(None)
    return tuple(sig)


def load_current() -> Theme:
    colors = _read(CURRENT_COLORS)
    name = current_name() or "Tokyo Night"
    if colors is None:
        return resolve(FALLBACK, "Tokyo Night", FOLLOW)
    return resolve(colors, pretty(name), FOLLOW)


def available() -> list[str]:
    """Theme slugs installed on this machine (stock + user), sorted."""
    slugs = set()
    for d in THEME_DIRS:
        try:
            for p in d.iterdir():
                if (p / "colors.toml").is_file():
                    slugs.add(p.name)
        except OSError:
            pass
    return sorted(slugs, key=str.lower)


def load(slug: str) -> Theme:
    if slug == FOLLOW:
        return load_current()
    for d in reversed(THEME_DIRS):  # user themes win over stock ones
        colors = _read(d / slug / "colors.toml")
        if colors is not None:
            return resolve(colors, pretty(slug), slug)
    return load_current()
