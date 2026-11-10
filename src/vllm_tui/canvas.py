"""A character grid drawn the way btop draws (SPEC.md §6–7): rounded boxes
with titles in notches, braille graphs, block sparklines, bars and a big
3-row hero number. Colours are theme roles (or literal #hex), resolved only
when the grid is turned into Rich text, so a theme switch is just a repaint.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from rich.style import Style
from rich.text import Text

from .theme import Theme, hexc, mix, rgb

H, V = "─", "│"
TL, TR, BL, BR = "╭", "╮", "╰", "╯"
NOTCH_TOP_L, NOTCH_TOP_R = "┐", "┌"
NOTCH_BOT_L, NOTCH_BOT_R = "┘", "└"
FULL, SHADE = "█", "░"
SEP = " · "


@dataclass
class Region:
    x0: int
    y0: int
    x1: int  # exclusive
    y1: int  # exclusive
    action: str


@dataclass
class Canvas:
    w: int
    h: int
    chars: list[list[str]] = field(init=False)
    colors: list[list[str]] = field(init=False)
    bolds: list[list[bool]] = field(init=False)
    regions: list[Region] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.chars = [[" "] * self.w for _ in range(self.h)]
        self.colors = [["VALUE"] * self.w for _ in range(self.h)]
        self.bolds = [[False] * self.w for _ in range(self.h)]

    # -- primitives ------------------------------------------------------
    def put(self, x: int, y: int, ch: str, role: str = "VALUE", bold: bool = False) -> None:
        if 0 <= x < self.w and 0 <= y < self.h:
            self.chars[y][x] = ch
            self.colors[y][x] = role
            self.bolds[y][x] = bold

    def text(
        self,
        x: int,
        y: int,
        s: str,
        role: str = "VALUE",
        bold: bool = False,
        limit: int | None = None,
    ) -> int:
        end = self.w if limit is None else min(limit, self.w)
        for ch in s:
            if x >= end:
                break
            self.put(x, y, ch, role, bold)
            x += 1
        return x

    def text_right(self, end: int, y: int, s: str, role: str = "VALUE", bold: bool = False) -> int:
        start = end - len(s)
        self.text(start, y, s, role, bold)
        return start

    def click(self, x0: int, y0: int, x1: int, y1: int, action: str) -> None:
        self.regions.append(Region(x0, y0, x1, y1, action))

    def hit(self, x: int, y: int) -> str | None:
        for r in reversed(self.regions):
            if r.x0 <= x < r.x1 and r.y0 <= y < r.y1:
                return r.action
        return None

    # -- boxes -----------------------------------------------------------
    def edge(
        self,
        y: int,
        box: str,
        top: bool,
        left: list[tuple[str, str, bool]] = (),
        right: list[tuple[str, str, bool]] = (),
        x0: int = 0,
        x1: int | None = None,
    ) -> list[tuple[int, int]]:
        """A box's top or bottom edge with notched labels on either side.
        Returns the (start, end) columns of each label's text, left first."""
        x1 = self.w if x1 is None else x1
        nl, nr = (NOTCH_TOP_L, NOTCH_TOP_R) if top else (NOTCH_BOT_L, NOTCH_BOT_R)
        self.put(x0, y, TL if top else BL, box)
        for x in range(x0 + 1, x1 - 1):
            self.put(x, y, H, box)
        self.put(x1 - 1, y, TR if top else BR, box)
        spans: list[tuple[int, int]] = []
        # Right notches first so left ones can be clipped against them.
        rx = x1 - 2
        right_spans = []
        for text, role, bold in right:
            need = len(text) + 2
            if rx - need < x0 + 2:
                break
            start = rx - need
            self.put(start, y, nl, box)
            self.text(start + 1, y, text, role, bold)
            self.put(start + 1 + len(text), y, nr, box)
            right_spans.append((start + 1, start + 1 + len(text)))
            rx = start - 1
        x = x0 + 2
        for text, role, bold in left:
            room = rx - x - 2
            if room < 1:
                break
            if len(text) > room:
                text = text[: max(0, room - 1)] + "…"
            self.put(x, y, nl, box)
            self.text(x + 1, y, text, role, bold)
            spans.append((x + 1, x + 1 + len(text)))
            x += len(text) + 1
            self.put(x, y, nr, box)
            x += 2
        return spans + list(reversed(right_spans))

    def sides(self, y: int, box: str, x0: int = 0, x1: int | None = None) -> None:
        x1 = self.w if x1 is None else x1
        self.put(x0, y, V, box)
        self.put(x1 - 1, y, V, box)

    def box(
        self,
        y: int,
        height: int,
        box: str,
        title: str,
        right: list[tuple[str, str, bool]] = (),
        bottom: bool = True,
    ) -> None:
        self.edge(y, box, True, [(title, "TITLE", True)], right)
        for r in range(y + 1, y + height - (1 if bottom else 0)):
            self.sides(r, box)
        if bottom:
            self.edge(y + height - 1, box, False)

    # -- rendering -------------------------------------------------------
    def render(self, theme: Theme, paint_bg: bool) -> list[Text]:
        bg = theme["BG"] if paint_bg else None
        cache: dict[tuple[str, bool], Style] = {}

        def style(role: str, bold: bool) -> Style:
            key = (role, bold)
            st = cache.get(key)
            if st is None:
                color = role if role.startswith("#") else theme.roles.get(role, theme["VALUE"])
                st = cache[key] = Style(color=color, bgcolor=bg, bold=bold)
            return st

        lines = []
        for y in range(self.h):
            t = Text(no_wrap=True, end="")
            row_c, row_k, row_b = self.chars[y], self.colors[y], self.bolds[y]
            run, key = [], None
            for x in range(self.w):
                k = (row_k[x], row_b[x])
                if k != key and run:
                    t.append("".join(run), style(*key))
                    run = []
                key = k
                run.append(row_c[x])
            if run:
                t.append("".join(run), style(*key))
            lines.append(t)
        return lines


# -- widgets -------------------------------------------------------------


def fill_role(pct: float) -> str:
    if pct is None or math.isnan(pct):
        return "DIM"
    return "GOOD" if pct < 70 else "WARN" if pct < 90 else "BAD"


def bar(c: Canvas, x: int, y: int, width: int, pct: float, role: str) -> None:
    filled = 0 if math.isnan(pct) else round(min(max(pct, 0.0), 100.0) / 100.0 * width)
    for i in range(width):
        if i < filled:
            c.put(x + i, y, FULL, role)
        else:
            c.put(x + i, y, SHADE, "METER")


def gradient(theme: Theme, a: str, b: str, steps: int) -> list[str]:
    """`steps` hex colours from role a (bottom) to role b (top)."""
    ca, cb = (
        rgb(theme[a] if not a.startswith("#") else a),
        rgb(theme[b] if not b.startswith("#") else b),
    )
    if steps <= 1:
        return [hexc(cb)]
    return [hexc(mix(ca, cb, i / (steps - 1))) for i in range(steps)]


BLOCKS = " ▁▂▃▄▅▆▇█"


def sparkline(
    c: Canvas,
    x: int,
    y: int,
    rows: int,
    cols: int,
    data: list[float],
    colors: list[str],
) -> float:
    """Block-element bars, one sample per column, bottom-aligned; scaled to
    the visible max. 0 is a baseline, a gap (NaN) stays empty.
    colors: one per row, bottom first."""
    d = data[-cols:] if cols > 0 else []
    d = [math.nan] * (cols - len(d)) + d
    finite = [v for v in d if not math.isnan(v)]
    vmax = max(finite) if finite else math.nan
    scale = vmax if finite and vmax > 0 else 1.0
    levels = rows * 8
    for i, v in enumerate(d):
        if math.isnan(v):
            continue
        h = max(1, round(max(v, 0.0) / scale * levels))
        for r in range(rows):
            base = (rows - 1 - r) * 8
            f = min(max(h - base, 0), 8)
            if f:
                c.put(x + i, y + r, BLOCKS[f], colors[rows - 1 - r])
    return vmax


_LEFT = (0x01, 0x02, 0x04, 0x40)
_RIGHT = (0x08, 0x10, 0x20, 0x80)


def braille(
    c: Canvas,
    x: int,
    y: int,
    rows: int,
    cols: int,
    data: list[float],
    vmax: float,
    colors: list[str],
) -> None:
    """Area graph, 2 samples × 4 levels per cell (SPEC.md §6)."""
    n = cols * 2
    d = data[-n:]
    d = [math.nan] * (n - len(d)) + d
    levels = rows * 4
    scale = vmax if vmax and vmax > 0 and not math.isnan(vmax) else 1.0
    h = []
    for v in d:
        if math.isnan(v):
            h.append(-1)
            continue
        lv = round(min(max(v, 0.0) / scale, 1.0) * levels)
        h.append(1 if v > 0 and lv == 0 else lv)
    for r in range(rows):
        role = colors[rows - 1 - r]
        for col in range(cols):
            bits = 0
            for side in (0, 1):
                hv = h[col * 2 + side]
                if hv < 0:
                    continue
                dots = _RIGHT if side else _LEFT
                for k in range(4):
                    if hv >= (rows - r - 1) * 4 + (3 - k) + 1:
                        bits |= dots[k]
            c.put(x + col, y + r, chr(0x2800 + bits), role)


# Big 3-row digits in the rounded box-drawing style of the borders.
BIG = {
    "0": ("╭─╮", "│ │", "╰─╯"),
    "1": ("╶┐ ", " │ ", "╶┴╴"),
    "2": ("╶─╮", "╭─╯", "╰─╴"),
    "3": ("╶─╮", " ─┤", "╶─╯"),
    "4": ("╷ ╷", "╰─┤", "  ╵"),
    "5": ("╭─╴", "╰─╮", "╶─╯"),
    "6": ("╭─╴", "├─╮", "╰─╯"),
    "7": ("╶─┐", "  │", "  ╵"),
    "8": ("╭─╮", "├─┤", "╰─╯"),
    "9": ("╭─╮", "╰─┤", "╶─╯"),
    ".": (" ", " ", "•"),
    ",": (" ", " ", "╯"),
    "K": ("╷╭╴", "├┤ ", "╵╰╴"),
    "M": ("╭┬╮", "│││", "╵╵╵"),
    " ": ("   ", "   ", "   "),
    "-": ("   ", "╶─╴", "   "),
}


def big_width(s: str) -> int:
    return sum(len(BIG.get(ch, BIG[" "])[0]) for ch in s) + max(0, len(s) - 1)


def big(c: Canvas, x: int, y: int, s: str, role: str) -> int:
    for ch in s:
        g = BIG.get(ch, BIG[" "])
        for r in range(3):
            c.text(x, y + r, g[r], role, True)
        x += len(g[0]) + 1
    return x


class Writer:
    """Puts runs of text left to right, clipped at `limit`."""

    def __init__(self, c: Canvas, y: int, x: int, limit: int | None = None) -> None:
        self.c, self.y, self.x = c, y, x
        self.limit = c.w - 2 if limit is None else limit

    def put(self, s: str, role: str = "VALUE", bold: bool = False) -> Writer:
        self.x = self.c.text(self.x, self.y, s, role, bold, self.limit)
        return self

    def to(self, x: int) -> Writer:
        self.x = x
        return self


Formatter = Callable[[float], str]
