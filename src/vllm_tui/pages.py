"""Pages (SPEC.md §7), laid out for whatever size the terminal is.

Every page is a stack of btop boxes. The last box's bottom edge is the
footer: the ≡ settings notch, the poll status, and which server is shown.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from . import fmt
from .canvas import (
    SEP,
    Canvas,
    Writer,
    bar,
    big,
    big_width,
    braille,
    fill_role,
    gradient,
    sparkline,
)
from .derive import ServerState, StreamState, View
from .theme import Theme, hexc, mix, rgb

BRAND = "vLLM·TUI"
MIN_W, MIN_H = 48, 18

STATE_ROLE = {
    ServerState.IDLE: "STATE_OK",
    ServerState.BUSY: "STATE_OK",
    ServerState.WARMING: "STATE_OK",
    ServerState.RESTART: "STATE_WAIT",
    ServerState.STALE: "STATE_WAIT",
    ServerState.CONNECTING: "STATE_WAIT",
    ServerState.DOWN: "STATE_BAD",
    ServerState.UNKNOWN: "STATE_BAD",
}

BARS = "⣀⣤⣶⣿"


@dataclass
class Strip:
    label: str
    state: ServerState
    tg: float


@dataclass
class Ctx:
    theme: Theme
    clock: str
    index: int
    servers: list[Strip]
    page: str = "overview"


def nan(x: float) -> bool:
    return x is None or math.isnan(x)


def vrole(v: float, role: str) -> str:
    return "DIM" if nan(v) else role


def short_version(v: str) -> str:
    return v.split("+")[0]


# -- chrome ------------------------------------------------------------------


def header(c: Canvas, v: View, ctx: Ctx) -> int:
    """Rows 0–3 (+1 server strip when there's more than one server)."""
    box = "BORDER"
    rtt = "" if nan(v.rtt_ms) else f"{v.rtt_ms:.0f}ms"
    lat = BARS + (" " + rtt if rtt else "")
    right = [(lat, "DIM", False)]
    if ctx.clock:
        right.append((ctx.clock, "VALUE", False))
    spans = c.edge(0, box, True, [(BRAND, "BRAND", True)], right)
    # Latency as braille bars (the device's Wi-Fi bars): lit in state colours.
    if len(spans) >= 2:
        x = spans[-1][0]
        up = v.state not in (ServerState.DOWN, ServerState.CONNECTING) and not nan(v.rtt_ms)
        n = 0
        if up:
            n = 4 if v.rtt_ms < 50 else 3 if v.rtt_ms < 150 else 2 if v.rtt_ms < 400 else 1
        lit = "STATE_OK" if n >= 3 else "STATE_WAIT" if n == 2 else "STATE_BAD"
        for i in range(4):
            c.put(x + i, 0, BARS[i], lit if i < n else "DIM")

    # Server · address, state right-aligned.
    c.sides(1, box)
    st = v.state.value
    sx = c.text_right(c.w - 2, 1, st, STATE_ROLE[v.state], True)
    # "name:port · address", the port only when it isn't the default, the
    # address only when it isn't just the name again.
    w = Writer(c, 1, 2, sx - 1)
    name = v.server or fmt.NO_DATA
    port = v.url.rsplit(":", 1)[-1].split("/")[0] if ":" in v.url.split("//", 1)[-1] else ""
    if port.isdigit() and port != "8000":
        name += ":" + port
    w.put(name, "TITLE" if v.server else "DIM", True)
    if v.server_ip and v.server_ip != v.server:
        w.put(SEP, "DIM").put(v.server_ip, "VALUE")

    # Engine · model, version on the right when there's room.
    c.sides(2, box)
    ver = short_version(v.version)
    limit = c.w - 2
    if ver:
        vx = c.w - 2 - len(ver) - 1
        if vx > 40:
            c.text(vx, 2, "v" + ver, "DIM")
            limit = vx - 1
    w = Writer(c, 2, 2, limit)
    w.put(v.engine or "???", "TITLE", True)
    if v.model:
        room = limit - w.x - 3
        model = v.model if len(v.model) <= room else v.model[: max(0, room - 1)] + "…"
        w.put(SEP, "DIM").put(model, "VALUE")
    row = 3
    if len(ctx.servers) > 1:
        c.sides(row, box)
        x = 2
        for i, s in enumerate(ctx.servers):
            active = i == ctx.index
            tg = "" if nan(s.tg) else " " + fmt.speed(s.tg)
            item_w = 2 + len(s.label) + len(tg) + 1
            if x + item_w > c.w - 2:
                c.text(x, row, "…", "DIM")
                break
            x0 = x
            c.put(x, row, "▶" if active else " ", "TITLE", True)
            c.put(x + 1, row, "●", STATE_ROLE[s.state])
            x = c.text(x + 2, row, " " + s.label, "TITLE" if active else "LABEL", active)
            x = c.text(x, row, tg, "DECODE" if active else "DIM")
            c.click(x0, row, x, row + 1, f"server:{i}")
            x += 2
        row += 1
    c.edge(row, box, False)
    return row + 1


def footer(c: Canvas, v: View, ctx: Ctx, box: str, hint: str) -> None:
    y = c.h - 1
    if len(ctx.servers) > 1:
        hint = f"{ctx.index + 1}/{len(ctx.servers)}"
    spans = c.edge(
        y,
        box,
        False,
        [("≡", "TITLE", True), (v.status or fmt.NO_DATA, "DIM", False)],
        [(hint, "DIM", False)],
    )
    if spans:
        a, b = spans[0]
        c.click(0, y, b + 2, y + 1, "settings")
    if len(spans) >= 3:
        a, b = spans[-1]
        c.click(a - 1, y, b + 1, y + 1, "next" if len(ctx.servers) > 1 else "page")


def too_small(c: Canvas) -> None:
    msg = [f"terminal too small: {c.w}×{c.h}", f"needs at least {MIN_W}×{MIN_H}"]
    for i, m in enumerate(msg):
        c.text(
            max(0, (c.w - len(m)) // 2),
            c.h // 2 - 1 + i,
            m,
            "DIM" if i else "STATE_WAIT",
            not i,
        )


# -- overview ----------------------------------------------------------------


def throughput(c: Canvas, y: int, v: View, ctx: Ctx, hero_rows: int) -> int:
    box = "BOX2"
    height = 2 + hero_rows + 3
    span = fmt.span(max(1, c.w - 2 - 33) * v.poll_s)
    c.box(y, height, box, "Throughput", [(span, "DIM", False)])
    top = y + 1
    c.text(2, top, "TG", "DECODE", True)
    hero_w = 21
    if nan(v.tg_live):
        c.text(5, top + 1, fmt.NO_DATA, "DIM")
    else:
        s = fmt.speed(v.tg_live)
        big(c, 5 + hero_w - big_width(s), top + (hero_rows - 3) // 2, s, "DECODE")
    c.text(5 + hero_w + 1, top, "tok/s", "DIM")
    c.text(5 + hero_w + 1, top + 1, "live", "DIM")
    sx = 5 + hero_w + 7
    cols = c.w - 2 - sx
    if cols > 3:
        colors = gradient(ctx.theme, "#" + _dim_hex(ctx.theme, "DECODE"), "DECODE", hero_rows)
        sparkline(c, sx, top, hero_rows, cols, v.tg_hist, colors)

    r = top + hero_rows
    w = Writer(c, r, 2)
    if v.pp_per_request:
        w.put("PP ", "PREFILL", True).put("last ", "LABEL").put(fmt.rate(v.pp_last), vrole(v.pp_last, "PREFILL"))
        if not nan(v.prefilling_s):
            w.put(SEP, "DIM").put(f"prefilling {v.prefilling_s:.0f}s", "WARN")
    elif all(nan(x) for x in (v.pp_live, v.pp_avg, v.pp_eng)):
        w.put("PP ", "PREFILL", True).put(fmt.NO_DATA, "DIM")
    else:
        # Live prefill rate where the engine measures it while it works
        # (Flare, in 4096-token chunks); window average beside it.
        w.put("PP ", "PREFILL", True).put("live ", "LABEL").put(fmt.rate(v.pp_live), vrole(v.pp_live, "PREFILL"))
        w.put("  avg ", "LABEL").put(fmt.speed(v.pp_avg), vrole(v.pp_avg, "PREFILL"))
        if not nan(v.prefilling_s):
            w.put(SEP, "DIM").put(f"prefilling {v.prefilling_s:.0f}s", "WARN")

    r += 1
    w = Writer(c, r, 2)
    if nan(v.running):
        w.put("run ", "LABEL").put(fmt.NO_DATA, "DIM")
    else:
        run = f"{v.running:.0f}" if nan(v.max_batch) else f"{v.running:.0f}/{v.max_batch:.0f}"
        w.put("run ", "LABEL").put(run, "VALUE").put(SEP, "DIM")
        w.put("wait ", "LABEL").put(fmt.count(v.waiting), "WARN" if v.waiting > 0 else vrole(v.waiting, "VALUE"))
    w.put(SEP, "DIM").put("TG ", "DECODE", True)
    if nan(v.tg_avg):
        w.put(fmt.NO_DATA, "DIM")
    else:
        w.put("avg ", "LABEL").put(fmt.speed(v.tg_avg), "DECODE").put(SEP, "DIM")
        w.put("max ", "LABEL").put(fmt.speed(v.tg_max), vrole(v.tg_max, "DECODE"))

    r += 1
    now, pk = fmt.pct(v.kv_pct), fmt.pct(v.kv_peak)
    tok = ""
    if not nan(v.kv_tokens) and not nan(v.kv_capacity):
        tok = f"{fmt.tokens(v.kv_tokens)}/{fmt.tokens(v.kv_capacity)}"
    text = (
        3 + len(now) + 1 + (0 if nan(v.kv_peak) else 3 + 5 + len(pk)) + 3 + (len(tok) + 4 if tok else len(fmt.NO_DATA))
    )
    bar_w = max(3, min(40, c.w - 4 - text))
    w = Writer(c, r, 2)
    w.put("KV ", "LABEL").put(now, fill_role(v.kv_pct)).put(" ")
    bar(c, w.x, r, bar_w, v.kv_pct, fill_role(v.kv_pct))
    w.to(w.x + bar_w)
    if not nan(v.kv_peak):
        w.put(SEP, "DIM").put("peak ", "LABEL").put(pk, fill_role(v.kv_peak))
    w.put(SEP, "DIM")
    if tok:
        w.put(tok, "VALUE").put(" tok", "DIM")
    else:
        w.put(fmt.NO_DATA, "DIM")
    return y + height


def _dim_hex(theme: Theme, role: str) -> str:
    """The role mixed 45% towards the background, for graph gradients."""
    return hexc(mix(rgb(theme[role]), rgb(theme["BG"]), 0.45))[1:]


STREAM_COLS = {"ctx": 22, "tokens": 29, "TTFT": 36, "TG": 44}


def streams(c: Canvas, y: int, v: View, rows: int) -> int:
    box = "BOX3"
    height = rows + 2
    c.box(y, height, box, "Streams")
    for label, end in STREAM_COLS.items():
        c.text_right(end, y, label, "DIM")
    wide = c.w >= 60
    if wide:
        c.text(46, y, "fill", "DIM")
    if not v.streams:
        c.text(3, y + 1, fmt.NO_DATA, "DIM")
    for k, s in enumerate(v.streams[:rows]):
        r = y + 1 + k
        c.text(3, r, str(s.id), "VALUE")
        label, role = {
            StreamState.DECODING: ("decoding", "GOOD"),
            StreamState.PREFILL: ("prefill", "WARN"),
            StreamState.IDLE: ("idle", "DIM"),
            StreamState.DONE: ("done", "LABEL"),
        }.get(s.state, ("", "DIM"))
        c.text(6, r, label, role)
        c.text_right(22, r, fmt.pct(s.ctx_pct), vrole(s.ctx_pct, fill_role(s.ctx_pct)))
        c.text_right(29, r, fmt.tokens(s.tokens), vrole(s.tokens, "VALUE"))
        if wide:
            bar(c, 46, r, c.w - 2 - 46 - 1, s.ctx_pct, fill_role(s.ctx_pct))
        if s.state is StreamState.IDLE:
            continue
        if nan(s.ttft_s) and nan(s.tg):
            c.text(31, r, fmt.NO_DATA, "DIM")
            continue
        c.text_right(36, r, fmt.dur_short(s.ttft_s), vrole(s.ttft_s, "VALUE"))
        c.text_right(44, r, fmt.rate(s.tg), vrole(s.tg, "DECODE"))
    return y + height


def mtp_role(p: float) -> str:
    if nan(p):
        return "DIM"
    return "GOOD" if p >= 70 else "WARN" if p >= 50 else "BAD"


def has_row2(v: View) -> bool:
    """Latency breakdown and request counters: filled as far as the engine
    exports them, `no data` for the rest — the same row in every engine."""
    return not all(nan(x) for x in (v.itl_cum, v.queue_cum, v.lat_cum, v.success, v.preemptions, v.failed))


def has_row3(v: View) -> bool:
    """One extra line: latency percentiles for engines that export histogram
    buckets (vLLM, TensorFold), Flare's verify/acceptance/depth/weights, and
    vLLM's prefix-cache hit rate."""
    if not all(nan(x) for x in (v.ttft_p50, v.ttft_p99, v.e2e_p50, v.e2e_p99)):
        return True
    if not all(nan(x) for x in (v.verify_rate, v.accept_per_round, v.verify_depth, v.weight_gbps)):
        return True
    return not nan(v.prefix_hit_pct)


def averages(c: Canvas, y: int, v: View, ctx: Ctx, height: int, hint: str) -> None:
    box = "BOX4"
    c.edge(y, box, True, [("Averages", "TITLE", True)])
    for r in range(y + 1, c.h - 1):
        c.sides(r, box)
    # Row 1 is the same for every engine: time to first token, MTP acceptance,
    # prompt-cache hit rate.
    w = Writer(c, y + 1, 2)
    if all(nan(x) for x in (v.ttft_cum, v.mtp_pct, v.cache_hit_pct)):
        w.put(fmt.NO_DATA, "DIM")
    else:
        w.put("TTFT ", "LABEL").put(fmt.dur(v.ttft_cum), vrole(v.ttft_cum, "VALUE")).put(SEP, "DIM")
        w.put("MTP ", "LABEL").put(fmt.pct(v.mtp_pct), mtp_role(v.mtp_pct)).put(SEP, "DIM")
        w.put("cache-hit ", "LABEL").put(fmt.pct(v.cache_hit_pct), vrole(v.cache_hit_pct, "VALUE"))
    row = y + 2
    if height >= 4 and row < c.h - 1 and has_row2(v):
        w = Writer(c, row, 2)
        w.put("ITL ", "LABEL").put(fmt.dur(v.itl_cum), vrole(v.itl_cum, "VALUE")).put(SEP, "DIM")
        w.put("queue ", "LABEL").put(fmt.dur(v.queue_cum), vrole(v.queue_cum, "VALUE")).put(SEP, "DIM")
        w.put("e2e ", "LABEL").put(fmt.dur(v.lat_cum), vrole(v.lat_cum, "VALUE")).put(SEP, "DIM")
        w.put("done ", "LABEL").put(fmt.count(v.success), vrole(v.success, "VALUE")).put(SEP, "DIM")
        w.put("preempt ", "LABEL").put(
            fmt.count(v.preemptions),
            "WARN" if v.preemptions > 0 else vrole(v.preemptions, "VALUE"),
        )
        if not nan(v.failed) and v.failed > 0:
            w.put(SEP, "DIM").put("failed ", "LABEL").put(fmt.count(v.failed), "BAD")
        row += 1
    if height >= 5 and row < c.h - 1 and has_row3(v):
        w = Writer(c, row, 2)
        if not all(nan(x) for x in (v.ttft_p50, v.ttft_p99, v.e2e_p50, v.e2e_p99)):
            # p50/p99 over the requests that finished in the recent window.
            w.put("TTFT ", "LABEL").put("p50 ", "DIM").put(fmt.dur_short(v.ttft_p50), vrole(v.ttft_p50, "VALUE"))
            w.put(" p99 ", "DIM").put(
                fmt.dur_short(v.ttft_p99),
                vrole(v.ttft_p99, "WARN" if not nan(v.ttft_p99) and v.ttft_p99 > 5 else "VALUE"),
            )
            w.put(SEP, "DIM").put("e2e ", "LABEL").put("p50 ", "DIM").put(
                fmt.dur_short(v.e2e_p50), vrole(v.e2e_p50, "VALUE")
            )
            w.put(" p99 ", "DIM").put(
                fmt.dur_short(v.e2e_p99), vrole(v.e2e_p99, "WARN" if not nan(v.e2e_p99) and v.e2e_p99 > 60 else "VALUE")
            )
            if not nan(v.pct_n):
                w.put(SEP, "DIM").put(fmt.count(v.pct_n) + " req " + fmt.span(v.pct_window_s), "DIM")
            if not nan(v.prefix_hit_pct):
                w.put(SEP, "DIM").put("prefix-hit ", "LABEL").put(
                    fmt.pct(v.prefix_hit_pct), vrole(v.prefix_hit_pct, "VALUE")
                )
        elif v.flare:
            w.put("verify ", "LABEL").put(
                f"{v.verify_rate:.1f}/s" if not nan(v.verify_rate) else fmt.NO_DATA,
                vrole(v.verify_rate, "VALUE"),
            )
            w.put(SEP, "DIM").put("accept ", "LABEL").put(
                f"{v.accept_per_round:.2f}/round" if not nan(v.accept_per_round) else fmt.NO_DATA,
                vrole(v.accept_per_round, "VALUE"),
            )
            w.put(SEP, "DIM").put("depth ", "LABEL").put(
                f"{v.verify_depth:.2f}" if not nan(v.verify_depth) else fmt.NO_DATA,
                vrole(v.verify_depth, "VALUE"),
            )
            w.put(SEP, "DIM").put("weights ", "LABEL").put(
                f"{v.weight_gbps:.0f} GB/s" if not nan(v.weight_gbps) else fmt.NO_DATA,
                vrole(v.weight_gbps, "VALUE"),
            )
            if not nan(v.reused_tokens):
                w.put(SEP, "DIM").put("reused ", "LABEL").put(
                    fmt.tokens(v.reused_tokens) + " tok",
                    vrole(v.reused_tokens, "VALUE"),
                )
        elif not nan(v.prefix_hit_pct):
            w.put("prefix-hit ", "LABEL").put(fmt.pct(v.prefix_hit_pct), vrole(v.prefix_hit_pct, "VALUE"))
    footer(c, v, ctx, box, hint)


def overview(c: Canvas, v: View, ctx: Ctx) -> None:
    if c.w < MIN_W or c.h < MIN_H:
        return too_small(c)
    y = header(c, v, ctx)
    # The same boxes for every engine: Streams always, one "no data" row when
    # the engine exports no per-slot series (vLLM and Flare have none).
    stream_rows = max(1, min(8, len(v.streams) or 4))
    avg_h = 2 + 1 + has_row2(v) + has_row3(v)
    fixed = y + 8 + stream_rows + 2 + avg_h
    extra = c.h - fixed
    if extra < 0:  # squeeze the stream table first
        cut = min(-extra, stream_rows - 1)
        stream_rows -= cut
        extra += cut
    if extra < 0 and avg_h > 3:
        avg_h, extra = 3, extra + 1
    if extra < 0:
        cut = min(-extra, stream_rows - 1)
        stream_rows -= cut
        extra += cut
    # Spare rows on a big terminal: the graphs page's boxes join the overview
    # (all three from 18 spare rows, only TG from 7); otherwise the hero grows.
    embed = extra >= 7
    hero_rows = 3 + (0 if embed else max(0, min(extra, 4)))
    y = throughput(c, y, v, ctx, hero_rows)
    y = streams(c, y, v, stream_rows)
    if embed:
        room = c.h - avg_h - y
        if room >= 18:
            y = graph_stack(c, y, v, ctx, room, close_last=True)
        else:
            y = graph(c, y, room - 2, graph_specs(v)[0], v, ctx, True)
    averages(c, y, v, ctx, c.h - y, "tab: graphs")


# -- graphs ------------------------------------------------------------------


@dataclass
class Stats:
    now: float = math.nan
    avg: float = math.nan
    max: float = math.nan
    min: float = math.nan


def stats(d: list[float], events_only: bool = False) -> Stats:
    s = Stats()
    s.now = d[-1] if d else math.nan
    if events_only:
        s.now = next((x for x in reversed(d) if not nan(x) and x > 0), math.nan)
    xs = [x for x in d if not nan(x) and not (events_only and x <= 0)]
    if xs:
        s.avg = sum(xs) / len(xs)
        s.max = max(xs)
        s.min = min(xs)
    return s


@dataclass
class GraphSpec:
    title: str
    box: str
    data: list[float]
    floor: float
    fixed_100: bool
    fmt: object
    role: str
    full_stats: bool
    events_only: bool = False
    grad: tuple[str, str] | None = None


STATS_W = 12


def graph(c: Canvas, y: int, rows: int, g: GraphSpec, v: View, ctx: Ctx, bottom: bool) -> int:
    cols = c.w - 2 - 2 - STATS_W - 1
    visible = g.data[-cols * 2 :]
    span = fmt.span(cols * 2 * v.poll_s)
    c.edge(y, g.box, True, [(g.title, "TITLE", True)], [(span, "DIM", False)])
    for r in range(y + 1, y + 1 + rows):
        c.sides(r, g.box)
    if bottom:
        c.edge(y + rows + 1, g.box, False)
    st = stats(visible, g.events_only)
    if all(nan(x) for x in visible):
        c.text((c.w - 7) // 2, y + 1 + (rows - 1) // 2, fmt.NO_DATA, "DIM")
        return y + rows + (2 if bottom else 1)
    vmax = 100.0 if g.fixed_100 else max(0.0 if nan(st.max) else st.max, g.floor)
    if g.grad:
        colors = gradient(ctx.theme, g.grad[0], g.grad[1], rows)
    else:
        colors = gradient(ctx.theme, "#" + _dim_hex(ctx.theme, g.role), g.role, rows)
    braille(c, 2, y + 1, rows, cols, visible, vmax, colors)
    lines = [
        ("last" if g.events_only else "now", st.now),
        ("avg", st.avg),
        ("max", st.max),
        ("min", st.min),
    ]
    if not g.full_stats:
        lines = [lines[0], lines[2]]
    sx = c.w - 2 - STATS_W
    for i, (label, val) in enumerate(lines[:rows]):
        s = g.fmt(val)
        c.text(sx, y + 1 + i, f"{label:<4}", "LABEL")
        c.text_right(c.w - 2, y + 1 + i, s, "DIM" if nan(val) else g.role)
    # scale marker, btop style
    if rows > len(lines) and not g.fixed_100:
        c.text(sx, y + rows, f"{'top':<4}", "DIM")
        c.text_right(c.w - 2, y + rows, g.fmt(vmax), "DIM")
    return y + rows + (2 if bottom else 1)


def graph_specs(v: View) -> list[GraphSpec]:
    return [
        GraphSpec("TG live tok/s", "BOX2", v.tg_hist, 10, False, fmt.speed, "DECODE", True),
        GraphSpec(
            "PP per request tok/s" if v.pp_per_request else "PP live tok/s",
            "BOX3",
            v.pp_hist,
            100,
            False,
            fmt.speed,
            "PREFILL",
            True,
            v.pp_per_request,
        ),
        GraphSpec(
            "KV %",
            "BOX4",
            v.kv_hist,
            0,
            True,
            fmt.pct,
            "GOOD",
            False,
            grad=("GOOD", "BAD"),
        ),
    ]


def graph_stack(c: Canvas, y: int, v: View, ctx: Ctx, height: int, close_last: bool) -> int:
    """TG / PP / KV boxes in `height` rows. Without close_last, the KV box
    has no bottom edge (the footer closes it)."""
    frames = 6 if close_last else 5
    content = max(6, height - frames)
    small = max(2, content // 5)
    tg_rows = content - 2 * small
    specs = graph_specs(v)
    y = graph(c, y, tg_rows, specs[0], v, ctx, True)
    y = graph(c, y, small, specs[1], v, ctx, True)
    y = graph(c, y, small if close_last else c.h - 1 - y - 1, specs[2], v, ctx, close_last)
    return y


def graphs(c: Canvas, v: View, ctx: Ctx) -> None:
    if c.w < MIN_W or c.h < MIN_H:
        return too_small(c)
    y = header(c, v, ctx)
    graph_stack(c, y, v, ctx, c.h - y, close_last=False)
    footer(c, v, ctx, "BOX4", "tab: overview")


# -- all servers -------------------------------------------------------------


@dataclass
class Col:
    label: str
    width: int
    width_of: Callable[[View], str]
    role: Callable[[View], str]


def _srv_cols() -> list[Col]:
    return [
        Col("TG", 8, lambda v: fmt.speed(v.tg_live), lambda v: vrole(v.tg_live, "DECODE")),
        Col("KV", 6, lambda v: fmt.pct(v.kv_pct), lambda v: fill_role(v.kv_pct)),
        Col(
            "PP",
            8,
            lambda v: fmt.rate(v.pp_live if not v.pp_per_request else v.pp_last),
            lambda v: vrole(v.pp_live if not v.pp_per_request else v.pp_last, "PREFILL"),
        ),
        Col(
            "run",
            8,
            lambda v: f"{v.running:.0f}" if nan(v.max_batch) else f"{v.running:.0f}/{v.max_batch:.0f}",
            lambda v: vrole(v.running, "VALUE"),
        ),
        Col(
            "TTFT",
            7,
            lambda v: fmt.dur_short(v.ttft_p50 if not nan(v.ttft_p50) else v.ttft_cum),
            lambda v: vrole(v.ttft_cum, "VALUE"),
        ),
        Col(
            "wait", 5, lambda v: fmt.count(v.waiting), lambda v: "WARN" if v.waiting > 0 else vrole(v.waiting, "VALUE")
        ),
        Col("MTP", 6, lambda v: fmt.pct(v.mtp_pct), lambda v: mtp_role(v.mtp_pct)),
        Col("hit", 6, lambda v: fmt.pct(v.cache_hit_pct), lambda v: vrole(v.cache_hit_pct, "VALUE")),
        Col("rtt", 6, lambda v: "" if nan(v.rtt_ms) else f"{v.rtt_ms:.0f}ms", lambda v: vrole(v.rtt_ms, "VALUE")),
    ]


def all_servers(c: Canvas, views: list[View], ctx: Ctx) -> None:
    """One row per server: the live numbers, at a glance (SPEC.md §7.1 has no
    such page on the device, which only shows the server on screen)."""
    if c.w < MIN_W or c.h < MIN_H:
        return too_small(c)
    active = views[ctx.index] if views else View()
    y = header(c, active, ctx)
    box = "BOX2"
    rows_room = c.h - y - 2  # top and bottom edges
    cols = _srv_cols()
    # Keep the columns that matter when the terminal is narrow: drop from
    # the least important (rightmost) while the name/state area needs room.
    while cols and (sum(col.width + 2 for col in cols) + 22) > (c.w - 3):
        cols.pop()
    ends: dict[str, int] = {}
    x = c.w - 3
    for col in reversed(cols):  # rightmost column sits at the right edge
        ends[col.label] = x
        x -= col.width + 2
    state_end = (ends[cols[0].label] - cols[0].width - 1) if cols else c.w - 3
    name_room = max(4, state_end - 4 - 10)

    c.edge(
        y,
        box,
        True,
        [("All servers", "TITLE", True)],
        [(col.label, "DIM", False) for col in reversed(cols)],
    )
    for r in range(y + 1, c.h - 1):
        c.sides(r, box)

    n = len(views)
    first = max(0, min(ctx.index - rows_room // 2, n - rows_room))
    for k, v in enumerate(views[first : first + rows_room]):
        i = first + k
        r = y + 1 + k
        sel = i == ctx.index
        stale = v.state in (ServerState.DOWN, ServerState.STALE, ServerState.UNKNOWN)
        c.put(1, r, "▶" if sel else " ", "TITLE", True)
        name = v.server or v.url or "?"
        if len(name) > name_room:
            name = name[: max(0, name_room - 1)] + "…"
        c.text(3, r, name, "TITLE" if sel else "VALUE", sel)
        c.text_right(state_end, r, v.state.value, STATE_ROLE[v.state], sel)
        for col in cols:
            c.text_right(ends[col.label], r, col.width_of(v), "DIM" if stale else col.role(v))
        c.click(0, r, c.w, r + 1, f"open:{i}")
    if not views:
        c.text(2, y + 1, fmt.NO_DATA, "DIM")
    hint_row = y + 1 + len(views) + 1
    if hint_row < c.h - 2:
        c.text(2, hint_row, "↑↓ select · enter: open · click a row", "DIM")
    # Footer: totals across every server, so the page answers "how is the
    # whole box doing" at a glance.
    tgs = [v.tg_live for v in views if not nan(v.tg_live)]
    runs = [v.running for v in views if not nan(v.running)]
    waits = [v.waiting for v in views if not nan(v.waiting)]
    total = f"Σ TG {fmt.speed(sum(tgs))} tok/s" if tgs else "Σ TG " + fmt.NO_DATA
    detail = ""
    if runs:
        detail = f"{SEP}run {sum(runs):.0f}"
        if waits:
            detail += f"{SEP}wait {sum(waits):.0f}"
    pos = f"{ctx.index + 1}/{len(views)}" if views else "0/0"
    room = c.w - 12 - len(pos) - 4
    mid = total + detail
    if len(mid) > room:
        mid = total
    row = c.h - 1
    spans = c.edge(row, box, False, [("≡", "TITLE", True), (mid, "LABEL", False)], [(pos, "DIM", False)])
    if spans:
        c.click(0, row, spans[0][1] + 2, row + 1, "settings")


# -- settings ----------------------------------------------------------------

POLL_OPTIONS = [1, 2, 3, 5, 10, 30]
ROTATE_OPTIONS = [0, 10, 30, 60, 300]


@dataclass
class SettingsState:
    poll_s: float
    rotate_s: int
    paint_bg: bool
    theme_slug: str
    themes: list[tuple[str, str]] = field(default_factory=list)  # (slug, label)
    cursor: int = 0  # 0 refresh, 1 rotate, 2 background, 3+ theme index
    scroll: int = 0

    @property
    def n_items(self) -> int:
        return 3 + len(self.themes)


def settings(c: Canvas, s: SettingsState, ctx: Ctx) -> int:
    """Returns the number of theme columns (for ←/→ navigation)."""
    if c.w < MIN_W or c.h < MIN_H:
        too_small(c)
        return 1
    box, list_box = "BORDER", "BOX2"
    c.box(0, 6, box, "Settings", [("esc: done", "DIM", False)])

    def stepper(row: int, idx: int, label: str, value: str, action: str) -> None:
        sel = s.cursor == idx
        c.text(2, row, ("▶ " if sel else "  ") + label, "TITLE" if sel else "LABEL", sel)
        mid = max(c.w // 2, 30)
        ax, bx = max(17, mid - 12), mid + 12
        c.put(ax, row, "‹", "TITLE", True)
        c.text(mid - len(value) // 2, row, value, "VALUE", True)
        c.put(bx, row, "›", "TITLE", True)
        c.click(1, row, mid, row + 1, f"{action}:-1")
        c.click(mid, row, c.w - 1, row + 1, f"{action}:1")

    stepper(1, 0, "Refresh", f"{s.poll_s:g} s", "poll")
    stepper(2, 1, "Auto-rotate", "off" if not s.rotate_s else fmt.span(s.rotate_s), "rotate")
    stepper(3, 2, "Background", "theme" if s.paint_bg else "terminal", "bg")
    w = Writer(c, 4, 2)
    w.put("TG ", "DECODE", True).put("101.3", "DECODE").put(SEP, "DIM")
    w.put("PP ", "PREFILL", True).put("3,354", "PREFILL").put(SEP, "DIM").put("KV ", "LABEL")
    bar(c, w.x, 4, 3, 50, "GOOD")
    w.to(w.x + 3).put(SEP, "DIM").put("OK", "STATE_OK", True).put(" ").put("STALE", "STATE_WAIT", True).put(" ")
    w.put("DOWN", "STATE_BAD", True)

    col_w = 26
    ncols = max(1, (c.w - 4) // col_w)
    list_top = 7
    list_rows = c.h - 1 - list_top
    per_page = list_rows * ncols
    sel_theme = s.cursor - 3
    if sel_theme >= 0:
        page = sel_theme // per_page
        s.scroll = page
    pages = max(1, math.ceil(len(s.themes) / per_page))
    s.scroll = min(s.scroll, pages - 1)
    c.edge(
        list_top - 1,
        list_box,
        True,
        [("Themes", "TITLE", True)],
        [(f"{s.scroll + 1}/{pages}", "DIM", False)],
    )
    for r in range(list_top, c.h - 1):
        c.sides(r, list_box)
    first = s.scroll * per_page
    for k in range(per_page):
        i = first + k
        if i >= len(s.themes):
            break
        col, row = divmod(k, list_rows)
        x = 2 + col * col_w
        y = list_top + row
        slug, label = s.themes[i]
        current = slug == s.theme_slug
        cursor = s.cursor == 3 + i
        if len(label) > col_w - 3:
            label = label[: col_w - 4] + "…"
        c.put(x, y, "▶" if cursor else " ", "TITLE", True)
        c.put(x + 1, y, "●" if current else " ", "STATE_OK")
        c.text(
            x + 3,
            y,
            label,
            "TITLE" if cursor or current else "VALUE",
            cursor or current,
        )
        c.click(x, y, x + col_w, y + 1, f"theme:{i}")
    c.edge(
        c.h - 1,
        list_box,
        False,
        [("≡", "TITLE", True), ("↑↓←→ move · enter apply · esc done", "DIM", False)],
    )
    c.click(0, c.h - 1, 5, c.h, "settings")
    return list_rows


# -- overlays ----------------------------------------------------------------

HELP = [
    ("← →  h l", "previous / next server"),
    ("1-9", "jump to server"),
    ("tab  g", "overview ↔ graphs"),
    ("s  ≡", "settings · themes"),
    ("+ -", "poll faster / slower"),
    ("t", "cycle theme"),
    ("r", "poll now"),
    ("b", "blank (keeps polling)"),
    ("q", "quit"),
    ("click", "left / right third: server"),
    ("", "centre: overview ↔ graphs"),
]


def help_overlay(c: Canvas) -> None:
    w = min(c.w - 4, 56)
    h = len(HELP) + 2
    x0 = (c.w - w) // 2
    y0 = max(0, (c.h - h) // 2)
    for y in range(y0, y0 + h):
        for x in range(x0, x0 + w):
            c.put(x, y, " ")
    c.edge(
        y0,
        "BOX4",
        True,
        [("Keys", "TITLE", True)],
        [("any key", "DIM", False)],
        x0,
        x0 + w,
    )
    for i, (k, d) in enumerate(HELP):
        y = y0 + 1 + i
        c.sides(y, "BOX4", x0, x0 + w)
        c.text(x0 + 2, y, k, "DECODE", True)
        c.text(x0 + 13, y, d, "VALUE", limit=x0 + w - 1)
    c.edge(y0 + h - 1, "BOX4", False, x0=x0, x1=x0 + w)
    c.click(0, 0, c.w, c.h, "close")


def blank(c: Canvas) -> None:
    msg = "blanked · any key to wake"
    c.text((c.w - len(msg)) // 2, c.h - 1, msg, "DIM")
    c.click(0, 0, c.w, c.h, "close")
