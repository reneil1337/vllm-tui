import copy
import math
from pathlib import Path

from vllm_tui import fmt, pages, prom, theme
from vllm_tui.canvas import Canvas
from vllm_tui.derive import Deriver, ServerState, View
from vllm_tui.pages import Ctx, Strip, graphs, overview
from vllm_tui.prom import Engine, F
from vllm_tui.server_info import (
    ServerInfo,
    identify,
    parse_health,
    parse_models,
    parse_version,
)

FIX = Path(__file__).parent / "fixtures"


def load(name):
    d = FIX / name
    s = prom.parse((d / "metrics.txt").read_text())
    info = ServerInfo()
    parse_models((d / "v1_models.json").read_text(), info)
    if (d / "health.json").exists():
        parse_health((d / "health.json").read_text(), info)
    if (d / "version.json").exists():
        parse_version((d / "version.json").read_text(), info)
    return s, info


def test_vllm_fixture():
    s, info = load("vllm-0.33.0rc1-idle")
    ident = identify(s, None, info)
    assert ident.engine is Engine.VLLM and ident.confirmed
    assert s.get(F.KV_POOL_TOKENS) == 355241
    assert s.get(F.RUNNING) == 0
    assert info.model == "qwen3.8-flash-next"
    assert info.context_length == 262144


def test_tensorfold_fixture():
    s, info = load("tensorfold-1.1.8-idle")
    ident = identify(s, "TensorFold", info)
    assert ident.engine is Engine.TENSORFOLD and ident.confirmed
    assert info.max_batch == 4 and info.version == "1.1.8"
    assert s.streams[0].present


def test_unknown():
    s = prom.parse("<html>hello</html>\nfoo_bar 1\n")
    assert identify(s, None, ServerInfo()).engine is Engine.UNKNOWN


def step(prev, t, **delta):
    s = copy.deepcopy(prev)
    s.t = t
    for k, v in delta.items():
        s.v[F[k]] += v
    return s


def test_rates_and_restart():
    s0, info = load("vllm-0.33.0rc1-idle")
    ident = identify(s0, None, info)
    d = Deriver()
    s0.t = 0.0
    d.on_sample(s0, ident, info)
    s1 = step(s0, 2.0, GEN_TOKENS=200, RUNNING=1)
    assert not d.on_sample(s1, ident, info)
    v = d.fill(View(), 2.0, 2.0, 0)
    assert math.isclose(v.tg_live, 100.0)
    assert v.state is ServerState.BUSY
    assert v.tg_max == 100.0
    s2 = step(s1, 4.0, GEN_TOKENS=-1000)
    assert d.on_sample(s2, ident, info)
    v = d.fill(View(), 4.0, 2.0, 0)
    assert v.state is ServerState.RESTART and math.isnan(v.tg_live)


def test_down_after_three_failures():
    d = Deriver()
    for _ in range(3):
        d.on_failure("refused")
    v = d.fill(View(), 1.0, 2.0, 4.0)
    assert v.state is ServerState.DOWN and v.status.startswith("refused")


def test_formats():
    assert fmt.speed(100.94) == "100.9"
    assert fmt.speed(1331) == "1,331"
    assert fmt.speed(120000) == "120K"
    assert fmt.speed(1.2e6) == "1.20M"
    assert fmt.dur(0.312) == "312ms" and fmt.dur(8.16) == "8.16s"
    assert fmt.speed(math.nan) == fmt.NO_DATA


def test_pages_render_at_any_size():
    s, info = load("vllm-0.33.0rc1-idle")
    d = Deriver()
    d.on_sample(s, identify(s, None, info), info)
    v = d.fill(View(server="gpu", url="http://x:8000"), 1.0, 2.0, 0)
    t = theme.resolve(theme.FALLBACK, "x", "x")
    ctx = Ctx(
        t,
        "11-10-26 1:00:00 PM",
        0,
        [Strip("gpu", v.state, v.tg_live), Strip("b", ServerState.DOWN, math.nan)],
    )
    for w, h in [(48, 18), (48, 20), (80, 24), (120, 40), (200, 60), (20, 10)]:
        for page in (overview, graphs):
            c = Canvas(w, h)
            page(c, v, ctx)
            c.render(t, True)
            assert all(len(r) == w for r in c.chars)


def test_every_theme_resolves():
    for slug in theme.available():
        t = theme.load(slug)
        assert all(r.startswith("#") and len(r) == 7 for r in t.roles.values())


def load_flare():
    d = FIX / "flare-idle"
    info = ServerInfo()
    parse_models((d / "v1_models.json").read_text(), info)
    parse_health((d / "health.json").read_text(), info)
    return [prom.parse((d / f).read_text()) for f in ("metrics.txt", "metrics-2.txt")], info


def test_flare_detected_from_metrics():
    """A server that isn't vLLM or TensorFold is still recognised: Flare's
    own `flare_*` names identify it, with flare_uptime_seconds confirming."""
    s, info = load_flare()[0][0], load_flare()[1]
    ident = identify(s, None, info)
    assert ident.engine is Engine.FLARE and ident.confirmed
    assert s.flare_uptime and s.other_prefixes == {"flare_"}
    assert info.model == "ds41-flash-local"
    assert info.max_batch == 32  # flare.max_concurrency from /v1/models
    assert info.context_length == 1048576
    # KV use excludes cached checkpoints and snapshot backing: idle reads 0.
    assert math.isclose(s.get(F.KV_POOL), 0.0, abs_tol=1e-6)
    assert s.get(F.KV_POOL_TOKENS) == 21029760


def test_flare_numbers_are_the_same_features():
    """The pages show the same metrics for Flare: live TG, live PP, MTP %,
    cache hit, request counts — plus Flare's verify/accept/depth."""
    samples, info = load_flare()
    ident = identify(samples[0], None, info)
    d = Deriver()
    samples[0].t, samples[1].t = 0.0, 4.0
    d.on_sample(samples[0], ident, info)
    d.on_sample(samples[1], ident, info)
    v = d.fill(View(server="dgx-spark"), 4.0, 2.0, 0)
    assert v.engine == "Flare" and v.state is ServerState.IDLE
    assert v.tg_live > 100  # committed generated tokens per second
    assert v.pp_live > 1000  # Flare prefills in chunks: a real live PP
    assert not v.pp_per_request  # ... so PP is live, not per finished request
    assert 50 < v.mtp_pct < 70  # accepted drafts per draft position
    assert v.cache_hit_pct > 90
    assert v.max_batch == 32 and v.running == 0
    assert v.verify_rate > 0 and v.accept_per_round > 0 and v.verify_depth > 3
    # PP live counts computed prefill only (cache reuse excluded)
    assert v.pp_live < 400000
    # An unknown engine now says which metric prefix it saw.
    d2 = Deriver()
    u = prom.parse("other_thing_total 1\nother_more 2\n")
    d2.on_sample(u, identify(u, None, ServerInfo()), ServerInfo())
    assert d2.fill(View(), 1.0, 2.0, 0).status.startswith("unknown engine · saw other_")


def test_percentiles_from_bucket_deltas():
    """p50/p99 come from the deltas of the engines' histogram buckets, so they
    describe the requests that finished recently, not since start."""
    from vllm_tui.derive import percentile

    d = [(0.1, 2.0), (0.5, 6.0), (1.0, 9.0), (math.inf, 10.0)]
    assert math.isclose(percentile(d, 50), 0.4)
    assert math.isclose(percentile(d, 99), 1.0)  # lands in +Inf: last bound

    def text(a, b, c, inf):
        return "\n".join(
            f'vllm:time_to_first_token_seconds_bucket{{le="{le}"}} {n}'
            for le, n in (("0.1", a), ("0.5", b), ("1.0", c), ("+Inf", inf))
        )

    info = ServerInfo(model="m")
    old = prom.parse(text(0, 0, 0, 0))
    new = prom.parse(text(2, 6, 9, 10))
    ident = identify(new, None, info)
    d2 = Deriver()
    old.t, new.t = 0.0, 30.0
    d2.on_sample(old, ident, info)
    d2.on_sample(new, ident, info)
    v = d2.fill(View(), 30.0, 2.0, 0)
    assert math.isclose(v.ttft_p50, 0.4) and math.isclose(v.ttft_p99, 1.0)
    assert v.pct_n == 10 and math.isclose(v.pct_window_s, 30.0)


def test_all_servers_page():
    t = theme.resolve(theme.FALLBACK, "x", "x")
    views = [
        View(server="gpu-box", state=ServerState.BUSY, tg_live=251.6, kv_pct=9.3, running=2, max_batch=8),
        View(server="dgx", state=ServerState.DOWN, status="refused · retry 9s"),
        View(server="dgx-spark", state=ServerState.BUSY, tg_live=480.2, kv_pct=0.1, running=1, max_batch=32),
    ]
    ctx = Ctx(t, "11-10-26 1:00:00 PM", 1, [Strip(v.server, v.state, v.tg_live) for v in views])
    for w, h in [(48, 20), (80, 24), (120, 30)]:
        c = Canvas(w, h)
        pages.all_servers(c, views, ctx)
        dump = "\n".join("".join(r) for r in c.chars)
        assert "All servers" in dump
        for v in views:
            assert v.server in dump
        assert "DOWN" in dump and "TG" in dump
        assert c.hit(5, 6) == "open:0"  # row click opens that server
        c.render(t, True)


def test_flare_overview_renders():
    samples, info = load_flare()
    d = Deriver()
    ident = identify(samples[0], None, info)
    samples[0].t, samples[1].t = 0.0, 4.0
    d.on_sample(samples[0], ident, info)
    d.on_sample(samples[1], ident, info)
    v = d.fill(View(server="dgx-spark", url="http://192.168.1.5:8000"), 4.0, 2.0, 0)
    t = theme.resolve(theme.FALLBACK, "x", "x")
    ctx = Ctx(t, "11-10-26 1:00:00 PM", 0, [Strip("dgx-spark", v.state, v.tg_live)])
    for w, h in [(48, 20), (110, 30), (160, 50)]:
        c = Canvas(w, h)
        overview(c, v, ctx)
        dump = "\n".join("".join(r) for r in c.chars)
        assert "Flare" in dump and "ds41-flash-local" in dump
        assert "MTP 60" in dump or "MTP 5" in dump  # same Averages row as vLLM
        assert "Streams" in dump
    # the graph page too
    c = Canvas(110, 30)
    graphs(c, v, ctx)
    assert all(len(r) == 110 for r in c.chars)
