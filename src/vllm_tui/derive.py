"""Turns successive Samples into the View the pages draw (SPEC.md §5): rates
from deltas between polls, history rings, restart detection, stream linger
and the server state."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .prom import NAN, SAMPLE_STREAMS, Engine, F, Sample, StreamPhase
from .server_info import Identity, ServerInfo

HIST_LEN = 600  # samples kept for the graphs (20 min at 2 s); pages show what fits
PEAK_LEN = 60  # KV peak covers the last 60 polls, as on the device
PCT_WINDOW_S = 300.0  # latency percentiles cover the last 5 minutes
PCT_SAMPLES = 300
WINDOW = 15
RESTART_SHOW_S = 30.0
STREAM_LINGER_S = 10.0


class ServerState(Enum):
    IDLE = "OK IDLE"
    BUSY = "OK BUSY"
    WARMING = "OK WARM"
    RESTART = "OK RESTART"
    STALE = "STALE"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"
    CONNECTING = "CONNECTING"


class StreamState(Enum):
    NO_DATA = 0
    IDLE = 1
    PREFILL = 2
    DECODING = 3
    DONE = 4


@dataclass
class StreamView:
    id: int = 0
    state: StreamState = StreamState.NO_DATA
    ctx_pct: float = NAN
    tokens: float = NAN
    ttft_s: float = NAN
    tg: float = NAN


@dataclass
class View:
    server: str = ""
    server_ip: str = ""
    url: str = ""
    rtt_ms: float = NAN
    engine: str = ""
    version: str = ""
    confirmed: bool = False
    model: str = ""
    state: ServerState = ServerState.CONNECTING

    tg_live: float = NAN
    tg_avg: float = NAN
    tg_eng: float = NAN
    tg_max: float = NAN
    pp_live: float = NAN
    pp_avg: float = NAN
    pp_eng: float = NAN
    pp_per_request: bool = False
    pp_last: float = NAN
    prefilling_s: float = NAN
    running: float = NAN
    max_batch: float = NAN
    waiting: float = NAN

    kv_pct: float = NAN
    kv_tokens: float = NAN
    kv_capacity: float = NAN
    kv_kept_pct: float = NAN
    kv_peak: float = NAN
    ctx_window: float = NAN

    streams: list[StreamView] = field(default_factory=list)

    ttft_cum: float = NAN
    ttft_live: float = NAN
    lat_cum: float = NAN
    n_requests: float = NAN
    mtp_pct: float = NAN
    mtp_accepted: float = NAN
    mtp_drafted: float = NAN
    cache_hit_pct: float = NAN
    # Flare extras (its own vocabulary: verify steps, acceptance per round,
    # verification depth, weight bandwidth)
    flare: bool = False
    verify_rate: float = NAN
    accept_per_round: float = NAN
    verify_depth: float = NAN
    weight_gbps: float = NAN
    failed: float = NAN
    reasoning_cuts: float = NAN
    reused_tokens: float = NAN
    # vLLM extras
    itl_cum: float = NAN
    queue_cum: float = NAN
    success: float = NAN
    preemptions: float = NAN
    prefix_hit_pct: float = NAN

    # Latency percentiles over the recent window (vLLM and TensorFold export
    # histogram buckets; Flare doesn't).
    ttft_p50: float = NAN
    ttft_p99: float = NAN
    e2e_p50: float = NAN
    e2e_p99: float = NAN
    queue_p50: float = NAN
    queue_p99: float = NAN
    pct_n: float = NAN
    pct_window_s: float = NAN

    tg_hist: list[float] = field(default_factory=list)
    pp_hist: list[float] = field(default_factory=list)
    kv_hist: list[float] = field(default_factory=list)
    poll_s: float = 2.0

    status: str = ""


def _isnan(x: float) -> bool:
    return math.isnan(x)


def _ratio(da: float, db: float) -> float:
    return da / db if (db > 0 and da >= 0) else NAN


def _gen_live(s: Sample, e: Engine) -> float:
    if e is Engine.TENSORFOLD:
        if s.has(F.GEN_LIVE):
            return s.get(F.GEN_LIVE)
        running = s.get(F.GEN_RUNNING) if s.has(F.GEN_RUNNING) else 0.0
        return s.get(F.GEN_TOKENS) + running
    return s.get(F.GEN_TOKENS)


def _kv_pct(s: Sample) -> float:
    if s.has(F.KV_POOL):
        return s.get(F.KV_POOL) * 100.0
    m = NAN
    for st in s.streams:
        if st.present and not _isnan(st.kv) and (_isnan(m) or st.kv > m):
            m = st.kv
    return NAN if _isnan(m) else m * 100.0


def _req_prompt(s: Sample) -> float:
    return s.get(F.REQ_PROMPT) if s.has(F.REQ_PROMPT) else s.get(F.PROMPT_TOKENS)


def _prefill_tokens(s: Sample, e: Engine) -> float:
    if s.has(F.REQ_COMPUTED):
        return s.get(F.REQ_COMPUTED)
    if e is Engine.TENSORFOLD and s.has(F.CACHED_TOKENS):
        return s.get(F.PROMPT_TOKENS) - s.get(F.CACHED_TOKENS)
    return _req_prompt(s)


def _pp_live(cur: Sample, prev: Sample, e: Engine, dt: float) -> float:
    """Prompt tokens per second as the engine works: Flare counts computed
    prefill per chunk (cache reuse excluded), so its own counter is exact.
    vLLM and TensorFold only count at request end (a spike), see SPEC.md §5."""
    if e is Engine.FLARE and cur.has(F.FLARE_PREFILL) and prev.has(F.FLARE_PREFILL):
        return _ratio(cur.get(F.FLARE_PREFILL) - prev.get(F.FLARE_PREFILL), dt)
    return _ratio(cur.get(F.PROMPT_TOKENS) - prev.get(F.PROMPT_TOKENS), dt)


def _req_gen(s: Sample) -> float:
    return s.get(F.REQ_GEN) if s.has(F.REQ_GEN) else s.get(F.GEN_TOKENS)


def _went_down(cur: Sample, prev: Sample, f: F) -> bool:
    return cur.has(f) and prev.has(f) and cur.get(f) < prev.get(f)


_PHASE = {
    StreamPhase.IDLE: StreamState.IDLE,
    StreamPhase.PREFILLING: StreamState.PREFILL,
    StreamPhase.DECODING: StreamState.DECODING,
}


def _stream_active(r: StreamView) -> bool:
    return r.state in (StreamState.PREFILL, StreamState.DECODING) or (r.state is StreamState.NO_DATA and r.ctx_pct > 0)


def _mean(sum_: float, count: float) -> float:
    return sum_ / count if count > 0 else NAN


def percentile(deltas: list[tuple[float, float]], p: float) -> float:
    """p-th percentile from cumulative histogram deltas [(le, count)].
    Linear interpolation inside the bucket; a percentile that lands in the
    +Inf bucket reads the last finite bound."""
    total = deltas[-1][1] if deltas else 0.0
    if total <= 0:
        return NAN
    rank = p / 100.0 * total
    prev_le, prev_c = 0.0, 0.0
    for le, c in deltas:
        if c >= rank:
            if math.isinf(le):
                return prev_le
            inner = c - prev_c
            return le if inner <= 0 else prev_le + (le - prev_le) * (rank - prev_c) / inner
        prev_le, prev_c = le, c
    return prev_le


class Deriver:
    def __init__(self) -> None:
        self.win: deque[Sample] = deque(maxlen=WINDOW)
        self.tg_ring: deque[float] = deque([NAN] * HIST_LEN, maxlen=HIST_LEN)
        self.pp_ring: deque[float] = deque([NAN] * HIST_LEN, maxlen=HIST_LEN)
        self.kv_ring: deque[float] = deque([NAN] * HIST_LEN, maxlen=HIST_LEN)
        self.id = Identity()
        self.info = ServerInfo()
        self.ever_ok = False
        self.last_ok = 0.0
        self.failures = 0
        self.reason = ""
        self.restarted = False
        self.restart_t = 0.0
        self.tg_peak = NAN
        self.tg_sum = 0.0
        self.tg_busy = 0
        self.last_active: list[StreamView | None] = [None] * SAMPLE_STREAMS
        self.was_active = [False] * SAMPLE_STREAMS
        self.lingering = [False] * SAMPLE_STREAMS
        self.ended_t = [0.0] * SAMPLE_STREAMS
        self.hist: deque[tuple[float, dict[str, list[tuple[float, float]]]]] = deque(maxlen=PCT_SAMPLES)
        self.last_pp_eng = NAN
        self.prefilling = False
        self.prefill_since = 0.0
        self.mean_prefix = NAN  # flare: mean cached prefix a hit restores, tokens

    # -- streams ---------------------------------------------------------
    def _live_streams(self, ctx_window: float) -> list[StreamView]:
        if not self.win:
            return []
        cur = self.win[-1]
        prev = self.win[-2] if len(self.win) >= 2 else None
        dt = (cur.t - prev.t) if prev else 0.0
        rows = []
        for i, st in enumerate(cur.streams):
            if not st.present:
                continue
            r = StreamView(
                id=i,
                state=_PHASE.get(st.phase, StreamState.NO_DATA),
                ctx_pct=NAN if _isnan(st.kv) else st.kv * 100.0,
                tokens=st.kv * ctx_window,
                ttft_s=st.ttft_s,
            )
            if prev and dt > 0:
                b = prev.streams[i]
                if st.phase is StreamPhase.DECODING and b.present and st.start_s == b.start_s:
                    r.tg = _ratio(st.gen_tokens - b.gen_tokens, dt)
            rows.append(r)
        return rows

    def _push(self, tg: float, pp: float, kv: float) -> None:
        self.tg_ring.append(tg)
        self.pp_ring.append(pp)
        self.kv_ring.append(kv)

    def _kv_now(self, s: Sample, e: Engine) -> float:
        kv = _kv_pct(s)
        if e is not Engine.FLARE:
            return kv
        shared = self._flare_kv_pct(s)
        return kv if _isnan(shared) else shared

    def _flare_kv_pct(self, s: Sample) -> float:
        """Flare reports the KV a live request owns privately; the context it
        resumed is shared with the checkpoint that owns it, so vLLM would count
        those blocks as used while flare files them under `kept`. Estimate that
        share from the cache itself: the mean prefix a lookup restores times
        the requests that are live now."""
        if not (s.has(F.FLARE_TOTAL_PAGES) and s.has(F.FLARE_PAGE_TOKENS) and s.has(F.KV_POOL)):
            return NAN
        total, page = s.get(F.FLARE_TOTAL_PAGES), s.get(F.FLARE_PAGE_TOKENS)
        if total <= 0 or page <= 0:
            return NAN
        if len(self.win) >= 2:
            prev = self.win[-2]
            if s.has(F.FLARE_REUSED) and prev.has(F.FLARE_REUSED) and s.has(F.FLARE_HITS) and prev.has(F.FLARE_HITS):
                reused = s.get(F.FLARE_REUSED) - prev.get(F.FLARE_REUSED)
                hits = s.get(F.FLARE_HITS) - prev.get(F.FLARE_HITS)
                if hits > 0 and reused >= 0:
                    self.mean_prefix = reused / hits
        if _isnan(self.mean_prefix) and s.has(F.FLARE_REUSED) and s.has(F.FLARE_HITS) and s.get(F.FLARE_HITS) > 0:
            # Seed from the server's own history before the first new hit.
            self.mean_prefix = s.get(F.FLARE_REUSED) / s.get(F.FLARE_HITS)
        if _isnan(self.mean_prefix):
            return NAN
        running = s.get(F.RUNNING) if s.has(F.RUNNING) else 0.0
        free = s.get(F.FLARE_FREE_PAGES) if s.has(F.FLARE_FREE_PAGES) else 0.0
        live = s.get(F.KV_POOL) * total + running * self.mean_prefix / page
        return max(0.0, min(1.0, min(live, total - free) / total)) * 100.0

    # -- events ----------------------------------------------------------
    def on_sample(self, s: Sample, ident: Identity, info: ServerInfo) -> bool:
        """A successful /metrics poll. Returns True on an engine restart."""
        self.id = ident
        self.info = info
        self.ever_ok = True
        self.last_ok = s.t
        self.failures = 0
        self.reason = ""

        restart = False
        if self.win:
            p = self.win[-1]
            restart = any(_went_down(s, p, f) for f in (F.PROMPT_TOKENS, F.GEN_TOKENS, F.TTFT_COUNT, F.E2E_COUNT))
        if restart:
            self.win.clear()
            self.hist.clear()
            self.tg_peak = NAN
            self.tg_sum = 0.0
            self.tg_busy = 0
            self.restarted = True
            self.restart_t = s.t
            self.mean_prefix = NAN
            self.was_active = [False] * SAMPLE_STREAMS
            self.lingering = [False] * SAMPLE_STREAMS

        self.win.append(s)
        if s.hist:
            self.hist.append((s.t, s.hist))
        e = ident.engine
        tg = pp = NAN
        # Flare completes prefill in chunks (4096 tokens), so Δprefill/Δt is a
        # real live prefill rate. vLLM and TensorFold only count at the end,
        # so for them PP is per finished request.
        per_request = e in (Engine.VLLM, Engine.TENSORFOLD)
        if len(self.win) >= 2:
            prev = self.win[-2]
            dt = s.t - prev.t
            if dt > 0:
                tg = _ratio(_gen_live(s, e) - _gen_live(prev, e), dt)
                pp = _pp_live(s, prev, e, dt)
            pp_eng = _ratio(
                _prefill_tokens(s, e) - _prefill_tokens(prev, e),
                s.get(F.PREFILL_S) - prev.get(F.PREFILL_S),
            )
            if not _isnan(pp_eng):
                self.last_pp_eng = pp_eng
            if per_request:
                pp = pp_eng if not _isnan(pp_eng) else (NAN if _isnan(pp) else 0.0)
        if tg > 0:
            if _isnan(self.tg_peak) or tg > self.tg_peak:
                self.tg_peak = tg
            self.tg_sum += tg
            self.tg_busy += 1
        kv = self._kv_now(s, e)
        # Prefill detection is engine-independent: a request is running,
        # nothing has been generated since the last poll, and its cache is
        # filling. Flare, vLLM and TensorFold all report enough for this.
        prefilling = e is not Engine.UNKNOWN and s.get(F.RUNNING) > 0 and tg == 0 and kv > 0
        if prefilling and not self.prefilling:
            self.prefill_since = s.t
        self.prefilling = prefilling
        self._push(tg, pp, kv)

        rows = self._live_streams(info.context_length)
        seen = [False] * SAMPLE_STREAMS
        for r in rows:
            i = r.id
            seen[i] = True
            if _stream_active(r):
                self.last_active[i] = r
                self.was_active[i] = True
                self.lingering[i] = False
            elif self.was_active[i]:
                self.was_active[i] = False
                self.lingering[i] = True
                self.ended_t[i] = s.t
        for i in range(SAMPLE_STREAMS):
            if not seen[i]:
                self.was_active[i] = self.lingering[i] = False
        return restart

    def on_failure(self, reason: str) -> None:
        self.failures += 1
        self.reason = reason or "error"
        self._push(NAN, NAN, NAN)

    def _percentiles(self, v: View, now: float) -> None:
        """p50/p99 of TTFT, end-to-end and queue time for the requests that
        finished in the recent window: deltas of the engines' histogram
        buckets between the oldest sample in the window and now."""
        if len(self.hist) < 2:
            return
        cutoff = now - PCT_WINDOW_S
        base = next(((t, h) for t, h in self.hist if t >= cutoff), self.hist[0])
        base_t, base_h = base
        cur_h = self.hist[-1][1]
        v.pct_window_s = now - base_t
        for fam, p50, p99 in (
            ("ttft", "ttft_p50", "ttft_p99"),
            ("e2e", "e2e_p50", "e2e_p99"),
            ("queue", "queue_p50", "queue_p99"),
        ):
            b, c = base_h.get(fam), cur_h.get(fam)
            if not b or not c:
                continue
            bd, cd = dict(b), dict(c)
            les = sorted(set(bd) | set(cd))
            deltas: list[tuple[float, float]] = []
            prev = 0.0
            ok = True
            for le in les:
                if le not in bd or le not in cd:
                    ok = False
                    break
                d = cd[le] - bd[le]
                if d < 0:  # counter went down: restart between the samples
                    ok = False
                    break
                prev = max(d, prev)
                deltas.append((le, prev))
            if not ok or not deltas or deltas[-1][1] <= 0:
                continue
            setattr(v, p50, percentile(deltas, 50))
            setattr(v, p99, percentile(deltas, 99))
            if fam == "ttft":
                v.pct_n = deltas[-1][1]

    def _flare(self, v: View, cur: Sample, have_prev: bool) -> None:
        """Flare's own dashboard vocabulary (docs/api.md): prefill and
        generation rates, verify steps, acceptance per draft-offering round,
        verification depth, and the weight bandwidth decode implies."""
        v.failed = cur.get(F.FLARE_FAILED)
        v.reasoning_cuts = cur.get(F.FLARE_REASONING_CUTS)
        v.reused_tokens = cur.get(F.FLARE_REUSED)
        if not have_prev:
            return
        prev = self.win[-2]
        dt = cur.t - prev.t
        if dt > 0:
            v.verify_rate = _ratio(cur.get(F.FLARE_VERIFY_ROUNDS) - prev.get(F.FLARE_VERIFY_ROUNDS), dt)
        rounds = cur.get(F.FLARE_DRAFT_ROUNDS) - prev.get(F.FLARE_DRAFT_ROUNDS)
        if rounds > 0:
            v.accept_per_round = (cur.get(F.FLARE_ACCEPTED) - prev.get(F.FLARE_ACCEPTED)) / rounds
        vrounds = cur.get(F.FLARE_VERIF_ROUNDS) - prev.get(F.FLARE_VERIF_ROUNDS)
        if vrounds > 0:
            v.verify_depth = (cur.get(F.FLARE_VERIF_DRAFTS) - prev.get(F.FLARE_VERIF_DRAFTS)) / vrounds
        weight = cur.get(F.FLARE_WEIGHT_BYTES)
        if not math.isnan(v.verify_rate) and not math.isnan(weight) and weight > 0:
            v.weight_gbps = v.verify_rate * weight / 1e9

    # -- view ------------------------------------------------------------
    def fill(self, v: View, now: float, poll_s: float, retry_s: float) -> View:
        v.poll_s = poll_s
        v.engine = self.id.engine.value
        v.confirmed = self.id.confirmed
        v.version = self.info.version
        v.model = self.info.model or (self.win[-1].model_name if self.win else "")
        v.tg_hist = list(self.tg_ring)
        v.pp_hist = list(self.pp_ring)
        v.kv_hist = list(self.kv_ring)
        peak = NAN
        for x in v.kv_hist[-PEAK_LEN:]:
            if not _isnan(x) and (_isnan(peak) or x > peak):
                peak = x
        v.kv_peak = peak

        v.flare = self.id.engine is Engine.FLARE
        if self.win:
            cur = self.win[-1]
            e = self.id.engine
            if len(self.win) >= 2:
                prev = self.win[-2]
                dt = cur.t - prev.t
                if dt > 0:
                    v.tg_live = _ratio(_gen_live(cur, e) - _gen_live(prev, e), dt)
                    v.pp_live = _pp_live(cur, prev, e, dt)
                v.tg_eng = _ratio(
                    _req_gen(cur) - _req_gen(prev),
                    cur.get(F.DECODE_S) - prev.get(F.DECODE_S),
                )
                v.pp_eng = _ratio(
                    _prefill_tokens(cur, e) - _prefill_tokens(prev, e),
                    cur.get(F.PREFILL_S) - prev.get(F.PREFILL_S),
                )
                v.ttft_live = _ratio(
                    cur.get(F.TTFT_SUM) - prev.get(F.TTFT_SUM),
                    cur.get(F.TTFT_COUNT) - prev.get(F.TTFT_COUNT),
                )
                first = self.win[0]
                span = cur.t - first.t
                if span > 0:
                    v.pp_avg = _ratio(
                        (
                            cur.get(F.FLARE_PREFILL)
                            if e is Engine.FLARE and cur.has(F.FLARE_PREFILL)
                            else cur.get(F.PROMPT_TOKENS)
                        )
                        - (
                            first.get(F.FLARE_PREFILL)
                            if e is Engine.FLARE and first.has(F.FLARE_PREFILL)
                            else first.get(F.PROMPT_TOKENS)
                        ),
                        span,
                    )

            v.tg_max = self.tg_peak
            v.tg_avg = self.tg_sum / self.tg_busy if self.tg_busy else NAN
            v.pp_per_request = e in (Engine.VLLM, Engine.TENSORFOLD)
            v.pp_last = self.last_pp_eng
            if self.prefilling:
                v.prefilling_s = now - self.prefill_since

            v.running = cur.get(F.RUNNING)
            v.waiting = cur.get(F.WAITING)
            v.max_batch = self.info.max_batch
            if math.isnan(v.max_batch) and cur.has(F.FLARE_FREE_SLOTS):
                v.max_batch = v.running + cur.get(F.FLARE_FREE_SLOTS)
            v.ctx_window = self.info.context_length
            v.kv_pct = self._kv_now(cur, e)
            v.kv_capacity = cur.get(F.KV_POOL_TOKENS) if cur.has(F.KV_POOL_TOKENS) else v.ctx_window
            v.kv_tokens = v.kv_pct / 100.0 * v.kv_capacity
            v.kv_kept_pct = cur.get(F.KV_KEPT) * 100.0 if cur.has(F.KV_KEPT) else NAN

            rows = self._live_streams(v.ctx_window)
            for k, r in enumerate(rows):
                i = r.id
                la = self.last_active[i]
                if not _stream_active(r) and self.lingering[i] and la and now - self.ended_t[i] < STREAM_LINGER_S:
                    rows[k] = StreamView(la.id, StreamState.DONE, la.ctx_pct, la.tokens, la.ttft_s, la.tg)
            v.streams = rows

            ttft_n, e2e_n = cur.get(F.TTFT_COUNT), cur.get(F.E2E_COUNT)
            v.ttft_cum = cur.get(F.TTFT_SUM) / ttft_n if ttft_n > 0 else NAN
            v.lat_cum = cur.get(F.E2E_SUM) / e2e_n if e2e_n > 0 else NAN
            v.n_requests = ttft_n
            v.mtp_accepted = cur.get(F.ACCEPTED_TOKENS)
            v.mtp_drafted = cur.get(F.DRAFT_TOKENS)
            v.mtp_pct = v.mtp_accepted / v.mtp_drafted * 100.0 if v.mtp_drafted > 0 else NAN
            prompt = cur.get(F.PROMPT_TOKENS)
            v.cache_hit_pct = (
                cur.get(F.CACHED_TOKENS) / prompt * 100.0 if cur.has(F.CACHED_TOKENS) and prompt > 0 else NAN
            )

            if v.flare:
                self._flare(v, cur, len(self.win) >= 2)
                if not math.isnan(self.info.max_batch):
                    v.max_batch = self.info.max_batch
            self._percentiles(v, now)
            v.itl_cum = _mean(cur.get(F.ITL_SUM), cur.get(F.ITL_COUNT))
            v.queue_cum = _mean(cur.get(F.QUEUE_SUM), cur.get(F.QUEUE_COUNT))
            # Finished requests: flare and vLLM count them; TensorFold's
            # end-to-end histogram count is the same thing.
            v.success = cur.get(F.SUCCESS) if cur.has(F.SUCCESS) else (e2e_n if e2e_n > 0 else NAN)
            v.preemptions = cur.get(F.PREEMPTIONS)
            q = cur.get(F.PREFIX_QUERIES)
            v.prefix_hit_pct = cur.get(F.PREFIX_HITS) / q * 100.0 if q > 0 else NAN

        # State and footer (SPEC.md §8)
        ago = now - self.last_ok
        dot = " · "
        if not self.ever_ok and self.failures < 3:
            v.state = ServerState.CONNECTING
            v.status = "connecting…"
        elif self.failures >= 3:
            v.prefilling_s = NAN
            v.state = ServerState.DOWN
            v.status = f"{self.reason}{dot}retry {math.ceil(retry_s)}s"
        elif self.id.engine is Engine.UNKNOWN:
            v.state = ServerState.UNKNOWN
            seen = sorted(self.win[-1].other_prefixes) if self.win else []
            v.status = "unknown engine" + (f"{dot}saw {', '.join(seen)}" if seen else "")
        elif ago > 3 * poll_s:
            v.state = ServerState.STALE
            v.status = f"stale {ago:.0f}s" + (f"{dot}{self.reason}" if self.reason else "")
        elif self.restarted and now - self.restart_t < RESTART_SHOW_S:
            v.state = ServerState.RESTART
            v.status = f"restarted {int(now - self.restart_t)}s ago"
        else:
            busy = v.running > 0 or v.waiting > 0
            v.state = ServerState.WARMING if self.info.warming else ServerState.BUSY if busy else ServerState.IDLE
            v.status = f"poll {poll_s:g}s{dot}ok {ago:.1f}s ago"
            if not self.id.confirmed:
                v.status += f"{dot}unconfirmed"
        return v
