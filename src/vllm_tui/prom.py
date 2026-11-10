"""Prometheus text parser for vLLM / TensorFold /metrics (SPEC.md §4.2).

One poll's raw values, before any deltas. NaN = the metric wasn't in the
response.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum, IntEnum

NAN = math.nan
SAMPLE_STREAMS = 8


class Engine(Enum):
    UNKNOWN = "???"
    TENSORFOLD = "TensorFold"
    VLLM = "vLLM"
    FLARE = "Flare"


class F(IntEnum):
    RUNNING = 0
    WAITING = 1
    PROMPT_TOKENS = 2  # counter
    GEN_TOKENS = 3  # counter (TF: finished requests; vLLM: live)
    GEN_RUNNING = 4  # gauge, TF only: tokens held by live streams
    PREFILL_S = 5  # hist sum, finished requests
    DECODE_S = 6  # hist sum, finished requests
    TTFT_SUM = 7
    TTFT_COUNT = 8
    E2E_SUM = 9
    E2E_COUNT = 10
    DRAFT_TOKENS = 11
    ACCEPTED_TOKENS = 12
    DRAFTS = 13
    CACHED_TOKENS = 14
    PREEMPTIONS = 15
    KV_POOL = 16  # max over pools / unlabeled series, ratio 0–1
    KV_POOL_TOKENS = 17  # vLLM: cache_config_info{kv_cache_size_tokens}
    REQ_PROMPT = 18  # vLLM: request_prompt_tokens_sum
    REQ_GEN = 19  # vLLM: request_generation_tokens_sum
    REQ_COMPUTED = 20  # vLLM: request_prefill_kv_computed_tokens_sum
    GEN_LIVE = 21  # tensorfold_health:completion_tokens_total
    HEALTH_PREFILL_S = 22
    HEALTH_CACHED = 23
    # Terminal extras (vLLM): latency breakdown and request counters.
    ITL_SUM = 24
    ITL_COUNT = 25
    QUEUE_SUM = 26
    QUEUE_COUNT = 27
    SUCCESS = 28
    PREFIX_HITS = 29
    PREFIX_QUERIES = 30
    # flare (an OpenAI-compatible engine that exports `flare_*` metrics)
    FLARE_KV_TOTAL = 31
    FLARE_KV_FREE = 32
    FLARE_KV_CACHED = 33
    FLARE_KV_BACKING = 45
    FLARE_TOTAL_PAGES = 46
    FLARE_FREE_PAGES = 47
    FLARE_CACHED_PAGES = 48
    FLARE_PAGE_TOKENS = 49
    FLARE_PREFILL = 34
    FLARE_REUSED = 35
    FLARE_FREE_SLOTS = 36
    FLARE_VERIFY_ROUNDS = 37
    FLARE_DRAFT_ROUNDS = 38
    FLARE_VERIF_DRAFTS = 39
    FLARE_VERIF_ROUNDS = 40
    FLARE_ACCEPTED = 41
    FLARE_WEIGHT_BYTES = 42
    FLARE_FAILED = 43
    FLARE_REASONING_CUTS = 44


F_COUNT = len(F)


class StreamPhase(Enum):
    NONE = -1
    IDLE = 0
    PREFILLING = 1
    DECODING = 2


@dataclass
class StreamSample:
    present: bool = False
    kv: float = NAN
    phase: StreamPhase = StreamPhase.NONE
    gen_tokens: float = NAN
    ttft_s: float = NAN
    start_s: float = NAN


@dataclass
class Sample:
    t: float = 0.0  # monotonic seconds when the response finished
    v: list[float] = field(default_factory=lambda: [NAN] * F_COUNT)
    streams: list[StreamSample] = field(default_factory=lambda: [StreamSample() for _ in range(SAMPLE_STREAMS)])
    model_name: str = ""
    # Histogram buckets (latency percentiles): family -> [(le, cumulative
    # count)] ascending, last le is +Inf. Buckets are cumulative since start;
    # the deriver turns deltas between polls into p50/p99.
    hist: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    n_tensorfold: int = 0
    n_vllm: int = 0
    n_flare: int = 0
    flare_uptime: bool = False  # flare_uptime_seconds seen: confirms the engine
    other_prefixes: set[str] = field(default_factory=set)
    lines: int = 0
    recognised: int = 0

    def get(self, f: F) -> float:
        return self.v[f]

    def has(self, f: F) -> bool:
        return not math.isnan(self.v[f])


(
    SUM,
    MAX,
    STREAM_KV,
    STREAM_STATE,
    STREAM_GEN,
    STREAM_TTFT,
    STREAM_START,
    KV_POOL_SIZE,
) = range(8)

TABLE: dict[str, tuple[F | None, int]] = {
    "num_requests_running": (F.RUNNING, SUM),
    "num_requests_waiting": (F.WAITING, SUM),
    "prompt_tokens_total": (F.PROMPT_TOKENS, SUM),
    "generation_tokens_total": (F.GEN_TOKENS, SUM),
    "generation_tokens_running": (F.GEN_RUNNING, SUM),
    "request_prefill_time_seconds_sum": (F.PREFILL_S, SUM),
    "request_decode_time_seconds_sum": (F.DECODE_S, SUM),
    "time_to_first_token_seconds_sum": (F.TTFT_SUM, SUM),
    "time_to_first_token_seconds_count": (F.TTFT_COUNT, SUM),
    "e2e_request_latency_seconds_sum": (F.E2E_SUM, SUM),
    "e2e_request_latency_seconds_count": (F.E2E_COUNT, SUM),
    "spec_decode_num_draft_tokens_total": (F.DRAFT_TOKENS, SUM),
    "spec_decode_num_accepted_tokens_total": (F.ACCEPTED_TOKENS, SUM),
    "spec_decode_num_drafts_total": (F.DRAFTS, SUM),
    "prompt_tokens_cached_total": (F.CACHED_TOKENS, SUM),
    "request_prompt_tokens_sum": (F.REQ_PROMPT, SUM),
    "request_generation_tokens_sum": (F.REQ_GEN, SUM),
    "request_prefill_kv_computed_tokens_sum": (F.REQ_COMPUTED, SUM),
    "cache_config_info": (F.KV_POOL_TOKENS, KV_POOL_SIZE),
    "preemptions_total": (F.PREEMPTIONS, SUM),
    "num_preemptions_total": (F.PREEMPTIONS, SUM),
    "kv_cache_usage_ratio": (F.KV_POOL, MAX),
    "gpu_cache_usage_perc": (F.KV_POOL, MAX),
    "kv_cache_usage_perc": (F.KV_POOL, STREAM_KV),
    "stream_state": (None, STREAM_STATE),
    "stream_generation_tokens": (None, STREAM_GEN),
    "stream_ttft_seconds": (None, STREAM_TTFT),
    "stream_request_start_seconds": (None, STREAM_START),
    "inter_token_latency_seconds_sum": (F.ITL_SUM, SUM),
    "inter_token_latency_seconds_count": (F.ITL_COUNT, SUM),
    "request_queue_time_seconds_sum": (F.QUEUE_SUM, SUM),
    "request_queue_time_seconds_count": (F.QUEUE_COUNT, SUM),
    "request_success_total": (F.SUCCESS, SUM),
    "prefix_cache_hits_total": (F.PREFIX_HITS, SUM),
    "prefix_cache_queries_total": (F.PREFIX_QUERIES, SUM),
}

# flare: same counters under a `flare_` prefix (no colon). Flat names, so the
# table is keyed by the whole suffix after "flare_".
FLARE_TABLE: dict[str, tuple[F | None, int]] = {
    "active_requests": (F.RUNNING, SUM),  # model slots with a live request
    "pending_requests": (F.WAITING, SUM),
    "generated_tokens_total": (F.GEN_TOKENS, SUM),  # committed, live
    "prefill_tokens_total": (F.FLARE_PREFILL, SUM),  # per completed chunk: PP live
    "cache_reused_tokens_total": (F.FLARE_REUSED, SUM),
    "total_kv_tokens": (F.FLARE_KV_TOTAL, SUM),
    "free_kv_tokens": (F.FLARE_KV_FREE, SUM),
    "cached_kv_tokens": (F.FLARE_KV_CACHED, SUM),
    "total_kv_pages": (F.FLARE_TOTAL_PAGES, SUM),
    "free_kv_pages": (F.FLARE_FREE_PAGES, SUM),
    "cached_kv_pages": (F.FLARE_CACHED_PAGES, SUM),
    "kv_page_tokens": (F.FLARE_PAGE_TOKENS, MAX),
    "snapshot_backing_pages": (F.FLARE_KV_BACKING, SUM),
    "free_request_slots": (F.FLARE_FREE_SLOTS, SUM),
    # MTP, with Flare's own vocabulary (docs/api.md, "Health and metrics")
    "accepted_draft_tokens_total": (F.FLARE_ACCEPTED, SUM),
    "draft_request_rounds_total": (F.FLARE_DRAFT_ROUNDS, SUM),
    "verification_draft_tokens_total": (F.FLARE_VERIF_DRAFTS, SUM),
    "verification_request_rounds_total": (F.FLARE_VERIF_ROUNDS, SUM),
    "speculative_rounds_total": (F.FLARE_VERIFY_ROUNDS, SUM),  # verify steps
    "decode_weight_bytes": (F.FLARE_WEIGHT_BYTES, MAX),  # bytes one decode step reads
    "preemptions_total": (F.PREEMPTIONS, SUM),
    "requests_completed_total": (F.SUCCESS, SUM),
    "requests_failed_total": (F.FLARE_FAILED, SUM),
    "reasoning_budget_cuts_total": (F.FLARE_REASONING_CUTS, SUM),
    "uptime_seconds": (None, SUM),  # identity: only Flare exports this
}

# Latency histograms worth a percentile: TTFT, end to end, queue time.
# TensorFold calls its end-to-end histogram request_latency_seconds.
HIST_FAMILIES = {
    "time_to_first_token_seconds_bucket": "ttft",
    "e2e_request_latency_seconds_bucket": "e2e",
    "request_latency_seconds_bucket": "e2e",
    "request_queue_time_seconds_bucket": "queue",
}

HEALTH_TABLE: dict[str, tuple[F, int]] = {
    "completion_tokens_total": (F.GEN_LIVE, SUM),
    "prefill_seconds_total": (F.HEALTH_PREFILL_S, SUM),
    "cached_tokens_total": (F.HEALTH_CACHED, SUM),
}

# Exporter boilerplate that says nothing about the engine.
_BOILERPLATE = (
    "python_",
    "process_",
    "http_",
    "promhttp_",
    "go_",
    "node_",
    "container_",
)

_LABEL_RE = re.compile(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"\s*,?')
_LINE_RE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+(\S+)")


def parse_labels(s: str | None) -> dict[str, str]:
    if not s:
        return {}
    out = {}
    for m in _LABEL_RE.finditer(s):
        out[m.group(1)] = re.sub(r"\\(.)", r"\1", m.group(2))
    return out


def _num(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return NAN


def _finite(v: float) -> bool:
    return not (math.isnan(v) or math.isinf(v))


def _le(label: str) -> float:
    label = label.strip()
    if label in ("+Inf", "Inf", "inf"):
        return math.inf
    v = _num(label)
    return v if not math.isnan(v) else NAN


def parse(text: str, sample: Sample | None = None) -> Sample:
    s = sample or Sample()
    buckets: dict[str, dict[float, float]] = {}
    v = s.v
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] == "#":
            continue
        s.lines += 1
        m = _LINE_RE.match(line)
        if not m:
            continue
        name, labels_s, value_s = m.groups()
        if name.startswith("tensorfold:"):
            suffix, tf, health = name[11:], True, False
        elif name.startswith("tensorfold_health:"):
            suffix, tf, health = name[18:], True, True
        elif name.startswith("vllm:"):
            suffix, tf, health = name[5:], False, False
        elif name.startswith("flare_"):
            suffix, tf, health = name[6:], False, False
        else:
            if len(s.other_prefixes) < 3 and not name.startswith(_BOILERPLATE):
                s.other_prefixes.add(name.split("_", 1)[0] + "_")
            continue
        if suffix.endswith("_created"):
            continue
        labels = parse_labels(labels_s) if labels_s else {}
        if suffix.endswith("_bucket"):
            family = HIST_FAMILIES.get(suffix)
            if family and "le" in labels:
                le = _le(labels["le"])
                count = _num(value_s)
                if not math.isnan(le) and _finite(count):
                    # Sum over label sets (e.g. per model) at the same bound.
                    fam = buckets.setdefault(family, {})
                    fam[le] = fam.get(le, 0.0) + count
            continue
        if not tf and labels and not s.model_name:
            s.model_name = labels.get("model_name", "")
        flare = name.startswith("flare_")
        entry = {True: FLARE_TABLE, False: HEALTH_TABLE if health else TABLE}[flare].get(suffix)
        if entry is None:
            if len(s.other_prefixes) < 3:
                s.other_prefixes.add(name.split("_", 1)[0] + "_")
            continue
        s.recognised += 1
        if flare:
            s.n_flare += 1
            if suffix == "uptime_seconds":
                s.flare_uptime = True
        elif tf:
            s.n_tensorfold += 1
        else:
            s.n_vllm += 1
        f, combine = entry
        val = _num(value_s)
        if f is None:  # identity-only metric
            continue
        slot = -1
        if "stream" in labels:
            try:
                slot = int(labels["stream"])
            except ValueError:
                slot = -1
            if not 0 <= slot < SAMPLE_STREAMS:
                slot = -1
        if combine == SUM:
            if _finite(val):
                v[f] = val if math.isnan(v[f]) else v[f] + val
        elif combine == MAX:
            if _finite(val) and (math.isnan(v[f]) or val > v[f]):
                v[f] = val
        elif combine == STREAM_KV:
            if not _finite(val):
                continue
            if slot >= 0:
                s.streams[slot].present = True
                s.streams[slot].kv = val
            elif math.isnan(v[F.KV_POOL]) or val > v[F.KV_POOL]:
                v[F.KV_POOL] = val
        elif combine == STREAM_STATE:
            if slot < 0 or val != 1.0:
                continue
            st = labels.get("state", "")
            s.streams[slot].present = True
            s.streams[slot].phase = {
                "decoding": StreamPhase.DECODING,
                "prefilling": StreamPhase.PREFILLING,
                "idle": StreamPhase.IDLE,
            }.get(st, StreamPhase.NONE)
        elif combine == STREAM_GEN:
            if slot >= 0 and not math.isnan(val):
                s.streams[slot].present = True
                s.streams[slot].gen_tokens = val
        elif combine == STREAM_TTFT:
            if slot >= 0:
                s.streams[slot].present = True
                if not math.isnan(val):
                    s.streams[slot].ttft_s = val
        elif combine == STREAM_START:
            if slot >= 0 and not math.isnan(val):
                s.streams[slot].present = True
                s.streams[slot].start_s = val
        elif combine == KV_POOL_SIZE:
            tokens = _num(labels.get("kv_cache_size_tokens", ""))
            if _finite(tokens) and tokens > 0:
                v[F.KV_POOL_TOKENS] = tokens
    # flare: prompt tokens and cache hits are only exported split up, and KV
    # use has to be derived from the token pool.
    # flare: KV use is what no reusable state claims — not free, and not
    # owned by a cached checkpoint or a snapshot (docs/runtime-core-design.md:
    # snapshot_backing_pages "counts page ids excluded by owned snapshots").
    if s.has(F.FLARE_TOTAL_PAGES) and s.has(F.FLARE_FREE_PAGES):
        total = s.get(F.FLARE_TOTAL_PAGES)
        claimed = s.get(F.FLARE_FREE_PAGES)
        for f in (F.FLARE_CACHED_PAGES, F.FLARE_KV_BACKING):
            if s.has(f):
                claimed += s.get(f)
        if total > 0 and math.isnan(s.v[F.KV_POOL]):
            s.v[F.KV_POOL] = max(0.0, min(1.0, (total - claimed) / total))
            if s.has(F.FLARE_PAGE_TOKENS):
                s.v[F.KV_POOL_TOKENS] = total * s.get(F.FLARE_PAGE_TOKENS)
    elif s.has(F.FLARE_KV_TOTAL):
        total = s.get(F.FLARE_KV_TOTAL)
        free = s.get(F.FLARE_KV_FREE) if s.has(F.FLARE_KV_FREE) else 0.0
        cached = s.get(F.FLARE_KV_CACHED) if s.has(F.FLARE_KV_CACHED) else 0.0
        if total > 0 and math.isnan(s.v[F.KV_POOL]):
            s.v[F.KV_POOL] = max(0.0, min(1.0, (total - free - cached) / total))
            s.v[F.KV_POOL_TOKENS] = total
    if s.has(F.FLARE_PREFILL) or s.has(F.FLARE_REUSED):
        prefill = s.get(F.FLARE_PREFILL) if s.has(F.FLARE_PREFILL) else 0.0
        reused = s.get(F.FLARE_REUSED) if s.has(F.FLARE_REUSED) else 0.0
        if math.isnan(s.v[F.PROMPT_TOKENS]):
            s.v[F.PROMPT_TOKENS] = prefill + reused
        if math.isnan(s.v[F.CACHED_TOKENS]):
            s.v[F.CACHED_TOKENS] = reused
    s.hist = {fam: sorted(b.items()) for fam, b in buckets.items()}
    # flare counts MTP in its own vocabulary; expose it the way every engine
    # exposes it, so the Averages box is the same everywhere.
    if s.has(F.FLARE_ACCEPTED) and not s.has(F.ACCEPTED_TOKENS):
        s.v[F.ACCEPTED_TOKENS] = s.get(F.FLARE_ACCEPTED)
    if s.has(F.FLARE_VERIF_DRAFTS) and not s.has(F.DRAFT_TOKENS):
        s.v[F.DRAFT_TOKENS] = s.get(F.FLARE_VERIF_DRAFTS)
    # A build without the histogram or cached counter gets /health's.
    if math.isnan(v[F.PREFILL_S]):
        v[F.PREFILL_S] = v[F.HEALTH_PREFILL_S]
    if math.isnan(v[F.CACHED_TOKENS]):
        v[F.CACHED_TOKENS] = v[F.HEALTH_CACHED]
    return s
