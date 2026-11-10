"""Slow-changing facts about a server: /health, /version, /v1/models, and
engine identification (SPEC.md §4.1, §4.3)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from .prom import NAN, Engine, Sample


@dataclass
class ServerInfo:
    health_ok: bool = False
    warming: bool = False
    tf_backend: bool = False
    max_batch: float = NAN
    version: str = ""
    model: str = ""
    context_length: float = NAN


@dataclass
class Identity:
    engine: Engine = Engine.UNKNOWN
    confirmed: bool = False


def _load(body: str):
    try:
        return json.loads(body)
    except (ValueError, TypeError):
        return None


def _num(x) -> float:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else NAN


def parse_health(body: str, info: ServerInfo) -> bool:
    doc = _load(body)
    if not isinstance(doc, dict):
        return False
    if isinstance(doc.get("status"), str):
        info.health_ok = doc["status"] == "ok"
    elif isinstance(doc.get("ok"), bool):
        info.health_ok = doc["ok"]
    else:
        return False
    info.warming = bool(doc.get("warming", False))
    info.tf_backend = doc.get("backend") == "tensorfold"
    mb = _num(doc.get("max_batch_size"))
    if math.isnan(mb) and isinstance(doc.get("streams"), dict):
        mb = _num(doc["streams"].get("max"))
    if not math.isnan(mb):
        info.max_batch = mb
    cl = _num(doc.get("context_length"))
    if not math.isnan(cl):
        info.context_length = cl
    if isinstance(doc.get("version"), str):
        info.version = doc["version"]
    if not info.model and isinstance(doc.get("model"), str):
        info.model = doc["model"]
    return True


def parse_models(body: str, info: ServerInfo) -> bool:
    doc = _load(body)
    if not isinstance(doc, dict) or not isinstance(doc.get("data"), list) or not doc["data"]:
        return False
    m = doc["data"][0]
    if not isinstance(m, dict) or not isinstance(m.get("id"), str):
        return False
    info.model = m["id"]
    cl = _num(m.get("context_length"))
    if math.isnan(cl):
        cl = _num(m.get("max_model_len"))
    if not math.isnan(cl):
        info.context_length = cl
    flare = m.get("flare")  # Flare puts its serving limits in the model card
    if isinstance(flare, dict):
        mc = _num(flare.get("max_concurrency"))
        if not math.isnan(mc):
            info.max_batch = mc
    return True


def parse_version(body: str, info: ServerInfo) -> bool:
    doc = _load(body)
    if not isinstance(doc, dict) or not isinstance(doc.get("version"), str):
        return False
    info.version = doc["version"]
    return True


def identify(s: Sample, server_header: str | None, info: ServerInfo) -> Identity:
    """The metric prefix with the most recognised names wins (SPEC.md §4.3)."""
    ident = Identity()
    counts = {
        Engine.TENSORFOLD: s.n_tensorfold,
        Engine.VLLM: s.n_vllm,
        Engine.FLARE: s.n_flare,
    }
    engine, best = max(counts.items(), key=lambda kv: kv[1])
    if best == 0 or list(counts.values()).count(best) > 1:
        return ident
    ident.engine = engine
    if engine is Engine.TENSORFOLD:
        ident.confirmed = (server_header or "").lower() == "tensorfold" or bool(info.version) or info.tf_backend
    elif engine is Engine.VLLM:
        ident.confirmed = bool(info.version)  # from /version
    else:  # Flare: a second signal, not just the prefix
        ident.confirmed = s.flare_uptime or info.health_ok
    return ident
