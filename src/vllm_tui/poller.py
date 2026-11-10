"""One async poller per server (SPEC.md §10, "Tasks"): GET /metrics every
poll, plus /health or /version and /v1/models when due; failures back off."""

from __future__ import annotations

import asyncio
import contextlib
import socket
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from . import prom
from .derive import Deriver, View
from .prom import Engine
from .server_info import ServerInfo, identify, parse_health, parse_models, parse_version

INFO_EVERY_S = 30.0
BACKOFF_S = (2.0, 4.0, 9.0)


@dataclass
class ServerCfg:
    url: str
    name: str = ""
    api_key: str = ""

    @property
    def host(self) -> str:
        return urlsplit(self.url).hostname or self.url

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        h = self.host
        try:  # an address is its own name; a host name is shortened
            socket.inet_aton(h)
            return h
        except OSError:
            return h.split(".")[0]


def normalize_url(s: str) -> str:
    s = s.strip().rstrip("/")
    if "://" not in s:
        s = "http://" + s
    # An OpenAI api_base ends in /v1; the metrics live at the root.
    s = s.removesuffix("/v1")
    parts = urlsplit(s)
    if parts.port is None and parts.scheme == "http" and parts.hostname:
        s = f"{parts.scheme}://{parts.hostname}:8000{parts.path}"
    return s


def _chain_text(exc: BaseException) -> str:
    """Every message down the exception chain: httpx collapses multi-address
    failures into "All connection attempts failed", and the errno lives a few
    links further down."""
    parts = []
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        parts.append(str(exc))
        for sub in getattr(exc, "exceptions", ()):  # ExceptionGroup
            parts.append(_chain_text(sub))
        exc = exc.__cause__ or exc.__context__
    return " ".join(parts).lower()


def describe(exc: Exception, host: str) -> str:
    """A short reason for the footer (SPEC.md §8): dns, refused, timeout, …"""
    text = _chain_text(exc)
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.ConnectError):
        if any(k in text for k in ("name or service", "nodename", "name resolution", "getaddrinfo", "no address")):
            return f"dns {host}"
        if "refused" in text or "errno 111" in text:
            return "refused"
        if "unreachable" in text or "errno 113" in text:
            return "unreachable"
        if "timed out" in text or "errno 110" in text:
            return "timeout"
        return "connect failed"
    if isinstance(exc, httpx.RemoteProtocolError):
        return "bad response"
    return type(exc).__name__.lower()


class Poller:
    def __init__(self, cfg: ServerCfg, get_poll_s) -> None:
        self.cfg = cfg
        self.get_poll_s = get_poll_s
        self.deriver = Deriver()
        self.info = ServerInfo()
        self.ip = ""
        self.rtt_ms = float("nan")
        self.retry_at = 0.0
        self.generation = 0
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        # Flare returns JSON from /metrics when Accept prefers it; text is
        # what the parser reads (docs/api.md, "Health and metrics").
        headers = {"User-Agent": "vllm-tui", "Accept": "text/plain"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        self.client = httpx.AsyncClient(
            base_url=cfg.url,
            headers=headers,
            timeout=httpx.Timeout(4.0, connect=2.0),
            follow_redirects=True,
        )

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        await self.client.aclose()

    def wake(self) -> None:
        """Poll now (after a refresh-rate change)."""
        self._wake.set()

    async def _nap(self, s: float) -> None:
        self._wake.clear()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._wake.wait(), timeout=max(0.05, s))

    async def _json(self, path: str, parse) -> bool:
        try:
            r = await self.client.get(path)
            return r.status_code == 200 and parse(r.text, self.info)
        except (httpx.HTTPError, OSError):
            return False

    def _peer_ip(self, r: httpx.Response) -> None:
        stream = r.extensions.get("network_stream")
        try:
            addr = stream.get_extra_info("server_addr") if stream else None
        except (AttributeError, OSError):
            addr = None
        if addr:
            self.ip = addr[0]

    async def _resolve(self) -> None:
        if self.ip:
            return
        try:
            infos = await asyncio.get_running_loop().getaddrinfo(self.cfg.host, None, type=socket.SOCK_STREAM)
            if infos:
                self.ip = infos[0][4][0]
        except OSError:
            pass

    async def _run(self) -> None:
        need_models = need_info = True
        last_info = 0.0
        engine = Engine.UNKNOWN
        while True:
            start = time.monotonic()
            poll_s = self.get_poll_s()
            if need_info or start - last_info >= INFO_EVERY_S:
                ok = False
                if engine is not Engine.VLLM:
                    ok = await self._json("/health", parse_health)
                if not ok and engine is not Engine.TENSORFOLD:
                    ok = await self._json("/version", parse_version)
                need_info = not ok
                last_info = start
            if need_models:
                need_models = not await self._json("/v1/models", parse_models)

            wait = poll_s
            t0 = time.monotonic()
            try:
                r = await self.client.get("/metrics")
                self.rtt_ms = (time.monotonic() - t0) * 1000
                self._peer_ip(r)
                if r.status_code != 200:
                    raise _HttpStatus(r.status_code)
                if not r.text.strip():
                    raise _HttpStatus(0)
                sample = prom.parse(r.text)
                sample.t = time.monotonic()
                ident = identify(sample, r.headers.get("server"), self.info)
                if ident.engine is not engine and ident.engine is not Engine.UNKNOWN:
                    engine = ident.engine
                    need_info = True
                if self.deriver.on_sample(sample, ident, self.info):
                    need_models = need_info = True
            except (httpx.HTTPError, OSError, _HttpStatus) as exc:
                reason = exc.reason if isinstance(exc, _HttpStatus) else describe(exc, self.cfg.host)
                self.deriver.on_failure(reason)
                fails = self.deriver.failures
                if fails >= 3:
                    wait = BACKOFF_S[min(fails - 3, len(BACKOFF_S) - 1)]
                    need_models = need_info = True
            await self._resolve()
            self.retry_at = start + wait
            self.generation += 1
            elapsed = time.monotonic() - start
            await self._nap(wait - elapsed)

    def snapshot(self) -> View:
        now = time.monotonic()
        v = View(
            server=self.cfg.label,
            server_ip=self.ip,
            url=self.cfg.url,
            rtt_ms=self.rtt_ms,
        )
        return self.deriver.fill(v, now, self.get_poll_s(), max(0.0, self.retry_at - now))


class _HttpStatus(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.reason = f"http {code}" if code else "no metrics"
