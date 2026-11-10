"""Simulated inference servers for `vllm-tui --demo` (and for testing).

Each one serves the same endpoints a real server does (/metrics, /version or
/health, /v1/models) from a small simulation with bursts of requests, so
every part of the dashboard has something to show.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import threading
import time
from dataclasses import dataclass


@dataclass
class Req:
    prompt: int
    cached: int
    out: int
    arrived: float
    started: float = 0.0
    prefill_left: float = 0.0
    generated: float = 0.0
    first_token: float = 0.0
    decode_start: float = 0.0
    prefill_s: float = 0.0
    slot: int = -1


class Sim:
    def __init__(
        self,
        engine: str,
        model: str,
        seed: int,
        max_seqs: int,
        ctx: int,
        pool: int,
        tg_single: float,
        pp_speed: float,
        load: float,
    ) -> None:
        self.engine, self.model = engine, model
        self.rng = random.Random(seed)
        self.max_seqs, self.ctx, self.pool = max_seqs, ctx, pool
        self.tg_single, self.pp_speed, self.load = tg_single, pp_speed, load
        self.decode_weight_bytes = 48_000_000_000 if engine == "flare" else 0
        self.t = time.monotonic()
        self.t0 = self.t
        self.queue: list[Req] = []
        self.running: list[Req] = []
        self.c = dict.fromkeys(
            [
                "prompt",
                "gen",
                "gen_finished",
                "req_prompt",
                "req_gen",
                "req_computed",
                "prefill_s",
                "decode_s",
                "ttft_sum",
                "ttft_n",
                "e2e_sum",
                "e2e_n",
                "draft",
                "accepted",
                "drafts",
                "cached",
                "queries",
                "hits",
                "itl_sum",
                "itl_n",
                "queue_sum",
                "queue_n",
                "success",
                "preempt",
                "admitted",
                "failed",
                "flare_prefill",
                "verify_rounds",
                "draft_rounds",
                "verif_drafts",
                "verif_rounds",
            ],
            0.0,
        )

    def rate(self, now: float) -> float:
        # Busy and quiet phases, so the graphs breathe.
        phase = math.sin((now - self.t0) / 40.0) + 0.6 * math.sin((now - self.t0) / 13.0)
        return max(0.0, self.load * (0.55 + 0.6 * phase))

    def advance(self) -> None:
        now = time.monotonic()
        while self.t < now:
            dt = min(0.1, now - self.t)
            self.step(self.t, dt)
            self.t += dt

    def step(self, t: float, dt: float) -> None:
        rng, c = self.rng, self.c
        if rng.random() < self.rate(t) * dt:
            prompt = int(rng.lognormvariate(8.3, 1.1))
            prompt = max(64, min(prompt, self.ctx // 2))
            cached = int(prompt * rng.choice([0, 0, 0.3, 0.8, 0.95]))
            self.queue.append(Req(prompt, cached, int(rng.uniform(80, 1800)), t))
        while self.queue and len(self.running) < self.max_seqs:
            r = self.queue.pop(0)
            r.started = t
            c["admitted"] += 1
            r.prefill_left = r.prompt - r.cached
            used = {x.slot for x in self.running}
            r.slot = next(i for i in range(self.max_seqs) if i not in used)
            self.running.append(r)
            c["queue_sum"] += t - r.arrived
            c["queue_n"] += 1
            c["queries"] += r.prompt
            c["hits"] += r.cached
            if self.engine == "vllm":
                c["cached"] += r.cached
        n = len(self.running)
        per_req = self.tg_single / (1.0 + 0.05 * (n - 1)) if n else 0
        for r in list(self.running):
            if r.prefill_left > 0:
                done = min(r.prefill_left, self.pp_speed * dt)
                r.prefill_left -= done
                r.prefill_s += dt
                c["flare_prefill"] += done
                if r.prefill_left <= 0:
                    r.first_token = t + dt
                    r.decode_start = t + dt
                    c["ttft_sum"] += r.first_token - r.arrived
                    c["ttft_n"] += 1
                    if self.engine == "vllm":
                        c["prompt"] += r.prompt
                continue
            step = per_req * dt * self.rng.uniform(0.85, 1.15)
            step = min(step, r.out - r.generated)
            r.generated += step
            c["gen"] += step
            drafted = step * 1.6
            c["draft"] += drafted
            # Flare counts accepted drafts against verification draft positions
            # (a ratio), vLLM against drafted tokens (also a ratio): both ~60%.
            c["accepted"] += step * self.rng.uniform(0.5, 0.7)
            c["drafts"] += drafted / 2
            c["itl_sum"] += step / max(per_req, 1.0)
            c["itl_n"] += step
            # Flare's speculative counters: ~2.2 accepted drafts per
            # draft-offering round, ~4.3 draft positions per verification.
            c["drafts"] += step / 2.2
            c["verify_rounds"] += step / 4.3
            c["draft_rounds"] += step / (2.2 * 2.2)
            c["verif_drafts"] += step
            c["verif_rounds"] += step / 4.3
            if r.generated >= r.out:
                self.running.remove(r)
                end = t + dt
                c["gen_finished"] += r.out
                c["req_prompt"] += r.prompt
                c["req_gen"] += r.out
                c["req_computed"] += r.prompt - r.cached
                c["prefill_s"] += r.prefill_s
                c["decode_s"] += end - r.decode_start
                c["e2e_sum"] += end - r.arrived
                c["e2e_n"] += 1
                c["success"] += 1
                if self.engine == "tensorfold":
                    c["prompt"] += r.prompt
                    c["cached"] += r.cached

    def kv_tokens(self, r: Req) -> float:
        filled = r.prompt - max(0.0, r.prefill_left)
        return filled + r.generated

    # -- responses -------------------------------------------------------
    def metrics(self) -> str:
        self.advance()
        if self.engine == "flare":
            return "\n".join(self.flare_lines()) + "\n"
        c = self.c
        p = "vllm:" if self.engine == "vllm" else "tensorfold:"
        lab = f'{{model_name="{self.model}",engine="0"}}' if self.engine == "vllm" else ""
        live_gen = sum(r.generated for r in self.running)
        lines = [
            f"# HELP {p}num_requests_running Number of requests in model execution batches.",
            f"# TYPE {p}num_requests_running gauge",
            f"{p}num_requests_running{lab} {len(self.running)}",
            f"{p}num_requests_waiting{lab} {len(self.queue)}",
            f"{p}prompt_tokens_total{lab} {c['prompt']:.0f}",
            f"{p}prompt_tokens_cached_total{lab} {c['cached']:.0f}",
            f"{p}request_prefill_time_seconds_sum{lab} {c['prefill_s']:.4f}",
            f'{p}request_prefill_time_seconds_bucket{{le="+Inf"}} {c["e2e_n"]:.0f}',
            f"{p}request_decode_time_seconds_sum{lab} {c['decode_s']:.4f}",
            f"{p}time_to_first_token_seconds_sum{lab} {c['ttft_sum']:.4f}",
            f"{p}time_to_first_token_seconds_count{lab} {c['ttft_n']:.0f}",
            f"{p}e2e_request_latency_seconds_sum{lab} {c['e2e_sum']:.4f}",
            f"{p}e2e_request_latency_seconds_count{lab} {c['e2e_n']:.0f}",
            f"{p}spec_decode_num_draft_tokens_total{lab} {c['draft']:.0f}",
            f"{p}spec_decode_num_accepted_tokens_total{lab} {c['accepted']:.0f}",
        ]
        if self.engine == "vllm":
            used = sum(self.kv_tokens(r) for r in self.running)
            lines += [
                f"{p}generation_tokens_total{lab} {c['gen']:.0f}",
                f"{p}kv_cache_usage_perc{lab} {min(1.0, used / self.pool):.6f}",
                f'{p}cache_config_info{{block_size="16",cache_dtype="fp8",kv_cache_size_tokens="{self.pool}"}} 1.0',
                f"{p}request_prompt_tokens_sum{lab} {c['req_prompt']:.0f}",
                f"{p}request_generation_tokens_sum{lab} {c['req_gen']:.0f}",
                f"{p}request_prefill_kv_computed_tokens_sum{lab} {c['req_computed']:.0f}",
                f"{p}spec_decode_num_drafts_total{lab} {c['drafts']:.0f}",
                f"{p}inter_token_latency_seconds_sum{lab} {c['itl_sum']:.4f}",
                f"{p}inter_token_latency_seconds_count{lab} {c['itl_n']:.0f}",
                f"{p}request_queue_time_seconds_sum{lab} {c['queue_sum']:.4f}",
                f"{p}request_queue_time_seconds_count{lab} {c['queue_n']:.0f}",
                f'{p}request_success_total{{finished_reason="stop",model_name="{self.model}"}} {c["success"]:.0f}',
                f"{p}num_preemptions_total{lab} {c['preempt']:.0f}",
                f"{p}prefix_cache_queries_total{lab} {c['queries']:.0f}",
                f"{p}prefix_cache_hits_total{lab} {c['hits']:.0f}",
            ]
        else:
            lines += [
                f"{p}generation_tokens_total {c['gen_finished']:.0f}",
                f"{p}generation_tokens_running {live_gen:.0f}",
            ]
            by_slot = {r.slot: r for r in self.running}
            for i in range(self.max_seqs):
                r = by_slot.get(i)
                kv = self.kv_tokens(r) / self.ctx if r else 0.0
                lines.append(f'{p}kv_cache_usage_perc{{stream="{i}"}} {kv:.6f}')
                lines.append(f'{p}kv_cache_usage_ratio{{pool="{i}"}} {kv:.6f}')
            for i in range(self.max_seqs):
                r = by_slot.get(i)
                state = "idle" if not r else "prefilling" if r.prefill_left > 0 else "decoding"
                lines.append(f'{p}stream_state{{stream="{i}",state="{state}"}} 1')
                lines.append(f'{p}stream_generation_tokens{{stream="{i}"}} {r.generated if r else 0:.0f}')
                ttft = f"{r.first_token - r.arrived:.3f}" if r and r.first_token else "NaN"
                lines.append(f'{p}stream_ttft_seconds{{stream="{i}"}} {ttft}')
                lines.append(f'{p}stream_request_start_seconds{{stream="{i}"}} {r.arrived + 1.79e9 if r else 0:.3f}')
        return "\n".join(lines) + "\n"

    def flare_lines(self) -> list[str]:
        c = self.c
        pool, page = 164295, 128
        used = sum(self.kv_tokens(r) for r in self.running) / page
        snapshot = int(pool * 0.29)
        free = max(0, int(pool * 0.02) - int(used))
        cached = max(0, pool - used - free - snapshot)
        return [
            f"flare_generated_tokens_total {c['gen']:.0f}",
            f"flare_prefill_tokens_total {c['flare_prefill']:.0f}",
            f"flare_cache_reused_tokens_total {c['hits']:.0f}",
            f"flare_cache_lookups_total {c['queries']:.0f}",
            f"flare_cache_hits_total {c['queries']:.0f}",
            f"flare_active_requests {len(self.running)}",
            f"flare_pending_requests {len(self.queue)}",
            f"flare_free_request_slots {self.max_seqs - len(self.running)}",
            f"flare_requests_admitted_total {c['admitted']:.0f}",
            f"flare_requests_completed_total {c['success']:.0f}",
            f"flare_requests_failed_total {c['failed']:.0f}",
            f"flare_preemptions_total {c['preempt']:.0f}",
            f"flare_reasoning_budget_cuts_total {int(c['success'] * 0.01)}",
            f"flare_accepted_draft_tokens_total {c['accepted']:.0f}",
            f"flare_draft_request_rounds_total {c['draft_rounds']:.0f}",
            f"flare_verification_draft_tokens_total {c['verif_drafts']:.0f}",
            f"flare_verification_request_rounds_total {c['verif_rounds']:.0f}",
            f'flare_speculative_rounds_total{{batch="8"}} {c["verify_rounds"]:.0f}',
            f"flare_decode_weight_bytes {self.decode_weight_bytes}",
            f"flare_total_kv_pages {pool}",
            f"flare_free_kv_pages {free:.0f}",
            f"flare_cached_kv_pages {cached:.0f}",
            f"flare_snapshot_backing_pages {snapshot}",
            f"flare_kv_page_tokens {page}",
            f"flare_uptime_seconds {time.monotonic() - self.t0 + 86400:.0f}",
        ]

    def route(self, path: str) -> tuple[int, str, str]:
        if path == "/metrics":
            return 200, "text/plain; version=0.0.4", self.metrics()
        if path == "/v1/models":
            m = {
                "id": self.model,
                "object": "model",
                "owned_by": self.engine,
                "max_model_len": self.ctx,
            }
            if self.engine == "tensorfold":
                m["context_length"] = self.ctx
            if self.engine == "flare":
                m["flare"] = {
                    "max_concurrency": self.max_seqs,
                    "prefill_chunk_size": 4096,
                }
            return 200, "application/json", json.dumps({"object": "list", "data": [m]})
        if path == "/version" and self.engine == "vllm":
            return 200, "application/json", json.dumps({"version": "0.33.0"})
        if path == "/health":
            if self.engine == "vllm":
                return 200, "application/json", ""  # vLLM's /health is an empty 200
            if self.engine == "flare":
                return 200, "application/json", json.dumps({"status": "ok"})
            return (
                200,
                "application/json",
                json.dumps(
                    {
                        "status": "ok",
                        "version": "1.1.8",
                        "max_batch_size": self.max_seqs,
                        "warming": False,
                    }
                ),
            )
        return 404, "text/plain", "not found"


DEMO_SERVERS = [
    (
        "gpu-box",
        Sim("vllm", "qwen3.8-flash-next", 1, 64, 262144, 355241, 72.0, 9000.0, 0.35),
    ),
    (
        "dgx",
        Sim("tensorfold", "deepseek-v4.1-flash", 2, 4, 1048576, 0, 58.0, 3400.0, 0.12),
    ),
    (
        "h100-node",
        Sim("vllm", "llama-4-scout-17b", 3, 128, 131072, 1200000, 110.0, 22000.0, 0.9),
    ),
    (
        "dgx-spark",
        Sim("flare", "DeepSeek-V4.1-Flash", 4, 32, 1048576, 0, 95.0, 60000.0, 0.5),
    ),
]


async def _serve(sim: Sim, port_holder: list, ready: threading.Event) -> None:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                parts = line.decode("latin-1").split()
                path = parts[1] if len(parts) > 1 else "/"
                while await reader.readline() not in (b"\r\n", b"\n", b""):
                    pass
                code, ctype, body = sim.route(path.split("?")[0])
                data = body.encode()
                head = (
                    f"HTTP/1.1 {code} {'OK' if code == 200 else 'Not Found'}\r\n"
                    f"Content-Type: {ctype}\r\nContent-Length: {len(data)}\r\n"
                    f"Server: {'TensorFold' if sim.engine == 'tensorfold' else 'uvicorn'}\r\n\r\n"
                )
                writer.write(head.encode() + data)
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port_holder.append(server.sockets[0].getsockname()[1])
    ready.set()
    async with server:
        await server.serve_forever()


def start_demo() -> list[tuple[str, str]]:
    """Starts the simulated servers in a background thread; returns
    (name, url) for each."""
    out = []
    for name, sim in DEMO_SERVERS:
        holder: list[int] = []
        ready = threading.Event()
        t = threading.Thread(
            target=lambda s=sim, h=holder, r=ready: asyncio.run(_serve(s, h, r)),
            daemon=True,
        )
        t.start()
        ready.wait(5)
        out.append((name, f"http://127.0.0.1:{holder[0]}"))
    return out


def main() -> None:
    """`python -m vllm_tui.demo`: serve the simulations until Ctrl-C."""
    for name, url in start_demo():
        print(f"{name:10} {url}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
