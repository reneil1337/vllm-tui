# vllm-tui

A btop-style terminal dashboard for your **vLLM** inference servers. It also speaks **Flare** (NVIDIA's Rust runtime) and **TensorFold**, and detects which is on the other end from its metric names — you point it at `host:port` and it works out the rest. It is a terminal version of [SYSTEM-D Monitor](https://github.com/dicksondickson/system-d-monitor), an ESP32 desk display, with the same metrics, formulas, states and pages. Its colours follow your **Omarchy theme**: run `omarchy theme set …` and a running dashboard recolours itself within a second.

```
╭─┐vLLM·TUI┌──────────────────────────┐11-10-26 11:12:45 PM┌─┐⣀⣤⣶⣿ 2ms┌─╮
│ gpu-box · 10.0.0.5  http://gpu-box.lan:8000                     OK BUSY │
│ vLLM · qwen3.8-flash-next                                       v0.33.0 │
│ ▶● gpu-box 251.6   ● dgx 59.7   ● h100-node 821.8                       │
╰─────────────────────────────────────────────────────────────────────────╯
╭─┐Throughput┌───────────────────────────────────────────────────┐65s┌─╮
│ TG    ╶─╮ ╭─╴ ╶┐    ╭─╴ tok/s                    ▁▂▃▅▅▆▆▇▇▇▇▇██▇██ │
│       ╭─╯ ╰─╮  │    ├─╮ live               ▂▃▄▅▆██████████████████ │
│       ╰─╴ ╶─╯ ╶┴╴ • ╰─╯             ▁▁▂▃▄███████████████████████████ │
│ PP last 7,169/s                                                        │
│ run 4 · wait 0 · TG avg 209.2 · max 269.5                              │
│ KV 7.8% ███░░░░░░░░░░░░░░░ · peak 10.2% · 28K/355K tok                 │
╰────────────────────────────────────────────────────────────────────────╯
╭─┐Averages┌─────────────────────────────────────────────────────────────╮
│ TTFT 372ms · MTP 63.0% · cache-hit 42.2%                               │
│ ITL 16ms · queue 0ms · e2e 15.0s · done 3 · preempt 0 · prefix-hit 42% │
╰─┘≡└─┘poll 2s · ok 0.2s ago└─────────────────────────────────────┘1/3└─╯
```

- **Nothing to install on the GPU box.** It polls each server's own `/metrics` (plus `/health`, `/version` and `/v1/models`) and works out the rates itself. No Prometheus needed.
- **Same dashboard for every engine.** vLLM, Flare and TensorFold get the same boxes and the same rows; whatever an engine doesn't export reads `no data`, and engine-only detail (Flare's verify/acceptance/depth, vLLM's prefix-cache rate) goes on one extra line. The engine name is shown in the header, and that's all you need to know about it.
- **Several servers at once.** Each one is polled in the background. The server strip shows every server's state and live decode speed. Switch with ←/→, the number keys, or a click. `tab` reaches an **All servers** page: one row per instance (TG, KV, PP, run/wait, TTFT, MTP, cache-hit, latency) plus totals for the whole box.
- **Accurate numbers.** If a metric isn't exported, hasn't arrived yet, or can't be measured, the dashboard shows `no data` instead of a made-up 0. Restarts and outages show up as gaps in the graphs, not as spikes.
- **Fits any terminal size.** On a tall terminal, the graphs move onto the overview page. The graphs show as much history as the width allows.

## Run it

```bash
uv run vllm-tui --demo                      # four simulated servers (vLLM, TensorFold, Flare)
uv run vllm-tui gpu-box.lan:8000            # one server (port defaults to 8000)
uv run vllm-tui 192.168.1.20:8000/v1        # an OpenAI api_base works too
uv run vllm-tui box=gpu-box.lan:8000 dgx=http://10.0.0.60:40000
```

Install it as a command with `uv tool install .` (then just `vllm-tui`).

For a permanent server list, run `vllm-tui --init`, which writes `~/.config/vllm-tui/config.toml`:

```toml
poll = 2          # seconds: 1, 2, 3, 5, 10 or 30
rotate = 0        # auto-advance to the next server every N s (0 = off)
clock = "12h"     # or "24h"

[[server]]
name = "gpu-box"
url = "http://gpu-box.lan:8000"
# api_key = "sk-…"   # Bearer token, for servers started with --api-key
```

Servers given on the command line replace the ones in the config file. Theme, refresh rate, page and the current server are saved in `~/.local/state/vllm-tui/state.json`.

## Keys

| Key | Does |
|---|---|
| `←` `→` / `h` `l` / `1`–`9` | previous / next / pick a server |
| `tab` / `space` | overview → graphs → all servers |
| `↑` `↓` on All servers | pick a server, `enter` opens it |
| `s` (or click `≡`) | settings: refresh rate, auto-rotate, background, theme |
| `+` `-` | poll faster / slower |
| `t` | cycle theme |
| `r` | poll now |
| `b` | blank the screen (polling continues) |
| `?` | help |
| `q` | quit |

Mouse: click the left or right third to switch servers (the device's touch zones). Click the centre to flip the page. Click a server in the strip to show it.

## Themes

By default the app follows the active Omarchy theme (`~/.local/state/omarchy/current/theme/colors.toml`). Colours are mapped the way Omarchy's own btop template maps them:

| Used for | `colors.toml` key |
|---|---|
| header / Throughput / Streams / Averages boxes | `magenta` / `green` / `red` / `accent` |
| decode (TG) · prefill (PP) | `accent` · `yellow` (or whichever theme colour stands out most from decode) |
| good / warn / bad thresholds | `green` / `yellow` / `red` |
| text, labels, dim text | `foreground`, `light_foreground`, `muted` (lightened if they'd be hard to read) |
| server state OK / waiting / down | the theme's green / orange / red. Monochrome themes fall back to fixed colours, so a DOWN server is always obvious |

In Settings you can pin any installed Omarchy theme instead (stock or from `~/.config/omarchy/themes`). You can also switch the background to *terminal* to keep your terminal's own background and opacity. `vllm-tui --theme nord` pins a theme for one run.

## What the numbers mean

| Term | Meaning |
|---|---|
| **TG live** | decode tokens per real second, summed over all running requests (Δ`generation_tokens_total` / Δt) |
| **TG avg / max** | average and peak of live TG since start (or the last server restart), counting only polls where the server was generating |
| **PP last** | prefill speed of the most recently finished request (computed prompt tokens ÷ its prefill seconds; cache hits excluded). Engines don't report prefill progress while it's happening, so there's no live PP |
| **prefilling Ns** | a request is running, nothing is being generated yet, and its KV cache is filling |
| **run · wait** | running (out of the batch limit, where known) and queued requests |
| **KV / peak** | KV-cache use now, and the highest reading over the last 60 polls (`used/pool tok`). Same definition on every engine: pages the pool has handed to live requests (`1 - free/total`, like vLLM's `kv_cache_usage_perc`). Flare counts only pages a request owns privately, so the TUI adds the context it resumed — the mean cached prefix per lookup × live requests — and shows the cache's retained share as `kept` |
| **Streams** | per-slot table for TensorFold. It's hidden for vLLM, which has no per-request metrics |
| **TTFT · MTP · cache-hit** | mean time to first token · speculative-decoding acceptance · prompt tokens served from the prefix cache |
| **TTFT p50/p99 · e2e p50/p99** | latency percentiles over the requests that finished in the last 5 minutes, from the engine's histogram buckets (vLLM and TensorFold export them; Flare has no histograms, so the line falls back to its own counters) |
| **ITL · queue · e2e · done · preempt · failed** | mean inter-token latency, queue time and end-to-end latency (where exported), finished requests, preemptions, failed requests |
| **verify · accept · depth · weights** | Flare's own vocabulary: speculative verification steps per second, accepted drafts per draft-offering round, draft positions per verification step, and the weight bandwidth decode implies |
| **All servers page** | every instance side by side, the selected row's header at the top, and `Σ TG`, run and wait totals in the footer |
| **prefix-hit** | vLLM's prefix-cache hit rate, on the same extra line |
| **header** | engine (`vLLM`, `Flare` or `TensorFold`, detected automatically), model, version, and the latency to the server as braille bars `⣀⣤⣶⣿` and in ms (the device showed Wi-Fi strength here). State: `OK IDLE/BUSY/WARM`; `CONNECTING`, `OK RESTART`, `STALE`; `DOWN`, `UNKNOWN` |
| **footer** | `≡` settings, then `poll 2s · ok 0.4s ago` or the problem (`refused · retry 9s`, `dns gpu-box`, `timeout`, `http 503`, …) |

A server that goes down is retried after 2 s, then 4 s, then every 9 s.

### Supported engines

| | vLLM | Flare | TensorFold |
|---|---|---|---|
| detected by | `vllm:*` metric names, confirmed by `/version` | `flare_*` metric names, confirmed by `flare_uptime_seconds` | `tensorfold:*` metric names, confirmed by `/health` |
| live decode (TG) | `generation_tokens_total` | `generated_tokens_total` | `generation_tokens_total` + running |
| prefill | PP last, per finished request | **PP live** (Flare counts prefill per 4096-token chunk) | PP last, per finished request |
| KV | `kv_cache_usage_perc` pool | pages a live request owns, plus the resumed context it shares with its checkpoint (estimated from `cache_reused_tokens_total / cache_hits_total`); the retained share is shown as `kept` | per-stream ratio |
| MTP | accepted ÷ drafted tokens | accepted ÷ verification draft positions | accepted ÷ drafted tokens |
| streams | not exported | not exported | `stream_*` series |

No API key is needed for `/metrics` or `/health` on vLLM or Flare. With a serving key configured, `/v1/models` needs one (it supplies the model name and, on Flare, the concurrency limit); set `api_key` in the config and it's sent as a Bearer token.

## Development

```bash
uv run pytest                 # parser (all three engines), derivation, layouts at many sizes, an end-to-end app run
uv run python -m vllm_tui.demo  # serve the simulated servers on their own, for testing
```

| File | What |
|---|---|
| `prom.py` | `/metrics` parser: vLLM, Flare and TensorFold names → one set of fields |
| `server_info.py` | `/health`, `/version`, `/v1/models`, engine identification |
| `derive.py` | rates, history, restart detection, states (ported from the device's `derive.cpp`) |
| `poller.py` | one async poller per server, with backoff |
| `theme.py` | Omarchy `colors.toml` → colour roles |
| `canvas.py` | character grid: btop boxes, braille graphs, sparklines, bars, big digits |
| `pages.py` | overview, graphs, settings, help |
| `app.py` | the Textual app: keys, clicks, timers, theme watching |

## Credits

The metrics, formulas, layout and states come from [SYSTEM-D Monitor](https://github.com/dicksondickson/system-d-monitor) by Dickson Chow (MIT). Its real vLLM and TensorFold captures are in `tests/fixtures`. Flare's metric meanings follow [NVIDIA/flare-local-inference](https://github.com/NVIDIA/flare-local-inference) `docs/api.md`; `tests/fixtures/flare-idle` is a capture from a running Flare server.
