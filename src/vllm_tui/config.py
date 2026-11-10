"""Configuration (~/.config/vllm-tui/config.toml) and saved UI state
(~/.local/state/vllm-tui/state.json)."""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .poller import ServerCfg, normalize_url

HOME = Path.home()
CONFIG_PATH = Path(os.environ.get("XDG_CONFIG_HOME", HOME / ".config")) / "vllm-tui/config.toml"
STATE_PATH = Path(os.environ.get("XDG_STATE_HOME", HOME / ".local/state")) / "vllm-tui/state.json"

EXAMPLE = """\
# vllm-tui configuration
#
# Each [[server]] is one inference server: vLLM, Flare or TensorFold (the
# engine is detected from its /metrics names). Only "url" is required; a bare
# host:port works too and defaults to http://. A missing name shows the host
# up to its first dot.

poll = 2          # seconds between /metrics polls: 1, 2, 3, 5, 10 or 30
rotate = 0        # auto-advance to the next server every N seconds (0 = off)
clock = "12h"     # header clock: "12h" or "24h"

[[server]]
name = "local"
url = "http://localhost:8000"
# api_key = "sk-…"   # sent as a Bearer token, for servers started with --api-key

# [[server]]
# name = "gpu-box"
# url = "http://gpu-box.lan:8000"
"""


@dataclass
class Config:
    servers: list[ServerCfg] = field(default_factory=list)
    poll_s: float = 2.0
    rotate_s: int = 0
    clock_24h: bool = False


def parse_server_arg(arg: str) -> ServerCfg:
    """`url`, `host:port`, or `name=url`."""
    name = ""
    if "=" in arg and "://" not in arg.split("=", 1)[0]:
        name, arg = arg.split("=", 1)
    return ServerCfg(url=normalize_url(arg), name=name.strip())


def load_config(path: Path | None = None) -> Config:
    path = path or CONFIG_PATH
    cfg = Config()
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        return cfg
    cfg.poll_s = float(data.get("poll", cfg.poll_s))
    cfg.rotate_s = int(data.get("rotate", cfg.rotate_s))
    cfg.clock_24h = str(data.get("clock", "12h")).lower().startswith("24")
    for s in data.get("server", []):
        if not isinstance(s, dict) or not s.get("url"):
            continue
        cfg.servers.append(
            ServerCfg(
                url=normalize_url(str(s["url"])),
                name=str(s.get("name", "")),
                api_key=str(s.get("api_key", "")),
            )
        )
    return cfg


def write_example(path: Path | None = None) -> Path:
    path = path or CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(EXAMPLE)
    return path


def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(STATE_PATH)
    except OSError:
        pass
