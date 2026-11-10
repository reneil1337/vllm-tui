"""Command line: `vllm-tui [SERVER ...]`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, theme
from .config import (
    CONFIG_PATH,
    load_config,
    load_state,
    parse_server_arg,
    write_example,
)
from .poller import ServerCfg


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="vllm-tui",
        description="btop-style terminal dashboard for vLLM, Flare and TensorFold inference servers "
        "(the engine is detected automatically). Colours follow your Omarchy theme.",
        epilog=f"Servers come from the command line, else from {CONFIG_PATH}. "
        "SERVER is a URL, host:port (port defaults to 8000), or name=URL.",
    )
    ap.add_argument(
        "servers",
        nargs="*",
        metavar="SERVER",
        help="e.g. gpu-box.lan:8000 or dgx=http://10.0.0.5:40000",
    )
    ap.add_argument("-c", "--config", type=Path, help=f"config file (default {CONFIG_PATH})")
    ap.add_argument("-p", "--poll", type=float, help="seconds between polls (1, 2, 3, 5, 10, 30)")
    ap.add_argument(
        "-t",
        "--theme",
        help="Omarchy theme slug to use instead of following the active one",
    )
    ap.add_argument(
        "-k",
        "--api-key",
        help="Bearer token for the servers given on the command line (only needed if /v1/models requires one)",
    )
    ap.add_argument(
        "--demo",
        action="store_true",
        help="monitor four simulated servers (vLLM, TensorFold, Flare)",
    )
    ap.add_argument("--init", action="store_true", help="write an example config file and exit")
    ap.add_argument(
        "--list-themes",
        action="store_true",
        help="list installed Omarchy themes and exit",
    )
    ap.add_argument("-V", "--version", action="version", version=f"vllm-tui {__version__}")
    args = ap.parse_args()

    if args.init:
        print(write_example(args.config))
        return
    if args.list_themes:
        print("\n".join(theme.available()))
        return

    cfg = load_config(args.config)
    if args.servers:
        cfg.servers = [parse_server_arg(s) for s in args.servers]
        for srv in cfg.servers:
            srv.api_key = args.api_key or srv.api_key
    if args.demo:
        from .demo import start_demo

        cfg.servers = [ServerCfg(url=url, name=name) for name, url in start_demo()]
    if not cfg.servers:
        cfg.servers = [ServerCfg(url="http://localhost:8000", name="localhost")]
    state = load_state()
    if args.poll:
        state["poll_s"] = args.poll
    if args.demo or args.servers:
        state.pop("server", None)

    from .app import VllmTui

    app = VllmTui(cfg, state, theme_override=args.theme)
    app.run()
    sys.exit(app.return_code or 0)
