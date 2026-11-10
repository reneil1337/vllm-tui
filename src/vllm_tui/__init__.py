"""vllm-tui: a btop-style terminal dashboard for vLLM inference servers."""

__version__ = "0.1.0"


def main() -> None:
    from .cli import main as _main

    _main()
