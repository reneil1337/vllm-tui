import asyncio
import shutil

import pytest

from vllm_tui import app as app_mod
from vllm_tui import config, theme
from vllm_tui.config import Config
from vllm_tui.demo import start_demo
from vllm_tui.poller import ServerCfg


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_PATH", tmp_path / "state.json")
    # A fake Omarchy "current theme" we can switch underneath the app.
    cur = tmp_path / "current/theme"
    cur.mkdir(parents=True)
    shutil.copy(theme.OMARCHY_PATH / "themes/tokyo-night/colors.toml", cur / "colors.toml") if (
        theme.OMARCHY_PATH / "themes/tokyo-night/colors.toml"
    ).exists() else (cur / "colors.toml").write_text('background = "#1a1b26"\nforeground = "#a9b1d6"\n')
    (tmp_path / "current/theme.name").write_text("tokyo-night")
    monkeypatch.setattr(theme, "CURRENT_COLORS", cur / "colors.toml")
    monkeypatch.setattr(theme, "CURRENT_NAME", tmp_path / "current/theme.name")
    return tmp_path


async def test_app_end_to_end(isolated):
    servers = [ServerCfg(url=u, name=n) for n, u in start_demo()[:2]]
    servers.append(ServerCfg(url="http://127.0.0.1:9", name="dead"))
    tui = app_mod.VllmTui(Config(servers=servers, poll_s=1), {"poll_s": 1})
    async with tui.run_test(size=(100, 34)) as pilot:
        await pilot.pause(3.5)
        assert tui.pollers[0].snapshot().state.name in ("BUSY", "IDLE")
        # keys: next server, page flip, settings, theme pick, close
        await pilot.press("right")
        assert tui.index == 1
        await pilot.press("tab")
        assert tui.page == "graphs"
        await pilot.press("3")
        await pilot.pause(8)
        assert tui.pollers[2].snapshot().state.name == "DOWN"
        await pilot.press("s")
        assert tui.overlay == "settings"
        await pilot.press("down", "down", "down", "down", "enter")
        assert tui.theme_slug != theme.FOLLOW
        await pilot.press(*["up"] * 8, "right")  # cursor back on Refresh: 1 s -> 2 s
        assert tui.poll_s == 2
        await pilot.press("escape")
        assert tui.overlay is None
        # click the centre: page flips back
        await pilot.click(offset=(50, 20))
        assert tui.page == "overview"
        # follow Omarchy: switching the theme underneath recolours the app
        tui.set_theme(theme.FOLLOW)
        before = tui.palette["BG"]
        theme.CURRENT_COLORS.write_text('background = "#ffffff"\nforeground = "#111111"\nmode = "light"\n')
        await pilot.pause(1.5)
        assert tui.palette["BG"] == "#ffffff" != before
        # terminal background
        tui.toggle_bg()
        assert tui.ansi_color is True
        await asyncio.sleep(0)
