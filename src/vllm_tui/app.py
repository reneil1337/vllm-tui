"""The Textual app: one full-screen widget that draws the current page."""

from __future__ import annotations

import time
from datetime import datetime

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widget import Widget

from . import pages, theme
from .canvas import Canvas
from .config import Config, save_state
from .derive import View
from .pages import POLL_OPTIONS, ROTATE_OPTIONS, Ctx, SettingsState, Strip
from .poller import Poller

ROTATE_PAUSE_S = 60.0  # any key or click pauses auto-rotate this long


class Dashboard(Widget):
    # Clicking is for the dashboard's own hit zones; don't let the terminal
    # start a text selection (Textual paints it over the whole screen).
    ALLOW_SELECT = False

    def __init__(self, tui: VllmTui) -> None:
        super().__init__()
        self.tui = tui
        self.canvas: Canvas | None = None

    def render(self) -> Text:
        w, h = self.size.width, self.size.height
        c = Canvas(max(1, w), max(1, h))
        self.tui.draw(c)
        self.canvas = c
        return Text("\n").join(c.render(self.tui.palette, self.tui.paint_bg))

    def on_click(self, event: events.Click) -> None:
        self.tui.clicked(event.x, event.y, self.size.width)


class VllmTui(App):
    CSS = """
    Screen { layers: base; overflow: hidden; }
    Dashboard { width: 100%; height: 100%; }
    """
    TITLE = "vllm-tui"
    ENABLE_COMMAND_PALETTE = False
    ALLOW_SELECT = False

    BINDINGS = (Binding("ctrl+c", "quit", "quit", show=False, priority=True),)

    def __init__(self, cfg: Config, state: dict, theme_override: str | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.poll_s = float(state.get("poll_s", cfg.poll_s))
        if self.poll_s not in POLL_OPTIONS:
            self.poll_s = min(POLL_OPTIONS, key=lambda o: abs(o - self.poll_s))
        self.rotate_s = int(state.get("rotate_s", cfg.rotate_s))
        self.paint_bg = bool(state.get("paint_bg", True))
        self.theme_slug = theme_override or state.get("theme", theme.FOLLOW)
        pages_order = ("overview", "graphs", "all")
        self.page = state.get("page") if state.get("page") in pages_order else "overview"
        self.index = int(state.get("server", 0)) if 0 <= int(state.get("server", 0)) < len(cfg.servers) else 0
        self.overlay: str | None = None  # "settings" | "help" | "blank"
        self.settings_state: SettingsState | None = None
        self.settings_rows = 1
        self.palette = theme.load(self.theme_slug)
        self.theme_sig = theme.current_signature()
        self.pollers = [Poller(s, lambda: self.poll_s) for s in cfg.servers]
        self.seen_gen = -1
        self.rotate_paused_until = 0.0
        self.last_rotate = time.monotonic()
        self.dash = Dashboard(self)

    # -- lifecycle -------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield self.dash

    def on_mount(self) -> None:
        self.apply_background()
        for p in self.pollers:
            p.start()
        self.set_interval(1.0, self.tick)
        self.set_interval(0.2, self.check_polls)

    async def on_unmount(self) -> None:
        self.save()
        for p in self.pollers:
            await p.stop()

    def save(self) -> None:
        save_state(
            {
                "poll_s": self.poll_s,
                "rotate_s": self.rotate_s,
                "paint_bg": self.paint_bg,
                "theme": self.theme_slug,
                "page": self.page,
                "server": self.index,
            }
        )

    def apply_background(self) -> None:
        if self.paint_bg:
            self.ansi_color = False
            self.screen.styles.background = self.palette["BG"]
            self.dash.styles.background = self.palette["BG"]
        else:
            self.ansi_color = True
            self.screen.styles.background = "ansi_default"
            self.dash.styles.background = "ansi_default"
        self.dash.refresh()

    def set_theme(self, slug: str) -> None:
        self.theme_slug = slug
        self.palette = theme.load(slug)
        if self.settings_state:
            self.settings_state.theme_slug = slug
        self.apply_background()
        self.save()

    # -- timers ----------------------------------------------------------
    def tick(self) -> None:
        sig = theme.current_signature()
        if sig != self.theme_sig:  # `omarchy theme set …` happened
            self.theme_sig = sig
            if self.theme_slug == theme.FOLLOW:
                self.palette = theme.load(theme.FOLLOW)
                self.apply_background()
            if self.settings_state:
                self.settings_state.themes = self.theme_list()
        now = time.monotonic()
        if (
            self.rotate_s
            and len(self.pollers) > 1
            and self.overlay is None
            and now >= self.rotate_paused_until
            and now - self.last_rotate >= self.rotate_s
        ):
            self.switch(1)
        self.dash.refresh()

    def check_polls(self) -> None:
        if not self.pollers:
            return
        gen = sum(p.generation for p in self.pollers)
        if gen != self.seen_gen:
            self.seen_gen = gen
            if self.overlay != "blank":
                self.dash.refresh()

    # -- drawing ---------------------------------------------------------
    def clock(self) -> str:
        now = datetime.now()
        if self.cfg.clock_24h:
            return now.strftime("%m-%d-%y %H:%M:%S")
        return now.strftime("%m-%d-%y ") + now.strftime("%I:%M:%S %p").lstrip("0")

    def snapshots(self) -> list[View]:
        return [p.snapshot() for p in self.pollers]

    def ctx(self, views: list[View]) -> Ctx:
        return Ctx(
            theme=self.palette,
            clock=self.clock(),
            index=self.index,
            servers=[Strip(v.server, v.state, v.tg_live) for v in views],
            page=self.page,
        )

    def draw(self, c: Canvas) -> None:
        if self.overlay == "blank":
            pages.blank(c)
            return
        views = self.snapshots()
        if self.overlay == "settings" and self.settings_state:
            self.settings_rows = pages.settings(c, self.settings_state, self.ctx(views))
            return
        if not views:
            c.text(2, 1, "no servers configured", "STATE_WAIT", True)
            return
        ctx = self.ctx(views)
        if self.page == "all":
            pages.all_servers(c, views, ctx)
        else:
            (pages.graphs if self.page == "graphs" else pages.overview)(c, views[self.index], ctx)
        if self.overlay == "help":
            pages.help_overlay(c)

    # -- actions ---------------------------------------------------------
    def touched(self) -> None:
        self.rotate_paused_until = time.monotonic() + ROTATE_PAUSE_S

    def switch(self, delta: int) -> None:
        if len(self.pollers) > 1:
            self.select((self.index + delta) % len(self.pollers))

    def select(self, i: int) -> None:
        if 0 <= i < len(self.pollers) and i != self.index:
            self.index = i
            self.last_rotate = time.monotonic()
            self.save()
        self.dash.refresh()

    def toggle_page(self) -> None:
        """overview → graphs → all servers → overview."""
        self.page = {"overview": "graphs", "graphs": "all", "all": "overview"}[self.page]
        self.save()
        self.dash.refresh()

    def set_poll(self, delta: int) -> None:
        i = POLL_OPTIONS.index(self.poll_s) if self.poll_s in POLL_OPTIONS else 1
        i = max(0, min(len(POLL_OPTIONS) - 1, i + delta))
        self.poll_s = POLL_OPTIONS[i]
        if self.settings_state:
            self.settings_state.poll_s = self.poll_s
        for p in self.pollers:
            p.wake()
        self.save()

    def set_rotate(self, delta: int) -> None:
        i = ROTATE_OPTIONS.index(self.rotate_s) if self.rotate_s in ROTATE_OPTIONS else 0
        self.rotate_s = ROTATE_OPTIONS[max(0, min(len(ROTATE_OPTIONS) - 1, i + delta))]
        self.last_rotate = time.monotonic()
        if self.settings_state:
            self.settings_state.rotate_s = self.rotate_s
        self.save()

    def toggle_bg(self) -> None:
        self.paint_bg = not self.paint_bg
        if self.settings_state:
            self.settings_state.paint_bg = self.paint_bg
        self.apply_background()
        self.save()

    def theme_list(self) -> list[tuple[str, str]]:
        cur = theme.current_name()
        follow = f"Omarchy · {theme.pretty(cur)}" if cur else "Omarchy (follow)"
        return [(theme.FOLLOW, follow)] + [(s, theme.pretty(s)) for s in theme.available()]

    def open_settings(self) -> None:
        themes = self.theme_list()
        cursor = next((3 + i for i, (s, _) in enumerate(themes) if s == self.theme_slug), 0)
        self.settings_state = SettingsState(self.poll_s, self.rotate_s, self.paint_bg, self.theme_slug, themes, cursor)
        self.overlay = "settings"
        self.dash.refresh()

    def close_overlay(self) -> None:
        self.overlay = None
        self.settings_state = None
        self.dash.refresh()

    def cycle_theme(self) -> None:
        slugs = [s for s, _ in self.theme_list()]
        i = slugs.index(self.theme_slug) if self.theme_slug in slugs else 0
        self.set_theme(slugs[(i + 1) % len(slugs)])
        self.notify(
            theme.pretty(self.theme_slug) if self.theme_slug != theme.FOLLOW else "following Omarchy",
            timeout=1.5,
        )

    def do_action(self, action: str) -> bool:
        kind, _, arg = action.partition(":")
        if kind == "server":
            self.select(int(arg))
        elif kind == "settings":
            self.close_overlay() if self.overlay == "settings" else self.open_settings()
        elif kind == "next":
            self.switch(1)
        elif kind == "page":
            self.toggle_page()
        elif kind == "open":
            self.select(int(arg))
            self.page = "overview"
            self.save()
        elif kind == "close":
            self.close_overlay()
        elif kind == "poll":
            self.set_poll(int(arg))
        elif kind == "rotate":
            self.set_rotate(int(arg))
        elif kind == "bg":
            self.toggle_bg()
        elif kind == "theme" and self.settings_state:
            i = int(arg)
            self.settings_state.cursor = 3 + i
            self.set_theme(self.settings_state.themes[i][0])
        else:
            return False
        self.dash.refresh()
        return True

    def clicked(self, x: int, y: int, width: int) -> None:
        self.touched()
        if self.screen.selections:
            self.screen.clear_selection()
        c = self.dash.canvas
        action = c.hit(x, y) if c else None
        if action and self.do_action(action):
            return
        if self.overlay:
            return
        # Touch zones from the device: left / right third switch servers,
        # the centre flips the page.
        if len(self.pollers) > 1 and x < width // 3:
            self.switch(-1)
        elif len(self.pollers) > 1 and x >= width * 2 // 3:
            self.switch(1)
        else:
            self.toggle_page()

    def on_key(self, event: events.Key) -> None:
        self.touched()
        key = event.key
        if self.overlay in ("help", "blank"):
            event.stop()
            self.close_overlay()
            return
        if self.overlay == "settings":
            event.stop()
            self.settings_key(key)
            return
        handled = True
        if key in ("left", "h", "shift+tab"):
            self.switch(-1)
        elif key in ("right", "l"):
            self.switch(1)
        elif key in ("up", "k") and self.page == "all":
            self.switch(-1)
        elif key in ("down", "j") and self.page == "all":
            self.switch(1)
        elif key in ("enter", "space") and self.page == "all":
            self.page = "overview"
            self.save()
            self.dash.refresh()
        elif key in ("tab", "space", "g", "enter"):
            self.toggle_page()
        elif key.isdigit() and key != "0":
            self.select(int(key) - 1)
        elif key in ("s", "m", "comma"):
            self.open_settings()
        elif key in ("plus", "equals_sign", "+", "="):
            self.set_poll(-1)
            self.notify(f"refresh {self.poll_s:g} s", timeout=1)
        elif key in ("minus", "-"):
            self.set_poll(1)
            self.notify(f"refresh {self.poll_s:g} s", timeout=1)
        elif key == "t":
            self.cycle_theme()
        elif key == "r":
            for p in self.pollers:
                p.wake()
        elif key == "b":
            self.overlay = "blank"
        elif key in ("q", "escape"):
            self.exit()
        elif key in ("question_mark", "?", "f1"):
            self.overlay = "help"
        else:
            handled = False
        if handled:
            event.stop()
            self.dash.refresh()

    def settings_key(self, key: str) -> None:
        s = self.settings_state
        if not s:
            return
        rows = max(1, self.settings_rows)
        if key in ("escape", "q", "s", "m"):
            self.close_overlay()
            return
        if key in ("up", "k"):
            s.cursor = max(0, s.cursor - 1)
        elif key in ("down", "j"):
            s.cursor = min(s.n_items - 1, s.cursor + 1)
        elif key in ("left", "h", "right", "l"):
            d = -1 if key in ("left", "h") else 1
            if s.cursor == 0:
                self.set_poll(d)
            elif s.cursor == 1:
                self.set_rotate(d)
            elif s.cursor == 2:
                self.toggle_bg()
            else:
                s.cursor = max(3, min(s.n_items - 1, s.cursor + d * rows))
        elif key in ("pageup", "pagedown"):
            if s.cursor >= 3:
                d = -1 if key == "pageup" else 1
                s.cursor = max(3, min(s.n_items - 1, s.cursor + d * rows * 3))
        elif key in ("enter", "space"):
            if s.cursor == 2:
                self.toggle_bg()
            elif s.cursor >= 3:
                self.set_theme(s.themes[s.cursor - 3][0])
        self.dash.refresh()
