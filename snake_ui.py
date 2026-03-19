from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Dict, Optional

import torch

from baseline_policies import AVAILABLE_POLICIES, get_policy_callable
from dqn_agent import DQNAgent
from snake_env import FoodType, MultiSnakeEnv, Snake


CELL_SIZE = 30
PADDING   = 3

# ── Palette ─────────────────────────────────────────────────────────────────
WINDOW_BG  = "#070d1a"
SURFACE_BG = "#0d1526"
CARD_BG    = "#0f1a2e"
CARD_BG2   = "#121f35"
BORDER     = "#1a3050"
GRID_BG    = "#0b1422"
GRID_DOT   = "#131f30"
TEXT_MAIN  = "#e8f0ff"
TEXT_SUB   = "#7a9bc0"
TEXT_MUTED = "#3d5470"
ACCENT     = "#22d3ee"
ACCENT_DIM = "#0e7490"
SUCCESS    = "#10b981"
DANGER     = "#f43f5e"
WARNING    = "#f59e0b"
BUTTON_BG  = "#132035"
BUTTON_HOV = "#1d3050"
INPUT_BG   = "#08111f"

SNAKE_COLORS = [
    "#22d3ee",  # cyan
    "#fb923c",  # orange
    "#c084fc",  # purple
    "#34d399",  # emerald
    "#fbbf24",  # amber
    "#f472b6",  # pink
    "#818cf8",  # indigo
    "#a3e635",  # lime
]
FOOD_COLORS = {
    FoodType.NORMAL: "#22c55e",
    FoodType.BONUS:  "#facc15",
    FoodType.RISKY:  "#ef4444",
}
DEAD_COLOR = "#1e2d3d"

CONTROLLERS = ("human",) + AVAILABLE_POLICIES

OUTCOME_LABELS = {
    "win_elimination": "VICTOIRE  (élimination)",
    "win_timeout":     "VICTOIRE  (timeout)",
    "draw_timeout":    "MATCH NUL (timeout)",
    "loss_death":      "DÉFAITE   (mort)",
    "loss_timeout":    "DÉFAITE   (timeout)",
    "ongoing":         "En cours…",
}


def _dim_color(hex_color: str, factor: float) -> str:
    factor = max(0.0, min(1.0, factor))
    r = int(int(hex_color[1:3], 16) * factor)
    g = int(int(hex_color[3:5], 16) * factor)
    b = int(int(hex_color[5:7], 16) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _blend_color(c1: str, c2: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    r = int(int(c1[1:3], 16) * (1 - t) + int(c2[1:3], 16) * t)
    g = int(int(c1[3:5], 16) * (1 - t) + int(c2[3:5], 16) * t)
    b = int(int(c1[5:7], 16) * (1 - t) + int(c2[5:7], 16) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _rounded_rect(canvas: tk.Canvas, x1: float, y1: float, x2: float, y2: float,
                   r: int = 6, **kw):
    pts = [
        x1 + r, y1,      x2 - r, y1,
        x2,     y1,      x2,     y1 + r,
        x2,     y2 - r,  x2,     y2,
        x2 - r, y2,      x1 + r, y2,
        x1,     y2,      x1,     y2 - r,
        x1,     y1 + r,  x1,     y1,
    ]
    return canvas.create_polygon(pts, smooth=True, **kw)


@dataclass
class SnakeControllerConfig:
    snake_id: int
    controller_var: tk.StringVar
    model_path_var: tk.StringVar
    row_widgets: list


class SnakeGameUI:
    def __init__(
        self,
        width: int = 15,
        height: int = 15,
        total_snakes: int = 3,
        max_steps: int = 300,
        seed: int = 42,
        model_path: Optional[str] = None,
    ) -> None:
        self.width  = width
        self.height = height
        self.max_steps = max_steps
        self.seed   = seed
        self.default_model_path = model_path or self._guess_default_model()

        self.root = tk.Tk()
        self.root.title("Multi-Snake RL")
        self.root.configure(bg=WINDOW_BG)
        self.root.geometry("1560x960")
        self.root.minsize(1200, 760)

        self.running   = False
        self.after_id: Optional[str] = None
        self.pending_direction: Optional[tuple[int, int]] = None
        self.last_reward = 0.0
        self.last_done   = False
        self.last_info:  dict = {}
        self.loaded_agents: Dict[str, DQNAgent] = {}
        self.loaded_agent_errors: list[str] = []
        self.snake_configs: list[SnakeControllerConfig] = []
        self.current_human_snake_id: Optional[int] = None
        self.current_human_warning:  Optional[str]  = None

        self.total_snakes_var = tk.IntVar(value=max(2, total_snakes))
        self.speed_var        = tk.IntVar(value=8)
        self.speed_label_var  = tk.StringVar(value="8x")
        self.status_var       = tk.StringVar(value="Prêt")

        self.env = MultiSnakeEnv(
            width=self.width, height=self.height,
            n_bots=max(1, self.total_snakes_var.get() - 1),
            max_steps=self.max_steps, seed=self.seed,
        )

        self._configure_styles()
        self._build_widgets()
        self._bind_keys()
        self.available_models = self._scan_model_files()
        self._build_snake_rows()
        self._reset_game(force_reload_models=True)

    # ── Styles ───────────────────────────────────────────────────────────────
    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        for name, bg in [("App", WINDOW_BG), ("Top", SURFACE_BG), ("Card", CARD_BG),
                         ("Card2", CARD_BG2), ("Panel", SURFACE_BG), ("Footer", SURFACE_BG)]:
            style.configure(f"{name}.TFrame", background=bg)

        style.configure("Title.TLabel",       background=SURFACE_BG, foreground=TEXT_MAIN,  font=("Segoe UI", 13, "bold"))
        style.configure("Inline.TLabel",      background=SURFACE_BG, foreground=TEXT_SUB,   font=("Segoe UI", 10))
        style.configure("MutedInline.TLabel", background=SURFACE_BG, foreground=TEXT_MUTED, font=("Segoe UI", 9))
        style.configure("CardTitle.TLabel",   background=CARD_BG,    foreground=TEXT_MAIN,  font=("Segoe UI", 11, "bold"))
        style.configure("CardText.TLabel",    background=CARD_BG,    foreground=TEXT_SUB,   font=("Segoe UI", 9))

        style.configure("Dark.TButton",
            background=BUTTON_BG, foreground=TEXT_SUB,
            bordercolor=BORDER, lightcolor=BUTTON_BG, darkcolor=BUTTON_BG,
            padding=(12, 7), relief="flat", font=("Segoe UI", 9))
        style.map("Dark.TButton",
            background=[("active", BUTTON_HOV), ("pressed", BUTTON_HOV)],
            foreground=[("active", TEXT_MAIN)])

        style.configure("Primary.TButton",
            background=ACCENT_DIM, foreground=TEXT_MAIN,
            bordercolor=ACCENT_DIM, lightcolor=ACCENT_DIM, darkcolor=ACCENT_DIM,
            padding=(14, 7), relief="flat", font=("Segoe UI", 9, "bold"))
        style.map("Primary.TButton",
            background=[("active", ACCENT), ("pressed", ACCENT)],
            foreground=[("active", WINDOW_BG), ("pressed", WINDOW_BG)])

        style.configure("Danger.TButton",
            background="#2a1018", foreground="#f87171",
            bordercolor="#3d1520", lightcolor="#2a1018", darkcolor="#2a1018",
            padding=(12, 7), relief="flat", font=("Segoe UI", 9))
        style.map("Danger.TButton",
            background=[("active", "#3d1828"), ("pressed", "#3d1828")],
            foreground=[("active", DANGER)])

        style.configure("Dark.TCombobox",
            fieldbackground=INPUT_BG, background=BUTTON_BG,
            foreground=TEXT_MAIN, arrowcolor=ACCENT,
            bordercolor=BORDER, padding=4)
        style.map("Dark.TCombobox",
            fieldbackground=[("readonly", INPUT_BG)],
            foreground=[("readonly", TEXT_MAIN)],
            bordercolor=[("focus", ACCENT)])

        style.configure("Dark.TEntry",
            fieldbackground=INPUT_BG, foreground=TEXT_MAIN,
            insertcolor=ACCENT, bordercolor=BORDER, padding=5)
        style.map("Dark.TEntry", bordercolor=[("focus", ACCENT)])

        style.configure("Dark.TSpinbox",
            fieldbackground=INPUT_BG, foreground=TEXT_MAIN,
            arrowcolor=ACCENT, bordercolor=BORDER, arrowsize=12, padding=4)
        style.map("Dark.TSpinbox", bordercolor=[("focus", ACCENT)])

        style.configure("Dark.Horizontal.TScale",
            background=SURFACE_BG, troughcolor=BUTTON_BG,
            lightcolor=ACCENT, darkcolor=ACCENT_DIM, sliderlength=18)

        style.configure("App.TNotebook",
            background=WINDOW_BG, borderwidth=0, tabmargins=[0, 4, 0, 0])
        style.configure("App.TNotebook.Tab",
            padding=(16, 8), background=SURFACE_BG, foreground=TEXT_MUTED,
            font=("Segoe UI", 10), borderwidth=0)
        style.map("App.TNotebook.Tab",
            background=[("selected", CARD_BG)],
            foreground=[("selected", ACCENT)],
            font=[("selected", ("Segoe UI", 10, "bold"))])

    # ── Widgets ──────────────────────────────────────────────────────────────
    def _build_widgets(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Header
        top = tk.Frame(self.root, bg=SURFACE_BG)
        top.grid(row=0, column=0, sticky="ew")
        tk.Frame(top, bg=ACCENT, height=2).pack(fill=tk.X, side=tk.TOP)

        inner_top = tk.Frame(top, bg=SURFACE_BG, padx=16, pady=10)
        inner_top.pack(fill=tk.X)

        # Titre
        title_frame = tk.Frame(inner_top, bg=SURFACE_BG)
        title_frame.pack(side=tk.LEFT, padx=(0, 16))
        tk.Label(title_frame, text="Multi-Snake ", bg=SURFACE_BG,
                 fg=TEXT_MAIN, font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT)
        tk.Label(title_frame, text="RL", bg=SURFACE_BG,
                 fg=ACCENT, font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT)

        tk.Frame(inner_top, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=(0, 16))

        # Snakes
        tk.Label(inner_top, text="Snakes", bg=SURFACE_BG,
                 fg=TEXT_MUTED, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        ttk.Spinbox(inner_top, from_=2, to=8, textvariable=self.total_snakes_var,
                    width=3, command=self._on_total_snakes_changed,
                    style="Dark.TSpinbox").pack(side=tk.LEFT, padx=(4, 16))

        # Vitesse
        tk.Label(inner_top, text="Vitesse", bg=SURFACE_BG,
                 fg=TEXT_MUTED, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        ttk.Scale(inner_top, from_=1, to=14, variable=self.speed_var,
                  orient=tk.HORIZONTAL, length=100,
                  style="Dark.Horizontal.TScale",
                  command=self._on_speed_changed).pack(side=tk.LEFT, padx=(4, 4))
        tk.Label(inner_top, textvariable=self.speed_label_var, bg=SURFACE_BG,
                 fg=ACCENT, font=("Consolas", 9, "bold"), width=4).pack(side=tk.LEFT, padx=(0, 16))

        tk.Frame(inner_top, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        # Boutons
        ttk.Button(inner_top, text="▶  Démarrer",  command=self.start,       style="Primary.TButton").pack(side=tk.LEFT, padx=3)
        ttk.Button(inner_top, text="⏸  Pause",     command=self.pause,       style="Dark.TButton").pack(side=tk.LEFT, padx=3)
        ttk.Button(inner_top, text="⟳  Reset",     command=lambda: self._reset_game(force_reload_models=True), style="Danger.TButton").pack(side=tk.LEFT, padx=3)
        ttk.Button(inner_top, text="›  Step",      command=self.step_once,   style="Dark.TButton").pack(side=tk.LEFT, padx=3)
        tk.Frame(inner_top, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(inner_top, text="⚙  Config",   command=lambda: self.notebook.select(self.tab_snakes), style="Dark.TButton").pack(side=tk.LEFT, padx=3)
        ttk.Button(inner_top, text="✓  Appliquer", command=self._apply_snake_setup, style="Dark.TButton").pack(side=tk.LEFT, padx=3)

        # Notebook
        self.notebook = ttk.Notebook(self.root, style="App.TNotebook")
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=12, pady=(8, 0))

        self.tab_match  = ttk.Frame(self.notebook, style="App.TFrame")
        self.tab_snakes = ttk.Frame(self.notebook, style="App.TFrame")
        self.notebook.add(self.tab_match,  text="  Arène  ")
        self.notebook.add(self.tab_snakes, text="  Configuration  ")

        self._build_match_tab()
        self._build_snakes_tab()

        # Status bar
        status_bar = tk.Frame(self.root, bg=SURFACE_BG)
        status_bar.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        tk.Frame(status_bar, bg=BORDER, height=1).pack(fill=tk.X)
        inner_status = tk.Frame(status_bar, bg=SURFACE_BG, padx=16, pady=6)
        inner_status.pack(fill=tk.X)
        self._status_label = tk.Label(
            inner_status, textvariable=self.status_var,
            bg=SURFACE_BG, fg=TEXT_MUTED, font=("Consolas", 9), anchor="w")
        self._status_label.pack(side=tk.LEFT)
        tk.Label(inner_status, text="[ESPACE] Pause  •  [Flèches / WASD] Joueur",
                 bg=SURFACE_BG, fg=TEXT_MUTED, font=("Segoe UI", 8)).pack(side=tk.RIGHT)

    def _build_match_tab(self) -> None:
        self.tab_match.columnconfigure(0, weight=3)
        self.tab_match.columnconfigure(1, weight=1, minsize=290)
        self.tab_match.rowconfigure(0, weight=1)

        # Arène
        arena_card = tk.Frame(self.tab_match, bg=CARD_BG, padx=12, pady=12)
        arena_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        arena_card.columnconfigure(0, weight=1)
        arena_card.rowconfigure(1, weight=1)

        # Titre + légende inline
        hdr = tk.Frame(arena_card, bg=CARD_BG)
        hdr.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tk.Label(hdr, text="ARÈNE", bg=CARD_BG, fg=ACCENT,
                 font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        for _, color, label in [
            (FoodType.NORMAL, FOOD_COLORS[FoodType.NORMAL], "normale"),
            (FoodType.BONUS,  FOOD_COLORS[FoodType.BONUS],  "bonus"),
            (FoodType.RISKY,  FOOD_COLORS[FoodType.RISKY],  "risquée"),
        ]:
            dot = tk.Canvas(hdr, width=10, height=10, bg=CARD_BG, highlightthickness=0)
            dot.create_oval(1, 1, 9, 9, fill=color, outline="")
            dot.pack(side=tk.LEFT, padx=(16, 3))
            tk.Label(hdr, text=label, bg=CARD_BG, fg=TEXT_MUTED,
                     font=("Segoe UI", 8)).pack(side=tk.LEFT)
        tk.Label(hdr, text="  H = humain  •  ● = modèle RL",
                 bg=CARD_BG, fg=TEXT_MUTED, font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(12, 0))

        # Canvas scrollable
        arena_wrap = tk.Frame(arena_card, bg=CARD_BG)
        arena_wrap.grid(row=1, column=0, sticky="nsew")
        arena_wrap.rowconfigure(0, weight=1)
        arena_wrap.columnconfigure(0, weight=1)

        self.arena_canvas_frame = tk.Canvas(arena_wrap, bg=CARD_BG, highlightthickness=0)
        self.arena_canvas_frame.grid(row=0, column=0, sticky="nsew")
        arena_vscroll = ttk.Scrollbar(arena_wrap, orient="vertical",   command=self.arena_canvas_frame.yview)
        arena_hscroll = ttk.Scrollbar(arena_wrap, orient="horizontal", command=self.arena_canvas_frame.xview)
        arena_vscroll.grid(row=0, column=1, sticky="ns")
        arena_hscroll.grid(row=1, column=0, sticky="ew")
        self.arena_canvas_frame.configure(yscrollcommand=arena_vscroll.set,
                                          xscrollcommand=arena_hscroll.set)

        inner = tk.Frame(self.arena_canvas_frame, bg=CARD_BG)
        self.arena_canvas_frame.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda *_: self.arena_canvas_frame.configure(
            scrollregion=self.arena_canvas_frame.bbox("all")))

        canvas_w = self.width  * CELL_SIZE
        canvas_h = self.height * CELL_SIZE
        self.canvas = tk.Canvas(inner, width=canvas_w, height=canvas_h,
                                bg=GRID_BG, highlightthickness=0)
        self.canvas.pack(anchor="nw", padx=4, pady=4)

        # Monitor
        side = tk.Frame(self.tab_match, bg=CARD_BG, padx=12, pady=12)
        side.grid(row=0, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)
        side.rowconfigure(3, weight=1)

        tk.Label(side, text="MONITOR", bg=CARD_BG, fg=ACCENT,
                 font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 10))

        self.score_canvas = tk.Canvas(side, bg=CARD_BG, highlightthickness=0)
        self.score_canvas.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        tk.Frame(side, bg=BORDER, height=1).grid(row=2, column=0, sticky="ew", pady=(0, 8))

        text_wrap = tk.Frame(side, bg=CARD_BG)
        text_wrap.grid(row=3, column=0, sticky="nsew")
        text_wrap.rowconfigure(0, weight=1)
        text_wrap.columnconfigure(0, weight=1)

        self.info_text = tk.Text(
            text_wrap, wrap="word",
            bg=INPUT_BG, fg=TEXT_MAIN, insertbackground=TEXT_MAIN,
            relief=tk.FLAT, bd=0, font=("Consolas", 9),
            padx=12, pady=10, cursor="arrow",
            selectbackground=BUTTON_HOV,
        )
        self.info_text.grid(row=0, column=0, sticky="nsew")
        info_scroll = ttk.Scrollbar(text_wrap, orient="vertical", command=self.info_text.yview)
        info_scroll.grid(row=0, column=1, sticky="ns")
        self.info_text.configure(yscrollcommand=info_scroll.set, state=tk.DISABLED)

        self.info_text.tag_configure("header",  foreground=ACCENT,    font=("Consolas", 9, "bold"), spacing1=6)
        self.info_text.tag_configure("alive",   foreground=SUCCESS)
        self.info_text.tag_configure("dead",    foreground=TEXT_MUTED)
        self.info_text.tag_configure("winner",  foreground=WARNING,   font=("Consolas", 9, "bold"))
        self.info_text.tag_configure("sub",     foreground=TEXT_SUB)
        self.info_text.tag_configure("alert",   foreground=DANGER)
        self.info_text.tag_configure("muted",   foreground=TEXT_MUTED)
        self.info_text.tag_configure("value",   foreground=TEXT_MAIN)

    def _build_snakes_tab(self) -> None:
        self.tab_snakes.columnconfigure(0, weight=1)
        self.tab_snakes.rowconfigure(1, weight=1)

        # Aide
        help_card = tk.Frame(self.tab_snakes, bg=CARD_BG2, padx=16, pady=12)
        help_card.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        hdr = tk.Frame(help_card, bg=CARD_BG2)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="⚙", bg=CARD_BG2, fg=ACCENT, font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(hdr, text="Configuration des snakes", bg=CARD_BG2,
                 fg=TEXT_MAIN, font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
        tk.Label(
            help_card,
            text="Assignez un contrôleur (human / IA / modèle RL) à chaque snake. "
                 "Choisissez un .pt dans le dropdown ou via Parcourir, puis ✓ Appliquer.",
            bg=CARD_BG2, fg=TEXT_SUB, justify=tk.LEFT, anchor="w",
            font=("Segoe UI", 9), wraplength=1200,
        ).pack(fill=tk.X, pady=(6, 0))

        # Table
        config_card = tk.Frame(self.tab_snakes, bg=CARD_BG, padx=16, pady=12)
        config_card.grid(row=1, column=0, sticky="nsew")
        config_card.rowconfigure(2, weight=1)
        config_card.columnconfigure(0, weight=1)

        table_hdr = tk.Frame(config_card, bg=CARD_BG)
        table_hdr.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        tk.Label(table_hdr, text="SNAKES", bg=CARD_BG, fg=ACCENT,
                 font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        ttk.Button(table_hdr, text="⟳  Rafraîchir les modèles",
                   style="Dark.TButton", command=self._refresh_model_list).pack(side=tk.RIGHT)

        tk.Frame(config_card, bg=BORDER, height=1).grid(row=1, column=0, sticky="ew", pady=(0, 8))

        wrap = tk.Frame(config_card, bg=CARD_BG)
        wrap.grid(row=2, column=0, sticky="nsew")
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)

        self.config_canvas = tk.Canvas(wrap, bg=CARD_BG, highlightthickness=0)
        self.config_canvas.grid(row=0, column=0, sticky="nsew")
        config_scroll = ttk.Scrollbar(wrap, orient="vertical", command=self.config_canvas.yview)
        config_scroll.grid(row=0, column=1, sticky="ns")
        self.config_canvas.configure(yscrollcommand=config_scroll.set)

        self.config_inner = tk.Frame(self.config_canvas, bg=CARD_BG)
        self._config_inner_win_id = self.config_canvas.create_window((0, 0), window=self.config_inner, anchor="nw")
        self.config_inner.bind("<Configure>",
            lambda *_: self.config_canvas.configure(scrollregion=self.config_canvas.bbox("all")))
        self.config_canvas.bind("<Configure>",
            lambda e: self.config_canvas.itemconfigure(self._config_inner_win_id, width=e.width))

    # ── Model / config helpers ───────────────────────────────────────────────
    def _build_snake_rows(self) -> None:
        saved = {
            cfg.snake_id: (cfg.controller_var.get(), cfg.model_path_var.get())
            for cfg in self.snake_configs
        }
        for child in self.config_inner.winfo_children():
            child.destroy()
        self.snake_configs.clear()

        for text, col in [("Snake", 1), ("Contrôleur", 2), ("Modèle (.pt)", 3)]:
            tk.Label(self.config_inner, text=text, bg=CARD_BG, fg=TEXT_MUTED,
                     font=("Segoe UI", 9, "bold"), anchor="w").grid(
                row=0, column=col, sticky="ew", padx=8, pady=(0, 6))
        self.config_inner.grid_columnconfigure(3, weight=1)

        total = max(2, int(self.total_snakes_var.get()))
        for snake_id in range(total):
            if snake_id in saved:
                default_ctrl, default_path = saved[snake_id]
            else:
                default_ctrl = "human" if snake_id == 0 else "heuristic"
                default_path = self.default_model_path if snake_id == 0 else ""
            controller_var = tk.StringVar(value=default_ctrl)
            model_path_var = tk.StringVar(value=default_path)

            row = snake_id + 1
            swatch_color = SNAKE_COLORS[snake_id % len(SNAKE_COLORS)]
            swatch = tk.Canvas(self.config_inner, width=16, height=16,
                               bg=CARD_BG, highlightthickness=0)
            swatch.create_oval(2, 2, 14, 14, fill=swatch_color, outline="")
            swatch.grid(row=row, column=0, padx=(8, 2), pady=6)

            tk.Label(self.config_inner, text=f"Snake {snake_id}", bg=CARD_BG,
                     fg=TEXT_MAIN, font=("Segoe UI", 10), anchor="w").grid(
                row=row, column=1, sticky="ew", padx=(2, 12), pady=6)

            combo = ttk.Combobox(self.config_inner, textvariable=controller_var,
                                 values=CONTROLLERS, state="readonly", width=14,
                                 style="Dark.TCombobox")
            combo.grid(row=row, column=2, sticky="ew", padx=8, pady=6)

            model_combo = ttk.Combobox(self.config_inner, textvariable=model_path_var,
                                       values=self.available_models, state="normal",
                                       width=40, style="Dark.TCombobox")
            model_combo.grid(row=row, column=3, sticky="ew", padx=8, pady=6)

            button = ttk.Button(self.config_inner, text="Parcourir", style="Dark.TButton",
                                command=lambda var=model_path_var: self._browse_model_for(var))
            button.grid(row=row, column=4, padx=8, pady=6)

            if snake_id < total - 1:
                tk.Frame(self.config_inner, bg=BORDER, height=1).grid(
                    row=row, column=0, columnspan=5, sticky="ew", padx=8)

            self.snake_configs.append(SnakeControllerConfig(
                snake_id, controller_var, model_path_var, [combo, model_combo, button]))

    def _browse_model_for(self, target_var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")])
        if path:
            target_var.set(path)

    def _scan_model_files(self) -> list[str]:
        base = Path(__file__).parent
        found = []
        for pt_file in sorted(base.rglob("*.pt")):
            if any(p.startswith("__") or p == ".git" for p in pt_file.parts):
                continue
            found.append(str(pt_file))
        return found

    def _refresh_model_list(self) -> None:
        self.available_models = self._scan_model_files()
        self._build_snake_rows()

    def _guess_default_model(self) -> str:
        for p in [
            Path("outputs_optimise/models/best_eval_model.pt"),
            Path("outputs_v3/models/best_eval_model.pt"),
            Path("outputs_v2/models/best_eval_model.pt"),
            Path("outputs/models/best_reward_model.pt"),
        ]:
            if p.exists():
                return str(p)
        return ""

    def _infer_hidden_dim(self, checkpoint: dict) -> int:
        q_net = checkpoint.get("q_net", {})
        for key in ("feature.0.weight", "net.0.weight"):
            if key in q_net:
                return int(q_net[key].shape[0])
        return 256

    def _load_agent_for_path(self, model_path: str) -> Optional[DQNAgent]:
        model_path = model_path.strip()
        if not model_path:
            return None
        if model_path in self.loaded_agents:
            return self.loaded_agents[model_path]
        path = Path(model_path)
        if not path.exists():
            self.loaded_agent_errors.append(f"Introuvable: {path.name}")
            return None
        try:
            checkpoint = torch.load(path, map_location="cpu")
            hidden_dim = self._infer_hidden_dim(checkpoint)
            state_dim  = int(checkpoint.get("state_dim",  self.env.observation_space_shape[0]))
            action_dim = int(checkpoint.get("action_dim", self.env.action_space_n))
            agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
            agent.load(path)
            self.loaded_agents[model_path] = agent
            return agent
        except Exception as exc:
            self.loaded_agent_errors.append(f"Erreur ({path.name}): {exc}")
            return None

    def _all_snake_controller_specs(self) -> list[dict]:
        return [
            {"snake_id": cfg.snake_id,
             "controller": cfg.controller_var.get().strip().lower(),
             "model_path": cfg.model_path_var.get().strip()}
            for cfg in self.snake_configs
        ]

    def _apply_snake_setup(self) -> None:
        self._reset_game(force_reload_models=True)
        self.notebook.select(self.tab_match)

    def _on_total_snakes_changed(self) -> None:
        self._build_snake_rows()
        self._reset_game(force_reload_models=False)

    def _on_speed_changed(self, _value: str) -> None:
        self.speed_label_var.set(f"{int(self.speed_var.get())}x")

    # ── Controls ─────────────────────────────────────────────────────────────
    def _bind_keys(self) -> None:
        self.root.bind("<Up>",    lambda e: self._queue_direction((0, -1)))
        self.root.bind("<Down>",  lambda e: self._queue_direction((0,  1)))
        self.root.bind("<Left>",  lambda e: self._queue_direction((-1, 0)))
        self.root.bind("<Right>", lambda e: self._queue_direction(( 1, 0)))
        self.root.bind("<w>",     lambda e: self._queue_direction((0, -1)))
        self.root.bind("<z>",     lambda e: self._queue_direction((0, -1)))
        self.root.bind("<s>",     lambda e: self._queue_direction((0,  1)))
        self.root.bind("<a>",     lambda e: self._queue_direction((-1, 0)))
        self.root.bind("<q>",     lambda e: self._queue_direction((-1, 0)))
        self.root.bind("<d>",     lambda e: self._queue_direction(( 1, 0)))
        self.root.bind("<space>", lambda e: self.toggle_running())
        self.root.bind_all("<MouseWheel>", self._on_mousewheel,       add=True)
        self.root.bind_all("<Button-4>",   self._on_mousewheel_linux, add=True)
        self.root.bind_all("<Button-5>",   self._on_mousewheel_linux, add=True)

    def _on_mousewheel(self, event: tk.Event) -> None:
        w = self.root.winfo_containing(event.x_root, event.y_root)
        if w and self._is_descendant(w, self.config_canvas):
            self.config_canvas.yview_scroll(int(-event.delta / 120), "units")
        elif w and self._is_descendant(w, self.info_text):
            self.info_text.yview_scroll(int(-event.delta / 120), "units")
        elif w and self._is_descendant(w, self.arena_canvas_frame):
            self.arena_canvas_frame.yview_scroll(int(-event.delta / 120), "units")

    def _on_mousewheel_linux(self, event: tk.Event) -> None:
        delta = -1 if event.num == 4 else 1
        w = self.root.winfo_containing(event.x_root, event.y_root)
        if w and self._is_descendant(w, self.config_canvas):
            self.config_canvas.yview_scroll(delta, "units")
        elif w and self._is_descendant(w, self.info_text):
            self.info_text.yview_scroll(delta, "units")
        elif w and self._is_descendant(w, self.arena_canvas_frame):
            self.arena_canvas_frame.yview_scroll(delta, "units")

    def _is_descendant(self, widget: tk.Widget, ancestor: tk.Widget) -> bool:
        current = widget
        while current is not None:
            if current == ancestor:
                return True
            parent_name = current.winfo_parent()
            if not parent_name:
                return False
            try:
                current = current.nametowidget(parent_name)
            except Exception:
                return False
        return False

    def _queue_direction(self, direction: tuple[int, int]) -> None:
        if self.current_human_snake_id is not None:
            self.pending_direction = direction

    def _absolute_direction_to_action(self, snake: Snake, desired_dir: tuple[int, int]) -> int:
        current_dir = snake.direction
        directions  = self.env.DIRECTIONS
        try:
            current_idx = directions.index(current_dir)
            desired_idx = directions.index(desired_dir)
        except ValueError:
            return 0
        diff = (desired_idx - current_idx) % 4
        if diff == 0: return 0
        if diff == 1: return 2
        if diff == 3: return 1
        return 0

    def _refresh_human_binding(self) -> None:
        human_ids = [s["snake_id"] for s in self._all_snake_controller_specs()
                     if s["controller"] == "human"]
        if not human_ids:
            self.current_human_snake_id = None
            self.current_human_warning  = None
            return
        self.current_human_snake_id = human_ids[0]
        self.current_human_warning  = None
        if len(human_ids) > 1:
            self.current_human_warning = f"Seul Snake {human_ids[0]} est piloté au clavier."

    # ── Game loop ────────────────────────────────────────────────────────────
    def _reset_game(self, force_reload_models: bool = False) -> None:
        self.pause()
        total_snakes = max(2, int(self.total_snakes_var.get()))
        self.env = MultiSnakeEnv(width=self.width, height=self.height,
                                 n_bots=total_snakes - 1,
                                 max_steps=self.max_steps, seed=self.seed)
        if force_reload_models:
            self.loaded_agents.clear()
        self.loaded_agent_errors    = []
        self.pending_direction      = None
        self.current_human_snake_id = None
        self.current_human_warning  = None
        _, self.last_info = self.env.reset(seed=self.seed)
        self.last_reward  = 0.0
        self.last_done    = False
        self._refresh_human_binding()
        self._draw()
        self._update_score_bars()
        self._update_side_panel()
        self._set_status("Prêt — Démarrer ou Espace", TEXT_SUB)

    def _set_status(self, msg: str, color: str = TEXT_SUB) -> None:
        self.status_var.set(msg)
        self._status_label.configure(fg=color)

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._set_status("En cours…", SUCCESS)
        self._loop()

    def pause(self) -> None:
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def toggle_running(self) -> None:
        if self.running:
            self.pause()
            self._set_status("En pause", WARNING)
        else:
            self.start()

    def step_once(self) -> None:
        self.pause()
        self._advance_game()

    def _controller_for_snake(self, snake_id: int) -> tuple[str, str]:
        for spec in self._all_snake_controller_specs():
            if spec["snake_id"] == snake_id:
                return spec["controller"], spec["model_path"]
        return ("heuristic", "")

    def _resolve_model_action(self, snake_id: int, model_path: str) -> int:
        agent = self._load_agent_for_path(model_path)
        if agent is None:
            snake = self.env.snakes[snake_id]
            return int(get_policy_callable("heuristic")(self.env, snake))
        state = self.env.get_observation_for_snake(snake_id)
        return int(agent.act(state, greedy=True))

    def _resolve_policy_action(self, snake_id: int, controller: str, model_path: str) -> int:
        if snake_id >= len(self.env.snakes):
            return 0
        snake = self.env.snakes[snake_id]
        if not snake.alive:
            return 0
        if controller == "human":
            if snake_id != self.current_human_snake_id:
                return int(get_policy_callable("safe_random")(self.env, snake))
            if self.pending_direction is None:
                return 0
            action = self._absolute_direction_to_action(snake, self.pending_direction)
            self.pending_direction = None
            return action
        if controller == "model":
            return self._resolve_model_action(snake_id, model_path)
        return int(get_policy_callable(controller)(self.env, snake))

    def _build_action_map(self) -> Dict[int, int]:
        self._refresh_human_binding()
        action_map: Dict[int, int] = {}
        for spec in self._all_snake_controller_specs():
            action_map[spec["snake_id"]] = self._resolve_policy_action(
                spec["snake_id"], spec["controller"], spec["model_path"])
        return action_map

    def _advance_game(self) -> None:
        if self.last_done:
            self._set_status("Partie terminée — Reset pour rejouer", TEXT_MUTED)
            return
        self.loaded_agent_errors = []
        action_map = self._build_action_map()
        _, reward, terminated, truncated, info = self.env.step(0, action_map=action_map)
        self.last_reward = reward
        self.last_info   = info
        self.last_done   = terminated or truncated

        steps   = self.env.steps
        total   = self.env.max_steps
        pct     = int(steps / total * 100)
        outcome = info.get("outcome", "ongoing")

        if self.last_done:
            label = OUTCOME_LABELS.get(outcome, outcome)
            msg   = f"  {label}  ({steps} steps)"
            color = SUCCESS if outcome.startswith("win") else (TEXT_MUTED if outcome.startswith("draw") else DANGER)
        else:
            msg   = f"  Step {steps}/{total}  ({pct}%)"
            color = TEXT_SUB
        if self.current_human_warning:
            msg += f"  •  {self.current_human_warning}"
        if self.loaded_agent_errors:
            msg += "  •  " + " ; ".join(self.loaded_agent_errors[:2])
        self._set_status(msg, color)

        self._draw()
        self._update_score_bars()
        self._update_side_panel()

    def _loop(self) -> None:
        self._advance_game()
        if self.running and not self.last_done:
            delay_ms = max(35, int(1000 / max(1, self.speed_var.get())))
            self.after_id = self.root.after(delay_ms, self._loop)
        else:
            self.running  = False
            self.after_id = None

    # ── Rendering ────────────────────────────────────────────────────────────
    def _draw(self) -> None:
        self.canvas.delete("all")
        canvas_w = self.width  * CELL_SIZE
        canvas_h = self.height * CELL_SIZE
        self.canvas.configure(width=canvas_w, height=canvas_h)

        # Fond
        self.canvas.create_rectangle(0, 0, canvas_w, canvas_h, fill=GRID_BG, outline="")

        # Grille : points subtils aux intersections
        for x in range(self.width + 1):
            for y in range(self.height + 1):
                px, py = x * CELL_SIZE, y * CELL_SIZE
                self.canvas.create_rectangle(px - 1, py - 1, px + 1, py + 1,
                                             fill=GRID_DOT, outline="")

        # Nourriture avec halo
        for (x, y), food_type in self.env.foods.items():
            color = FOOD_COLORS[food_type]
            cx = x * CELL_SIZE + CELL_SIZE // 2
            cy = y * CELL_SIZE + CELL_SIZE // 2
            r_halo  = CELL_SIZE // 2 - 3
            r_inner = CELL_SIZE // 2 - 6
            self.canvas.create_oval(cx - r_halo, cy - r_halo, cx + r_halo, cy + r_halo,
                                    fill=_dim_color(color, 0.25), outline="")
            self.canvas.create_oval(cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner,
                                    fill=color, outline="")

        # Snakes avec cellules arrondies et gradient
        for snake in self.env.snakes:
            base_color = SNAKE_COLORS[snake.snake_id % len(SNAKE_COLORS)]
            controller, _ = self._controller_for_snake(snake.snake_id)
            n = max(1, len(snake.body))

            for idx, (x, y) in enumerate(snake.body):
                x1 = x * CELL_SIZE + PADDING
                y1 = y * CELL_SIZE + PADDING
                x2 = (x + 1) * CELL_SIZE - PADDING
                y2 = (y + 1) * CELL_SIZE - PADDING

                if snake.alive:
                    factor = 1.0 - (idx / n) * 0.65
                    cell_color = _dim_color(base_color, factor)
                else:
                    cell_color = DEAD_COLOR

                _rounded_rect(self.canvas, x1, y1, x2, y2, r=5,
                               fill=cell_color, outline="")

                if idx == 0:
                    head_label = "H" if snake.snake_id == self.current_human_snake_id else str(snake.snake_id)
                    self.canvas.create_text(
                        x * CELL_SIZE + CELL_SIZE / 2,
                        y * CELL_SIZE + CELL_SIZE / 2,
                        text=head_label,
                        fill="#ffffff" if snake.alive else TEXT_MUTED,
                        font=("Arial", 9, "bold"),
                    )
                    if controller == "model" and snake.alive:
                        self.canvas.create_oval(x2 - 8, y1 + 2, x2 - 2, y1 + 8,
                                                fill=ACCENT, outline="")

        # Overlay fin
        if self.last_done:
            self._draw_end_overlay(canvas_w, canvas_h)

        self.canvas.update_idletasks()
        self.arena_canvas_frame.configure(scrollregion=self.arena_canvas_frame.bbox("all"))

    def _draw_end_overlay(self, canvas_w: int, canvas_h: int) -> None:
        rankings = self._compute_rankings()
        outcome  = self.last_info.get("outcome", "ongoing")

        oh = canvas_h // 3
        self.canvas.create_rectangle(0, oh, canvas_w, canvas_h - oh,
                                     fill="#000000", stipple="gray50", outline="")
        self.canvas.create_rectangle(0, oh, canvas_w, canvas_h - oh,
                                     fill=GRID_BG, stipple="gray25", outline="")

        cx = canvas_w // 2
        cy = canvas_h // 2

        if outcome.startswith("win"):
            title, title_color = "VICTOIRE !", WARNING
        elif outcome.startswith("draw"):
            title, title_color = "MATCH NUL", TEXT_SUB
        else:
            title, title_color = "DÉFAITE", DANGER

        self.canvas.create_rectangle(cx - 110, cy - 38, cx + 110, cy - 14,
                                     fill=_blend_color(GRID_BG, title_color, 0.18), outline="")
        self.canvas.create_text(cx, cy - 26, text=title, fill=title_color,
                                font=("Segoe UI", 18, "bold"), anchor="center")

        if rankings:
            winner = rankings[0]
            sid    = winner["snake_id"]
            ctrl, _ = self._controller_for_snake(sid)
            name   = "Human" if sid == self.current_human_snake_id else f"Snake {sid} [{ctrl}]"
            self.canvas.create_text(cx, cy + 4,
                                    text=f"{name}  •  score {winner['score']}",
                                    fill=TEXT_MAIN, font=("Segoe UI", 10), anchor="center")

        self.canvas.create_text(cx, cy + 26, text="⟳  Reset pour rejouer",
                                fill=TEXT_MUTED, font=("Segoe UI", 9), anchor="center")

    def _update_score_bars(self) -> None:
        self.score_canvas.delete("all")
        w = self.score_canvas.winfo_width()
        if w < 10:
            w = 300

        snakes = self.env.snakes
        if not snakes:
            return

        max_score = max((s.score for s in snakes), default=1) or 1
        bar_h     = 18
        pad_x     = 10
        label_w   = 76
        step_ratio = self.env.steps / max(1, self.env.max_steps)
        track_w    = w - 2 * pad_x

        # Ajuste dynamiquement la hauteur du canvas selon le nombre de snakes
        needed_h = 22 + len(snakes) * (bar_h + 8) + 4
        self.score_canvas.configure(height=max(60, needed_h))

        # Barre progression steps
        prog_y = 6
        self.score_canvas.create_rectangle(pad_x, prog_y, pad_x + track_w, prog_y + 6,
                                           fill=BUTTON_BG, outline="")
        fill_w = int(track_w * step_ratio)
        if fill_w > 0:
            self.score_canvas.create_rectangle(pad_x, prog_y, pad_x + fill_w, prog_y + 6,
                                               fill=ACCENT_DIM, outline="")
        self.score_canvas.create_text(w - pad_x, prog_y + 3,
                                      text=f"{self.env.steps}/{self.env.max_steps}",
                                      fill=TEXT_MUTED, font=("Consolas", 8), anchor="e")

        # Barres de score
        y_start = 22
        avail_w = w - 2 * pad_x - label_w - 8

        for i, snake in enumerate(snakes):
            y     = y_start + i * (bar_h + 8)
            color = SNAKE_COLORS[snake.snake_id % len(SNAKE_COLORS)]
            alive = snake.alive

            # Pastille
            self.score_canvas.create_oval(pad_x, y + 4, pad_x + 10, y + 14,
                                          fill=color if alive else DEAD_COLOR, outline="")
            ctrl, _ = self._controller_for_snake(snake.snake_id)
            short = "H" if snake.snake_id == self.current_human_snake_id else ctrl[:5]
            self.score_canvas.create_text(pad_x + 14, y + bar_h // 2,
                                          text=f"S{snake.snake_id}  {short}",
                                          fill=TEXT_MAIN if alive else TEXT_MUTED,
                                          font=("Consolas", 8), anchor="w")

            bx = pad_x + label_w
            _rounded_rect(self.score_canvas, bx, y, bx + avail_w, y + bar_h,
                          r=4, fill=BUTTON_BG, outline="")

            fill = int(avail_w * snake.score / max_score)
            if fill > 0:
                bar_color = _dim_color(color, 0.85) if alive else DEAD_COLOR
                _rounded_rect(self.score_canvas, bx, y, bx + fill, y + bar_h,
                              r=4, fill=bar_color, outline="")

            self.score_canvas.create_text(bx + avail_w + 6, y + bar_h // 2,
                                          text=str(snake.score),
                                          fill=TEXT_MAIN if alive else TEXT_MUTED,
                                          font=("Consolas", 8, "bold"), anchor="w")
            if not alive:
                self.score_canvas.create_text(
                    bx + (fill + 6 if fill > 8 else 6), y + bar_h // 2,
                    text="✕", fill=DANGER, font=("Consolas", 7), anchor="w")

    def _compute_rankings(self) -> list[dict]:
        rankings = self.last_info.get("rankings")
        if rankings:
            return rankings
        return sorted(
            [{"snake_id": s.snake_id, "is_player": s.is_player,
              "alive": s.alive, "score": s.score, "length": s.length}
             for s in self.env.snakes],
            key=lambda x: (x["score"], x["alive"], x["length"]),
            reverse=True,
        )

    def _update_side_panel(self) -> None:
        rankings = self._compute_rankings()
        outcome  = self.last_info.get("outcome", "ongoing")
        deaths   = self.last_info.get("deaths", {})

        self.info_text.configure(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)

        def w(text: str, tag: str = "") -> None:
            self.info_text.insert(tk.END, text, tag or ())

        def section(title: str) -> None:
            w(f"  {title}\n", "header")

        alive_count = sum(1 for s in self.env.snakes if s.alive)

        section("STATUT")
        w(f"  Step     ", "muted"); w(f"{self.env.steps}", "value"); w(f" / {self.env.max_steps}\n", "muted")
        outcome_label = OUTCOME_LABELS.get(outcome, outcome)
        outcome_tag   = "winner" if outcome.startswith("win") else ("alert" if outcome.startswith("loss") else "sub")
        w(f"  Résultat ", "muted"); w(f"{outcome_label}\n", outcome_tag)
        w(f"  Vivants  ", "muted"); w(f"{alive_count}", "alive" if alive_count > 0 else "dead"); w(f"/{len(self.env.snakes)}\n", "muted")
        w("\n")

        section("CLASSEMENT")
        medals = ["🥇", "🥈", "🥉"]
        for rank, item in enumerate(rankings, start=1):
            sid   = item["snake_id"]
            ctrl, model_path = self._controller_for_snake(sid)
            model_name = Path(model_path).name if model_path else ""
            alive = item.get("alive", False)
            medal = medals[rank - 1] if rank <= 3 else f"  {rank}."
            name_tag     = "winner" if (rank == 1 and self.last_done) else ("alive" if alive else "dead")
            ctrl_display = "human" if sid == self.current_human_snake_id else ctrl
            w(f"  {medal} ", "muted")
            w(f"Snake {sid}", name_tag)
            w(f"  [{ctrl_display}]", "muted")
            w(f"  {item['score']}pts", "sub")
            w(f"  len={item['length']}", "muted")
            w(f"  {'●' if alive else '✕'}\n", "alive" if alive else "dead")
            if ctrl == "model" and model_name:
                w(f"     ↳ {model_name}\n", "muted")
        w("\n")

        if deaths:
            section("DÉCÈS")
            for snake_id, reason in deaths.items():
                w(f"  Snake {snake_id}: {reason}\n", "dead")
            w("\n")

        section("NOURRITURE")
        n_normal = sum(1 for f in self.env.foods.values() if f == FoodType.NORMAL)
        n_bonus  = sum(1 for f in self.env.foods.values() if f == FoodType.BONUS)
        n_risky  = sum(1 for f in self.env.foods.values() if f == FoodType.RISKY)
        w(f"  ● {n_normal} normale  ● {n_bonus} bonus  ● {n_risky} risky\n", "sub")

        if self.loaded_agent_errors:
            w("\n"); section("ALERTES")
            for err in self.loaded_agent_errors:
                w(f"  ⚠ {err}\n", "alert")

        self.info_text.configure(state=tk.DISABLED)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Snake RL UI")
    parser.add_argument("--width",      type=int, default=15)
    parser.add_argument("--height",     type=int, default=15)
    parser.add_argument("--snakes",     type=int, default=3)
    parser.add_argument("--max-steps",  dest="max_steps", type=int, default=300)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--model-path", dest="model_path", type=str, default=None)
    args = parser.parse_args()

    ui = SnakeGameUI(
        width=args.width,
        height=args.height,
        total_snakes=max(2, args.snakes),
        max_steps=args.max_steps,
        seed=args.seed,
        model_path=args.model_path,
    )
    ui.run()


if __name__ == "__main__":
    main()
