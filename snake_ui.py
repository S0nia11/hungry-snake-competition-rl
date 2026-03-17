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


CELL_SIZE = 26
PADDING = 2

WINDOW_BG = "#0b1220"
SURFACE_BG = "#111827"
CARD_BG = "#0f172a"
BORDER = "#22314f"
GRID_BG = "#1b2435"
GRID_LINE = "#263348"
TEXT_MAIN = "#e5e7eb"
TEXT_SUB = "#94a3b8"
TEXT_MUTED = "#64748b"
ACCENT = "#60a5fa"
BUTTON_BG = "#1e293b"
BUTTON_ACTIVE = "#334155"
INPUT_BG = "#0b1324"

FOOD_COLORS = {
    FoodType.NORMAL: "#22c55e",
    FoodType.BONUS: "#facc15",
    FoodType.RISKY: "#ef4444",
}
SNAKE_COLORS = ["#3b82f6", "#f97316", "#a855f7", "#14b8a6", "#eab308", "#ec4899", "#06b6d4", "#84cc16"]
DEAD_COLOR = "#374151"
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
    """Assombrit une couleur hex par factor (0=noir, 1=original)."""
    factor = max(0.0, min(1.0, factor))
    r = int(int(hex_color[1:3], 16) * factor)
    g = int(int(hex_color[3:5], 16) * factor)
    b = int(int(hex_color[5:7], 16) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


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
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.seed = seed
        self.default_model_path = model_path or self._guess_default_model()

        self.root = tk.Tk()
        self.root.title("Multi-Snake RL")
        self.root.configure(bg=WINDOW_BG)
        self.root.geometry("1440x920")
        self.root.minsize(1100, 740)

        self.running = False
        self.after_id: Optional[str] = None
        self.pending_direction: Optional[tuple[int, int]] = None
        self.last_reward = 0.0
        self.last_done = False
        self.last_info: dict = {}
        self.loaded_agents: Dict[str, DQNAgent] = {}
        self.loaded_agent_errors: list[str] = []
        self.snake_configs: list[SnakeControllerConfig] = []
        self.current_human_snake_id: Optional[int] = None
        self.current_human_warning: Optional[str] = None

        self.total_snakes_var = tk.IntVar(value=max(2, total_snakes))
        self.speed_var = tk.IntVar(value=8)
        self.speed_label_var = tk.StringVar(value="8x")
        self.status_var = tk.StringVar(value="Prêt")

        self.env = MultiSnakeEnv(
            width=self.width,
            height=self.height,
            n_bots=max(1, self.total_snakes_var.get() - 1),
            max_steps=self.max_steps,
            seed=self.seed,
        )

        self._configure_styles()
        self._build_widgets()
        self._bind_keys()
        self.available_models = self._scan_model_files()
        self._build_snake_rows()
        self._reset_game(force_reload_models=True)

    # -------------------------- UI layout ---------------------------
    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background=WINDOW_BG)
        style.configure("Top.TFrame", background=SURFACE_BG)
        style.configure("Card.TFrame", background=CARD_BG)
        style.configure("Panel.TFrame", background=SURFACE_BG)

        style.configure("Title.TLabel", background=SURFACE_BG, foreground=TEXT_MAIN, font=("Segoe UI", 12, "bold"))
        style.configure("Inline.TLabel", background=SURFACE_BG, foreground=TEXT_MAIN, font=("Segoe UI", 10))
        style.configure("MutedInline.TLabel", background=SURFACE_BG, foreground=TEXT_SUB, font=("Segoe UI", 9))
        style.configure("CardTitle.TLabel", background=CARD_BG, foreground=TEXT_MAIN, font=("Segoe UI", 11, "bold"))
        style.configure("CardText.TLabel", background=CARD_BG, foreground=TEXT_SUB, font=("Segoe UI", 9))

        style.configure(
            "Dark.TButton",
            background=BUTTON_BG,
            foreground=TEXT_MAIN,
            bordercolor=BORDER,
            lightcolor=BUTTON_BG,
            darkcolor=BUTTON_BG,
            padding=(10, 6),
            relief="flat",
        )
        style.map("Dark.TButton", background=[("active", BUTTON_ACTIVE), ("pressed", BUTTON_ACTIVE)])

        style.configure(
            "Primary.TButton",
            background=ACCENT,
            foreground="#08111f",
            bordercolor=ACCENT,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
            padding=(10, 6),
            relief="flat",
            font=("Segoe UI", 9, "bold"),
        )
        style.map("Primary.TButton", background=[("active", "#93c5fd"), ("pressed", "#93c5fd")])

        style.configure("Dark.TCombobox", fieldbackground=INPUT_BG, background=INPUT_BG, foreground=TEXT_MAIN, arrowcolor=TEXT_MAIN)
        style.map("Dark.TCombobox", fieldbackground=[("readonly", INPUT_BG)], foreground=[("readonly", TEXT_MAIN)])
        style.configure("Dark.TEntry", fieldbackground=INPUT_BG, foreground=TEXT_MAIN, insertcolor=TEXT_MAIN, padding=5)
        style.configure("Dark.TSpinbox", fieldbackground=INPUT_BG, foreground=TEXT_MAIN, arrowsize=12, padding=3)
        style.configure("Dark.Horizontal.TScale", background=SURFACE_BG, troughcolor="#1e293b", lightcolor=ACCENT, darkcolor=ACCENT)
        style.configure("App.TNotebook", background=WINDOW_BG, borderwidth=0)
        style.configure("App.TNotebook.Tab", padding=(12, 6), background=SURFACE_BG, foreground=TEXT_SUB)
        style.map("App.TNotebook.Tab", background=[("selected", CARD_BG)], foreground=[("selected", TEXT_MAIN)])

    def _build_widgets(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top = ttk.Frame(self.root, style="Top.TFrame", padding=(10, 8))
        top.grid(row=0, column=0, sticky="ew")
        for col in range(20):
            top.columnconfigure(col, weight=0)
        top.columnconfigure(11, weight=1)

        ttk.Label(top, text="Multi-Snake RL", style="Title.TLabel").grid(row=0, column=0, padx=(0, 12), sticky="w")
        ttk.Label(top, text="Snakes", style="Inline.TLabel").grid(row=0, column=1, sticky="w")
        snakes_spin = ttk.Spinbox(top, from_=2, to=8, textvariable=self.total_snakes_var, width=4, command=self._on_total_snakes_changed, style="Dark.TSpinbox")
        snakes_spin.grid(row=0, column=2, padx=(6, 12), sticky="w")

        ttk.Label(top, text="Vitesse", style="Inline.TLabel").grid(row=0, column=3, sticky="w")
        speed = ttk.Scale(top, from_=1, to=14, variable=self.speed_var, orient=tk.HORIZONTAL, length=110, style="Dark.Horizontal.TScale", command=self._on_speed_changed)
        speed.grid(row=0, column=4, padx=(6, 6), sticky="w")
        ttk.Label(top, textvariable=self.speed_label_var, style="MutedInline.TLabel").grid(row=0, column=5, padx=(0, 12), sticky="w")

        ttk.Button(top, text="Appliquer", command=self._apply_snake_setup, style="Primary.TButton").grid(row=0, column=6, padx=3)
        ttk.Button(top, text="▶  Start", command=self.start, style="Dark.TButton").grid(row=0, column=7, padx=3)
        ttk.Button(top, text="⏸  Pause", command=self.pause, style="Dark.TButton").grid(row=0, column=8, padx=3)
        ttk.Button(top, text="↺  Reset", command=lambda: self._reset_game(force_reload_models=True), style="Dark.TButton").grid(row=0, column=9, padx=3)
        ttk.Button(top, text="›  Step", command=self.step_once, style="Dark.TButton").grid(row=0, column=10, padx=3)

        ttk.Button(top, text="Config Snakes", command=lambda: self.notebook.select(self.tab_snakes), style="Dark.TButton").grid(row=0, column=12, padx=(12, 3))
        ttk.Button(top, text="Arène", command=lambda: self.notebook.select(self.tab_match), style="Dark.TButton").grid(row=0, column=13, padx=3)

        self.notebook = ttk.Notebook(self.root, style="App.TNotebook")
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=10, pady=(8, 0))

        self.tab_match = ttk.Frame(self.notebook, style="App.TFrame")
        self.tab_snakes = ttk.Frame(self.notebook, style="App.TFrame")
        self.notebook.add(self.tab_match, text="  Match  ")
        self.notebook.add(self.tab_snakes, text="  Snakes & modèles  ")

        self._build_match_tab()
        self._build_snakes_tab()

        status = ttk.Frame(self.root, style="Top.TFrame", padding=(10, 5))
        status.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(status, textvariable=self.status_var, style="MutedInline.TLabel").pack(anchor="w")

    def _build_match_tab(self) -> None:
        self.tab_match.columnconfigure(0, weight=3)
        self.tab_match.columnconfigure(1, weight=1)
        self.tab_match.rowconfigure(0, weight=1)

        # --- Arène ---
        arena_card = ttk.Frame(self.tab_match, style="Card.TFrame", padding=10)
        arena_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=0)
        arena_card.columnconfigure(0, weight=1)
        arena_card.rowconfigure(1, weight=1)

        ttk.Label(arena_card, text="Arène", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))

        arena_wrap = tk.Frame(arena_card, bg=CARD_BG)
        arena_wrap.grid(row=1, column=0, sticky="nsew")
        arena_wrap.rowconfigure(0, weight=1)
        arena_wrap.columnconfigure(0, weight=1)

        self.arena_canvas_frame = tk.Canvas(arena_wrap, bg=CARD_BG, highlightthickness=0)
        self.arena_canvas_frame.grid(row=0, column=0, sticky="nsew")
        arena_vscroll = ttk.Scrollbar(arena_wrap, orient="vertical", command=self.arena_canvas_frame.yview)
        arena_hscroll = ttk.Scrollbar(arena_wrap, orient="horizontal", command=self.arena_canvas_frame.xview)
        arena_vscroll.grid(row=0, column=1, sticky="ns")
        arena_hscroll.grid(row=1, column=0, sticky="ew")
        self.arena_canvas_frame.configure(yscrollcommand=arena_vscroll.set, xscrollcommand=arena_hscroll.set)

        inner = tk.Frame(self.arena_canvas_frame, bg=CARD_BG)
        self.arena_canvas_frame.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: self.arena_canvas_frame.configure(scrollregion=self.arena_canvas_frame.bbox("all")))

        canvas_w = self.width * CELL_SIZE
        canvas_h = self.height * CELL_SIZE
        self.canvas = tk.Canvas(inner, width=canvas_w, height=canvas_h, bg=GRID_BG, highlightthickness=0)
        self.canvas.pack(anchor="nw")

        # Légende visuelle
        legend_frame = tk.Frame(arena_card, bg=CARD_BG)
        legend_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        for _, color, label in [
            (FoodType.NORMAL, FOOD_COLORS[FoodType.NORMAL], "normal"),
            (FoodType.BONUS,  FOOD_COLORS[FoodType.BONUS],  "bonus"),
            (FoodType.RISKY,  FOOD_COLORS[FoodType.RISKY],  "risky"),
        ]:
            dot = tk.Canvas(legend_frame, width=12, height=12, bg=CARD_BG, highlightthickness=0)
            dot.create_oval(1, 1, 11, 11, fill=color, outline="")
            dot.pack(side=tk.LEFT, padx=(8, 2))
            tk.Label(legend_frame, text=label, bg=CARD_BG, fg=TEXT_SUB, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(legend_frame, text="H = humain  •  ● = modèle RL", bg=CARD_BG, fg=TEXT_MUTED, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(16, 0))

        # --- Monitor ---
        side_card = ttk.Frame(self.tab_match, style="Card.TFrame", padding=10)
        side_card.grid(row=0, column=1, sticky="nsew")
        side_card.columnconfigure(0, weight=1)
        side_card.rowconfigure(2, weight=1)

        ttk.Label(side_card, text="Monitor", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))

        # Canvas barres de score
        self.score_canvas = tk.Canvas(side_card, bg=CARD_BG, height=120, highlightthickness=0)
        self.score_canvas.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        # Texte monitor
        right = tk.Frame(side_card, bg=CARD_BG)
        right.grid(row=2, column=0, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.info_text = tk.Text(
            right,
            wrap="word",
            bg=INPUT_BG,
            fg=TEXT_MAIN,
            insertbackground=TEXT_MAIN,
            relief=tk.FLAT,
            font=("Consolas", 9),
            padx=10,
            pady=8,
            cursor="arrow",
        )
        self.info_text.grid(row=0, column=0, sticky="nsew")
        info_scroll = ttk.Scrollbar(right, orient="vertical", command=self.info_text.yview)
        info_scroll.grid(row=0, column=1, sticky="ns")
        self.info_text.configure(yscrollcommand=info_scroll.set, state=tk.DISABLED)

        # Tags de couleur pour le texte monitor
        self.info_text.tag_configure("header",  foreground=ACCENT,   font=("Consolas", 9, "bold"))
        self.info_text.tag_configure("alive",   foreground="#4ade80")
        self.info_text.tag_configure("dead",    foreground=TEXT_MUTED)
        self.info_text.tag_configure("winner",  foreground="#fbbf24", font=("Consolas", 9, "bold"))
        self.info_text.tag_configure("sub",     foreground=TEXT_SUB)
        self.info_text.tag_configure("alert",   foreground="#f87171")
        self.info_text.tag_configure("muted",   foreground=TEXT_MUTED)

    def _build_snakes_tab(self) -> None:
        self.tab_snakes.columnconfigure(0, weight=1)
        self.tab_snakes.rowconfigure(1, weight=1)

        help_card = ttk.Frame(self.tab_snakes, style="Card.TFrame", padding=10)
        help_card.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(help_card, text="Configuration des snakes", style="CardTitle.TLabel").pack(anchor="w")
        tk.Label(
            help_card,
            text="Assignez un contrôleur (human / IA baseline / modèle RL) à chaque snake. Sélectionnez le fichier .pt dans le dropdown ou via Parcourir.",
            bg=CARD_BG, fg=TEXT_SUB, justify=tk.LEFT, anchor="w", font=("Segoe UI", 9), wraplength=1200,
        ).pack(fill=tk.X, pady=(4, 0))

        config_card = ttk.Frame(self.tab_snakes, style="Card.TFrame", padding=10)
        config_card.grid(row=1, column=0, sticky="nsew")
        config_card.rowconfigure(1, weight=1)
        config_card.columnconfigure(0, weight=1)
        config_card.columnconfigure(1, weight=0)
        ttk.Label(config_card, text="Contrôleur et modèle par snake", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Button(config_card, text="↺ Rafraîchir les modèles", style="Dark.TButton", command=self._refresh_model_list).grid(row=0, column=1, sticky="e", padx=(8, 0), pady=(0, 8))

        wrap = tk.Frame(config_card, bg=CARD_BG)
        wrap.grid(row=1, column=0, columnspan=2, sticky="nsew")
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)

        self.config_canvas = tk.Canvas(wrap, bg=CARD_BG, highlightthickness=0, height=420)
        self.config_canvas.grid(row=0, column=0, sticky="nsew")
        config_scroll = ttk.Scrollbar(wrap, orient="vertical", command=self.config_canvas.yview)
        config_scroll.grid(row=0, column=1, sticky="ns")
        self.config_canvas.configure(yscrollcommand=config_scroll.set)

        self.config_inner = tk.Frame(self.config_canvas, bg=CARD_BG)
        self.config_canvas.create_window((0, 0), window=self.config_inner, anchor="nw")
        self.config_inner.bind("<Configure>", lambda e: self.config_canvas.configure(scrollregion=self.config_canvas.bbox("all")))
        self.config_canvas.bind("<Configure>", lambda e: self.config_canvas.itemconfigure(1, width=e.width))

    # ----------------------- model / config helpers -----------------------
    def _build_snake_rows(self) -> None:
        # Sauvegarder les valeurs actuelles avant de tout reconstruire
        saved = {
            cfg.snake_id: (cfg.controller_var.get(), cfg.model_path_var.get())
            for cfg in self.snake_configs
        }

        for child in self.config_inner.winfo_children():
            child.destroy()
        self.snake_configs.clear()

        # En-têtes
        for text, col in [("Snake", 1), ("Contrôleur", 2), ("Modèle (.pt)", 3)]:
            lbl = tk.Label(self.config_inner, text=text, bg=CARD_BG, fg=TEXT_MAIN, font=("Segoe UI", 10, "bold"), anchor="w")
            lbl.grid(row=0, column=col, sticky="ew", padx=6, pady=(2, 8))
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

            # Pastille couleur
            swatch_color = SNAKE_COLORS[snake_id % len(SNAKE_COLORS)]
            swatch = tk.Canvas(self.config_inner, width=14, height=14, bg=CARD_BG, highlightthickness=0)
            swatch.create_rectangle(1, 1, 13, 13, fill=swatch_color, outline="", width=0)
            swatch.grid(row=snake_id + 1, column=0, padx=(6, 2), pady=4)

            label = tk.Label(self.config_inner, text=f"Snake {snake_id}", bg=CARD_BG, fg=TEXT_MAIN, font=("Segoe UI", 10), anchor="w")
            label.grid(row=snake_id + 1, column=1, sticky="ew", padx=(2, 6), pady=4)

            combo = ttk.Combobox(self.config_inner, textvariable=controller_var, values=CONTROLLERS, state="readonly", width=14, style="Dark.TCombobox")
            combo.grid(row=snake_id + 1, column=2, sticky="ew", padx=6, pady=4)

            model_combo = ttk.Combobox(self.config_inner, textvariable=model_path_var, values=self.available_models, state="normal", width=40, style="Dark.TCombobox")
            model_combo.grid(row=snake_id + 1, column=3, sticky="ew", padx=6, pady=4)

            button = ttk.Button(self.config_inner, text="Parcourir", style="Dark.TButton", command=lambda var=model_path_var: self._browse_model_for(var))
            button.grid(row=snake_id + 1, column=4, sticky="ew", padx=6, pady=4)

            self.snake_configs.append(SnakeControllerConfig(snake_id, controller_var, model_path_var, [label, combo, model_combo, button]))

    def _browse_model_for(self, target_var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")])
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
        candidates = [
            Path("outputs_v3/models/best_eval_model.pt"),
            Path("outputs_v2/models/best_eval_model.pt"),
            Path("outputs/models/best_reward_model.pt"),
            Path("/mnt/data/outputs_v3/models/best_eval_model.pt"),
            Path("/mnt/data/outputs_v2/models/best_eval_model.pt"),
            Path("/mnt/data/outputs/models/best_reward_model.pt"),
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        return ""

    def _infer_hidden_dim(self, checkpoint: dict) -> int:
        q_net = checkpoint.get("q_net_state_dict", {})
        for key in ("net.0.weight", "feature.0.weight"):
            if key in q_net:
                return int(q_net[key].shape[0])
        return 128

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
            state_dim = int(checkpoint.get("state_dim", self.env.observation_space_shape[0]))
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
            {"snake_id": cfg.snake_id, "controller": cfg.controller_var.get().strip().lower(), "model_path": cfg.model_path_var.get().strip()}
            for cfg in self.snake_configs
        ]

    def _apply_snake_setup(self) -> None:
        self._reset_game(force_reload_models=True)
        self.notebook.select(self.tab_match)

    def _on_total_snakes_changed(self) -> None:
        self._build_snake_rows()

    def _on_speed_changed(self, _value: str) -> None:
        self.speed_label_var.set(f"{int(self.speed_var.get())}x")

    # ------------------------- controls / input -------------------------
    def _bind_keys(self) -> None:
        self.root.bind("<Up>",    lambda e: self._queue_direction((0, -1)))
        self.root.bind("<Down>",  lambda e: self._queue_direction((0, 1)))
        self.root.bind("<Left>",  lambda e: self._queue_direction((-1, 0)))
        self.root.bind("<Right>", lambda e: self._queue_direction((1, 0)))
        self.root.bind("<w>", lambda e: self._queue_direction((0, -1)))
        self.root.bind("<z>", lambda e: self._queue_direction((0, -1)))
        self.root.bind("<s>", lambda e: self._queue_direction((0, 1)))
        self.root.bind("<a>", lambda e: self._queue_direction((-1, 0)))
        self.root.bind("<q>", lambda e: self._queue_direction((-1, 0)))
        self.root.bind("<d>", lambda e: self._queue_direction((1, 0)))
        self.root.bind("<space>", lambda e: self.toggle_running())
        self.root.bind_all("<MouseWheel>",  self._on_mousewheel, add=True)
        self.root.bind_all("<Button-4>",    self._on_mousewheel_linux, add=True)
        self.root.bind_all("<Button-5>",    self._on_mousewheel_linux, add=True)

    def _on_mousewheel(self, event: tk.Event) -> None:
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        if widget and self._is_descendant(widget, self.config_canvas):
            self.config_canvas.yview_scroll(int(-event.delta / 120), "units")
        elif widget and self._is_descendant(widget, self.info_text):
            self.info_text.yview_scroll(int(-event.delta / 120), "units")
        elif widget and self._is_descendant(widget, self.arena_canvas_frame):
            self.arena_canvas_frame.yview_scroll(int(-event.delta / 120), "units")

    def _on_mousewheel_linux(self, event: tk.Event) -> None:
        delta = -1 if event.num == 4 else 1
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        if widget and self._is_descendant(widget, self.config_canvas):
            self.config_canvas.yview_scroll(delta, "units")
        elif widget and self._is_descendant(widget, self.info_text):
            self.info_text.yview_scroll(delta, "units")
        elif widget and self._is_descendant(widget, self.arena_canvas_frame):
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
        directions = self.env.DIRECTIONS
        current_idx = directions.index(current_dir)
        desired_idx = directions.index(desired_dir)
        diff = (desired_idx - current_idx) % 4
        if diff == 0:
            return 0
        if diff == 1:
            return 2
        if diff == 3:
            return 1
        return 0

    def _refresh_human_binding(self) -> None:
        human_ids = [spec["snake_id"] for spec in self._all_snake_controller_specs() if spec["controller"] == "human"]
        if not human_ids:
            self.current_human_snake_id = None
            self.current_human_warning = None
            return
        self.current_human_snake_id = human_ids[0]
        self.current_human_warning = None
        if len(human_ids) > 1:
            self.current_human_warning = f"Seul Snake {human_ids[0]} est piloté au clavier."

    # ---------------------------- game loop ----------------------------
    def _reset_game(self, force_reload_models: bool = False) -> None:
        self.pause()
        total_snakes = max(2, int(self.total_snakes_var.get()))
        self.env = MultiSnakeEnv(width=self.width, height=self.height, n_bots=total_snakes - 1, max_steps=self.max_steps, seed=self.seed)
        if force_reload_models:
            self.loaded_agents.clear()
        self.loaded_agent_errors = []
        self.pending_direction = None
        self.current_human_snake_id = None
        self.current_human_warning = None
        _, self.last_info = self.env.reset(seed=self.seed)
        self.last_reward = 0.0
        self.last_done = False
        self._refresh_human_binding()
        self._draw()
        self._update_score_bars()
        self._update_side_panel()
        self.status_var.set("Prêt — appuie sur Start ou Espace")

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._loop()

    def pause(self) -> None:
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def toggle_running(self) -> None:
        if self.running:
            self.pause()
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
            action_map[spec["snake_id"]] = self._resolve_policy_action(spec["snake_id"], spec["controller"], spec["model_path"])
        return action_map

    def _advance_game(self) -> None:
        if self.last_done:
            self.status_var.set("Partie terminée — Reset pour rejouer")
            return

        self.loaded_agent_errors = []
        action_map = self._build_action_map()
        _, reward, terminated, truncated, info = self.env.step(0, action_map=action_map)
        self.last_reward = reward
        self.last_info = info
        self.last_done = terminated or truncated

        steps = self.env.steps
        total = self.env.max_steps
        pct = int(steps / total * 100)
        outcome = info.get("outcome", "ongoing")

        if self.last_done:
            label = OUTCOME_LABELS.get(outcome, outcome)
            msg = f"FIN  {label}  ({steps} steps)"
        else:
            msg = f"Step {steps}/{total}  ({pct}%)"
        if self.current_human_warning:
            msg += f"  •  {self.current_human_warning}"
        if self.loaded_agent_errors:
            msg += "  •  " + " ; ".join(self.loaded_agent_errors[:2])
        self.status_var.set(msg)

        self._draw()
        self._update_score_bars()
        self._update_side_panel()

    def _loop(self) -> None:
        self._advance_game()
        if self.running and not self.last_done:
            delay_ms = max(35, int(1000 / max(1, self.speed_var.get())))
            self.after_id = self.root.after(delay_ms, self._loop)
        else:
            self.running = False
            self.after_id = None

    # ---------------------------- rendering ----------------------------
    def _draw(self) -> None:
        self.canvas.delete("all")
        canvas_w = self.width * CELL_SIZE
        canvas_h = self.height * CELL_SIZE
        self.canvas.configure(width=canvas_w, height=canvas_h)
        self.canvas.create_rectangle(0, 0, canvas_w, canvas_h, fill=GRID_BG, outline=GRID_BG)

        # Grille
        for x in range(self.width + 1):
            self.canvas.create_line(x * CELL_SIZE, 0, x * CELL_SIZE, canvas_h, fill=GRID_LINE)
        for y in range(self.height + 1):
            self.canvas.create_line(0, y * CELL_SIZE, canvas_w, y * CELL_SIZE, fill=GRID_LINE)

        # Nourriture
        for (x, y), food_type in self.env.foods.items():
            color = FOOD_COLORS[food_type]
            cx = x * CELL_SIZE + CELL_SIZE // 2
            cy = y * CELL_SIZE + CELL_SIZE // 2
            r = CELL_SIZE // 2 - 5
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline="")

        # Snakes avec gradient tête → queue
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
                    # Gradient : tête = 100%, queue = 40%
                    factor = 1.0 - (idx / n) * 0.60
                    cell_color = _dim_color(base_color, factor)
                else:
                    cell_color = DEAD_COLOR

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=cell_color, outline="")

                if idx == 0:
                    # Label sur la tête
                    head_label = "H" if snake.snake_id == self.current_human_snake_id else str(snake.snake_id)
                    self.canvas.create_text(
                        x * CELL_SIZE + CELL_SIZE / 2,
                        y * CELL_SIZE + CELL_SIZE / 2,
                        text=head_label,
                        fill="#ffffff" if snake.alive else "#6b7280",
                        font=("Arial", 9, "bold"),
                    )
                    # Pastille blanche si modèle RL
                    if controller == "model" and snake.alive:
                        self.canvas.create_oval(x2 - 7, y1 + 1, x2 - 1, y1 + 7, fill="#ffffff", outline="")

        # Overlay fin de partie
        if self.last_done:
            self._draw_end_overlay(canvas_w, canvas_h)

        self.canvas.update_idletasks()
        self.arena_canvas_frame.configure(scrollregion=self.arena_canvas_frame.bbox("all"))

    def _draw_end_overlay(self, canvas_w: int, canvas_h: int) -> None:
        """Affiche un bandeau de résultat semi-transparent sur l'arène."""
        rankings = self._compute_rankings()
        outcome = self.last_info.get("outcome", "ongoing")

        # Fond semi-transparent (stipple gray25)
        self.canvas.create_rectangle(0, canvas_h // 3, canvas_w, canvas_h * 2 // 3,
                                     fill="#000000", stipple="gray50", outline="")
        self.canvas.create_rectangle(0, canvas_h // 3, canvas_w, canvas_h * 2 // 3,
                                     fill="#0b1220", stipple="gray25", outline="")

        cx, cy = canvas_w // 2, canvas_h // 2

        if outcome.startswith("win"):
            title = "VICTOIRE !"
            title_color = "#fbbf24"
        elif outcome.startswith("draw"):
            title = "MATCH NUL"
            title_color = "#94a3b8"
        else:
            title = "DÉFAITE"
            title_color = "#f87171"

        self.canvas.create_text(cx, cy - 18, text=title, fill=title_color,
                                font=("Segoe UI", 16, "bold"), anchor="center")

        if rankings:
            winner = rankings[0]
            sid = winner["snake_id"]
            ctrl, _ = self._controller_for_snake(sid)
            name = "Human" if sid == self.current_human_snake_id else f"Snake {sid} [{ctrl}]"
            score_txt = f"{name}  •  score {winner['score']}"
            self.canvas.create_text(cx, cy + 10, text=score_txt, fill="#e5e7eb",
                                    font=("Segoe UI", 10), anchor="center")

        self.canvas.create_text(cx, cy + 34, text="Appuie sur Reset pour rejouer",
                                fill=TEXT_MUTED, font=("Segoe UI", 8), anchor="center")

    def _update_score_bars(self) -> None:
        """Dessine des barres horizontales de score pour chaque snake."""
        self.score_canvas.delete("all")
        w = self.score_canvas.winfo_width()
        if w < 10:
            w = 280  # fallback avant premier rendu

        snakes = self.env.snakes
        if not snakes:
            return

        max_score = max((s.score for s in snakes), default=1)
        if max_score == 0:
            max_score = 1

        bar_h = 16
        padding_x = 6
        label_w = 68
        step_ratio = self.env.steps / max(1, self.env.max_steps)

        # Barre de progression des steps (fine, en haut)
        prog_y = 4
        self.score_canvas.create_rectangle(padding_x, prog_y, w - padding_x, prog_y + 5,
                                           fill="#1e293b", outline="")
        prog_w = int((w - 2 * padding_x) * step_ratio)
        if prog_w > 0:
            self.score_canvas.create_rectangle(padding_x, prog_y, padding_x + prog_w, prog_y + 5,
                                               fill=ACCENT, outline="")
        self.score_canvas.create_text(w - padding_x, prog_y + 2,
                                      text=f"{self.env.steps}/{self.env.max_steps}",
                                      fill=TEXT_MUTED, font=("Consolas", 8), anchor="e")

        # Barres de score par snake
        y_start = 18
        available_w = w - 2 * padding_x - label_w - 4

        for i, snake in enumerate(snakes):
            y = y_start + i * (bar_h + 6)
            color = SNAKE_COLORS[snake.snake_id % len(SNAKE_COLORS)]

            # Pastille + nom
            self.score_canvas.create_rectangle(padding_x, y + 2, padding_x + 10, y + 12,
                                               fill=color if snake.alive else DEAD_COLOR, outline="")
            ctrl, _ = self._controller_for_snake(snake.snake_id)
            label = f"S{snake.snake_id} {ctrl[:4]}"
            self.score_canvas.create_text(padding_x + 14, y + 7, text=label,
                                          fill=TEXT_MAIN if snake.alive else TEXT_MUTED,
                                          font=("Consolas", 8), anchor="w")

            # Fond de barre
            bx = padding_x + label_w
            self.score_canvas.create_rectangle(bx, y, bx + available_w, y + bar_h,
                                               fill="#1e293b", outline="")

            # Barre remplie
            bar_fill = int(available_w * snake.score / max_score)
            if bar_fill > 0:
                bar_color = color if snake.alive else DEAD_COLOR
                self.score_canvas.create_rectangle(bx, y, bx + bar_fill, y + bar_h,
                                                   fill=bar_color, outline="")

            # Score numérique
            self.score_canvas.create_text(bx + available_w + 4, y + bar_h // 2,
                                          text=str(snake.score),
                                          fill=TEXT_MAIN if snake.alive else TEXT_MUTED,
                                          font=("Consolas", 8), anchor="w")

    def _compute_rankings(self) -> list[dict]:
        rankings = self.last_info.get("rankings")
        if rankings:
            return rankings
        return sorted(
            [{"snake_id": s.snake_id, "is_player": s.is_player, "alive": s.alive, "score": s.score, "length": s.length} for s in self.env.snakes],
            key=lambda x: (x["score"], x["alive"], x["length"]),
            reverse=True,
        )

    def _update_side_panel(self) -> None:
        rankings = self._compute_rankings()
        outcome = self.last_info.get("outcome", "ongoing")
        deaths = self.last_info.get("deaths", {})

        self.info_text.configure(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)

        def w(text: str, tag: str = "") -> None:
            if tag:
                self.info_text.insert(tk.END, text, tag)
            else:
                self.info_text.insert(tk.END, text)

        # -- Statut --
        w("STATUT\n", "header")
        w(f"  Step    {self.env.steps}/{self.env.max_steps}\n", "sub")
        outcome_label = OUTCOME_LABELS.get(outcome, outcome)
        outcome_tag = "winner" if outcome.startswith("win") else ("alert" if outcome.startswith("loss") else "sub")
        w(f"  Outcome {outcome_label}\n", outcome_tag)
        alive_count = sum(1 for s in self.env.snakes if s.alive)
        w(f"  Vivants {alive_count}/{len(self.env.snakes)}\n", "sub")
        w("\n")

        # -- Classement --
        w("CLASSEMENT\n", "header")
        for rank, item in enumerate(rankings, start=1):
            sid = item["snake_id"]
            ctrl, model_path = self._controller_for_snake(sid)
            model_name = Path(model_path).name if model_path else "-"
            alive = item.get("alive", False)

            medal = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f" {rank}."
            name_tag = "winner" if rank == 1 and self.last_done else ("alive" if alive else "dead")
            ctrl_display = "human" if sid == self.current_human_snake_id else ctrl
            w(f"  {medal} Snake {sid}", name_tag)
            w(f"  [{ctrl_display}]", "muted")
            w(f"  score={item['score']}  len={item['length']}", "sub")
            w(f"  {'●' if alive else '✕'}\n", "alive" if alive else "dead")
            if ctrl == "model" and model_name != "-":
                w(f"     {model_name}\n", "muted")
        w("\n")

        # -- Décès récents --
        if deaths:
            w("DÉCÈS\n", "header")
            for snake_id, reason in deaths.items():
                w(f"  Snake {snake_id}: {reason}\n", "dead")
            w("\n")

        # -- Nourriture --
        w("NOURRITURES\n", "header")
        n_normal = sum(1 for f in self.env.foods.values() if f == FoodType.NORMAL)
        n_bonus  = sum(1 for f in self.env.foods.values() if f == FoodType.BONUS)
        n_risky  = sum(1 for f in self.env.foods.values() if f == FoodType.RISKY)
        w(f"  ● normale {n_normal}   ● bonus {n_bonus}   ● risky {n_risky}\n", "sub")
        w("\n")

        # -- Contrôles --
        w("CONTRÔLES\n", "header")
        w("  ↑/W/Z  ↓/S  ←/A/Q  →/D\n", "muted")
        w("  Espace = pause / reprise\n", "muted")

        # -- Alertes --
        if self.loaded_agent_errors:
            w("\nALERTES\n", "alert")
            for err in self.loaded_agent_errors:
                w(f"  {err}\n", "alert")

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
