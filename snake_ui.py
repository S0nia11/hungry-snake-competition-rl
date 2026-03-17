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


CELL_SIZE = 24
PADDING = 2

WINDOW_BG = "#0b1220"
SURFACE_BG = "#111827"
CARD_BG = "#0f172a"
BORDER = "#22314f"
GRID_BG = "#1b2435"
GRID_LINE = "#334155"
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
DEAD_COLOR = "#64748b"
CONTROLLERS = ("human",) + AVAILABLE_POLICIES


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
        self.root.title("Multi-Snake RL UI")
        self.root.configure(bg=WINDOW_BG)
        self.root.geometry("1400x900")
        self.root.minsize(1080, 720)

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
        ttk.Button(top, text="Start", command=self.start, style="Dark.TButton").grid(row=0, column=7, padx=3)
        ttk.Button(top, text="Pause", command=self.pause, style="Dark.TButton").grid(row=0, column=8, padx=3)
        ttk.Button(top, text="Reset", command=lambda: self._reset_game(force_reload_models=True), style="Dark.TButton").grid(row=0, column=9, padx=3)
        ttk.Button(top, text="Step", command=self.step_once, style="Dark.TButton").grid(row=0, column=10, padx=3)

        ttk.Button(top, text="Onglet Snakes", command=lambda: self.notebook.select(self.tab_snakes), style="Dark.TButton").grid(row=0, column=12, padx=(12, 3))
        ttk.Button(top, text="Onglet Match", command=lambda: self.notebook.select(self.tab_match), style="Dark.TButton").grid(row=0, column=13, padx=3)

        self.notebook = ttk.Notebook(self.root, style="App.TNotebook")
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=10, pady=(8, 0))

        self.tab_match = ttk.Frame(self.notebook, style="App.TFrame")
        self.tab_snakes = ttk.Frame(self.notebook, style="App.TFrame")
        self.notebook.add(self.tab_match, text="Match")
        self.notebook.add(self.tab_snakes, text="Snakes & modèles")

        self._build_match_tab()
        self._build_snakes_tab()

        status = ttk.Frame(self.root, style="Top.TFrame", padding=(10, 6))
        status.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(status, textvariable=self.status_var, style="Inline.TLabel").pack(anchor="w")

    def _build_match_tab(self) -> None:
        self.tab_match.columnconfigure(0, weight=2)
        self.tab_match.columnconfigure(1, weight=1)
        self.tab_match.rowconfigure(0, weight=1)

        arena_card = ttk.Frame(self.tab_match, style="Card.TFrame", padding=10)
        arena_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=0)
        arena_card.columnconfigure(0, weight=1)
        arena_card.rowconfigure(1, weight=1)

        ttk.Label(arena_card, text="Arène", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))

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

        legend = tk.Label(
            arena_card,
            text="Vert = normal   •   Jaune = bonus   •   Rouge = risky   •   H = humain   •   pastille blanche = modèle",
            bg=CARD_BG,
            fg=TEXT_SUB,
            anchor="w",
            justify=tk.LEFT,
            font=("Segoe UI", 9),
        )
        legend.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        side_card = ttk.Frame(self.tab_match, style="Card.TFrame", padding=10)
        side_card.grid(row=0, column=1, sticky="nsew")
        side_card.columnconfigure(0, weight=1)
        side_card.rowconfigure(1, weight=1)

        ttk.Label(side_card, text="Monitor", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        right = tk.Frame(side_card, bg=CARD_BG)
        right.grid(row=1, column=0, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.info_text = tk.Text(
            right,
            wrap="word",
            bg=INPUT_BG,
            fg=TEXT_MAIN,
            insertbackground=TEXT_MAIN,
            relief=tk.FLAT,
            font=("Consolas", 10),
            padx=10,
            pady=10,
        )
        self.info_text.grid(row=0, column=0, sticky="nsew")
        info_scroll = ttk.Scrollbar(right, orient="vertical", command=self.info_text.yview)
        info_scroll.grid(row=0, column=1, sticky="ns")
        self.info_text.configure(yscrollcommand=info_scroll.set, state=tk.DISABLED)

    def _build_snakes_tab(self) -> None:
        self.tab_snakes.columnconfigure(0, weight=1)
        self.tab_snakes.rowconfigure(1, weight=1)

        help_card = ttk.Frame(self.tab_snakes, style="Card.TFrame", padding=10)
        help_card.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(help_card, text="Configuration compacte", style="CardTitle.TLabel").pack(anchor="w")
        help_text = (
            "Chaque snake peut avoir un contrôleur différent. Le panneau ci-dessous a son propre scroll, "
            "donc il ne pousse plus l'arène vers le bas. Une seule ligne 'human' est pilotée au clavier."
        )
        tk.Label(help_card, text=help_text, bg=CARD_BG, fg=TEXT_SUB, justify=tk.LEFT, anchor="w", font=("Segoe UI", 9), wraplength=1100).pack(fill=tk.X, pady=(4, 0))

        config_card = ttk.Frame(self.tab_snakes, style="Card.TFrame", padding=10)
        config_card.grid(row=1, column=0, sticky="nsew")
        config_card.rowconfigure(1, weight=1)
        config_card.columnconfigure(0, weight=1)
        ttk.Label(config_card, text="Contrôleur et modèle par snake", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))

        wrap = tk.Frame(config_card, bg=CARD_BG)
        wrap.grid(row=1, column=0, sticky="nsew")
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
        for child in self.config_inner.winfo_children():
            child.destroy()
        self.snake_configs.clear()

        headers = [("Snake", 0), ("Contrôleur", 1), ("Modèle (.pt)", 2)]
        for text, col in headers:
            lbl = tk.Label(self.config_inner, text=text, bg=CARD_BG, fg=TEXT_MAIN, font=("Segoe UI", 10, "bold"), anchor="w")
            lbl.grid(row=0, column=col, sticky="ew", padx=6, pady=(2, 8))
        self.config_inner.grid_columnconfigure(2, weight=1)

        total = max(2, int(self.total_snakes_var.get()))
        for snake_id in range(total):
            controller_var = tk.StringVar(value="human" if snake_id == 0 else "heuristic")
            model_path_var = tk.StringVar(value=self.default_model_path if snake_id == 0 else "")

            label = tk.Label(self.config_inner, text=f"Snake {snake_id}", bg=CARD_BG, fg=TEXT_MAIN, font=("Segoe UI", 10), anchor="w")
            label.grid(row=snake_id + 1, column=0, sticky="ew", padx=6, pady=4)

            combo = ttk.Combobox(self.config_inner, textvariable=controller_var, values=CONTROLLERS, state="readonly", width=14, style="Dark.TCombobox")
            combo.grid(row=snake_id + 1, column=1, sticky="ew", padx=6, pady=4)

            entry = ttk.Entry(self.config_inner, textvariable=model_path_var, style="Dark.TEntry")
            entry.grid(row=snake_id + 1, column=2, sticky="ew", padx=6, pady=4)

            button = ttk.Button(self.config_inner, text="Parcourir", style="Dark.TButton", command=lambda var=model_path_var: self._browse_model_for(var))
            button.grid(row=snake_id + 1, column=3, sticky="ew", padx=6, pady=4)

            self.snake_configs.append(SnakeControllerConfig(snake_id, controller_var, model_path_var, [label, combo, entry, button]))

    def _browse_model_for(self, target_var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")])
        if path:
            target_var.set(path)

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
            self.loaded_agent_errors.append(f"Modèle introuvable: {model_path}")
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
            self.loaded_agent_errors.append(f"Chargement échoué ({model_path}): {exc}")
            return None

    def _all_snake_controller_specs(self) -> list[dict]:
        return [
            {"snake_id": cfg.snake_id, "controller": cfg.controller_var.get().strip().lower(), "model_path": cfg.model_path_var.get().strip()}
            for cfg in self.snake_configs
        ]

    def _apply_snake_setup(self) -> None:
        self._build_snake_rows()
        self._reset_game(force_reload_models=True)
        self.notebook.select(self.tab_match)

    def _on_total_snakes_changed(self) -> None:
        self._build_snake_rows()

    def _on_speed_changed(self, _value: str) -> None:
        self.speed_label_var.set(f"{int(self.speed_var.get())}x")

    # ------------------------- controls / input -------------------------
    def _bind_keys(self) -> None:
        self.root.bind("<Up>", lambda e: self._queue_direction((0, -1)))
        self.root.bind("<Down>", lambda e: self._queue_direction((0, 1)))
        self.root.bind("<Left>", lambda e: self._queue_direction((-1, 0)))
        self.root.bind("<Right>", lambda e: self._queue_direction((1, 0)))
        self.root.bind("<w>", lambda e: self._queue_direction((0, -1)))
        self.root.bind("<z>", lambda e: self._queue_direction((0, -1)))
        self.root.bind("<s>", lambda e: self._queue_direction((0, 1)))
        self.root.bind("<a>", lambda e: self._queue_direction((-1, 0)))
        self.root.bind("<q>", lambda e: self._queue_direction((-1, 0)))
        self.root.bind("<d>", lambda e: self._queue_direction((1, 0)))
        self.root.bind("<space>", lambda e: self.toggle_running())
        self.root.bind_all("<MouseWheel>", self._on_mousewheel, add=True)
        self.root.bind_all("<Button-4>", self._on_mousewheel_linux, add=True)
        self.root.bind_all("<Button-5>", self._on_mousewheel_linux, add=True)

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
            self.current_human_warning = f"Une seule ligne 'human' peut être pilotée. Snake {human_ids[0]} est contrôlé au clavier."

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
        self._update_side_panel()
        status = "Nouvelle partie prête. L'onglet Snakes reste scrollable et n'écrase plus l'arène."
        if self.current_human_warning:
            status += f" {self.current_human_warning}"
        self.status_var.set(status)

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
            self.status_var.set("Partie terminée. Reset pour recommencer.")
            return

        self.loaded_agent_errors = []
        action_map = self._build_action_map()
        _, reward, terminated, truncated, info = self.env.step(0, action_map=action_map)
        self.last_reward = reward
        self.last_info = info
        self.last_done = terminated or truncated

        if self.last_done:
            msg = f"Fin de partie | outcome={info.get('outcome', 'ongoing')} | reward={reward:.2f}"
        else:
            msg = f"step={self.env.steps} | snakes={len(self.env.snakes)} | outcome={info.get('outcome', 'ongoing')}"
        if self.current_human_warning:
            msg += f" | {self.current_human_warning}"
        if self.loaded_agent_errors:
            msg += " | " + " ; ".join(self.loaded_agent_errors[:2])
        self.status_var.set(msg)

        self._draw()
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

        for x in range(self.width + 1):
            px = x * CELL_SIZE
            self.canvas.create_line(px, 0, px, canvas_h, fill=GRID_LINE)
        for y in range(self.height + 1):
            py = y * CELL_SIZE
            self.canvas.create_line(0, py, canvas_w, py, fill=GRID_LINE)

        for (x, y), food_type in self.env.foods.items():
            color = FOOD_COLORS[food_type]
            x1 = x * CELL_SIZE + 6
            y1 = y * CELL_SIZE + 6
            x2 = (x + 1) * CELL_SIZE - 6
            y2 = (y + 1) * CELL_SIZE - 6
            self.canvas.create_oval(x1, y1, x2, y2, fill=color, outline="")

        for snake in self.env.snakes:
            base_color = SNAKE_COLORS[snake.snake_id % len(SNAKE_COLORS)]
            color = base_color if snake.alive else DEAD_COLOR
            controller, _ = self._controller_for_snake(snake.snake_id)
            head_label = "H" if snake.snake_id == self.current_human_snake_id else str(snake.snake_id)
            for idx, (x, y) in enumerate(snake.body):
                x1 = x * CELL_SIZE + PADDING
                y1 = y * CELL_SIZE + PADDING
                x2 = (x + 1) * CELL_SIZE - PADDING
                y2 = (y + 1) * CELL_SIZE - PADDING
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                if idx == 0:
                    self.canvas.create_text(x * CELL_SIZE + CELL_SIZE / 2, y * CELL_SIZE + CELL_SIZE / 2, text=head_label, fill="#ffffff", font=("Arial", 10, "bold"))
                    if controller == "model":
                        self.canvas.create_oval(x1 + 3, y1 + 3, x1 + 8, y1 + 8, fill="#ffffff", outline="")
        self.canvas.update_idletasks()
        self.arena_canvas_frame.configure(scrollregion=self.arena_canvas_frame.bbox("all"))

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

        lines = [
            "STATUT",
            f"Step         : {self.env.steps}/{self.env.max_steps}",
            f"Reward P0    : {self.last_reward:.2f}",
            f"Outcome      : {outcome}",
            f"Snakes       : {len(self.env.snakes)}",
            f"Alive        : {sum(1 for s in self.env.snakes if s.alive)}/{len(self.env.snakes)}",
            f"Human        : {'aucun' if self.current_human_snake_id is None else f'Snake {self.current_human_snake_id}'}",
            "",
            "CONTRÔLE UTILISATEUR",
            "↑/W/Z = haut   ↓/S = bas",
            "←/A/Q = gauche  →/D = droite",
            "Demi-tour interdit : le snake continue tout droit",
            "Espace : pause / reprise",
            "",
            "CONTRÔLEURS",
        ]

        for spec in self._all_snake_controller_specs():
            model_name = Path(spec["model_path"]).name if spec["model_path"] else "-"
            lines.append(f"Snake {spec['snake_id']}: {spec['controller']:<10} | {model_name}")

        lines.extend(["", "CLASSEMENT"])
        for rank, item in enumerate(rankings, start=1):
            sid = item["snake_id"]
            ctrl, _ = self._controller_for_snake(sid)
            label = f"Snake {sid}"
            if sid == self.current_human_snake_id:
                label += " [human]"
            elif ctrl == "model":
                label += " [model]"
            alive_txt = "alive" if item.get("alive") else "dead"
            lines.append(f"{rank:>2}. {label:<18} | score={item['score']:<3} | len={item['length']:<2} | {alive_txt}")

        if deaths:
            lines.extend(["", "DERNIERS DÉCÈS"])
            for snake_id, reason in deaths.items():
                lines.append(f"Snake {snake_id}: {reason}")

        lines.extend(["", "NOURRITURES", f"normale : {sum(1 for f in self.env.foods.values() if f == FoodType.NORMAL)}", f"bonus   : {sum(1 for f in self.env.foods.values() if f == FoodType.BONUS)}", f"risky   : {sum(1 for f in self.env.foods.values() if f == FoodType.RISKY)}"])

        if self.loaded_agent_errors:
            lines.extend(["", "ALERTES MODELS"])
            lines.extend(self.loaded_agent_errors)

        self.info_text.configure(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, "\n".join(lines))
        self.info_text.configure(state=tk.DISABLED)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="UI multi-snake RL compacte avec scroll.")
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--snakes", type=int, default=3, help="Nombre total de snakes")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    ui = SnakeGameUI(width=args.width, height=args.height, total_snakes=max(2, args.snakes), max_steps=args.max_steps, seed=args.seed, model_path=args.model_path)
    ui.run()


if __name__ == "__main__":
    main()
