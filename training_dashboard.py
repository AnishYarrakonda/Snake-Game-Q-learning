# Tkinter training dashboard with live board view.
from __future__ import annotations

from collections import deque
from dataclasses import replace
import json
import os
import queue
import threading
from typing import Callable

import numpy as np
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox

try:
    from .agent import SnakeDQNAgent
    from .game_logic import SnakeGame
    from .utils import (
        APPLE_CHOICES,
        BOARD_SIZES,
        MAX_HIDDEN_LAYERS,
        MIN_HIDDEN_LAYERS,
        MODELS_DIR,
        STATE_ENCODING_BOARD,
        STATE_ENCODING_INTEGER,
        SUPPORTED_STATE_ENCODINGS,
        TrainConfig,
        default_model_path,
        encode_state,
        make_game,
        parse_hidden_layer_widths,
        run_episode,
    )
except ImportError:
    from agent import SnakeDQNAgent
    from game_logic import SnakeGame
    from utils import (
        APPLE_CHOICES,
        BOARD_SIZES,
        MAX_HIDDEN_LAYERS,
        MIN_HIDDEN_LAYERS,
        MODELS_DIR,
        STATE_ENCODING_BOARD,
        STATE_ENCODING_INTEGER,
        SUPPORTED_STATE_ENCODINGS,
        TrainConfig,
        default_model_path,
        encode_state,
        make_game,
        parse_hidden_layer_widths,
        run_episode,
    )


class TrainingDashboard:
    """Dashboard to train/watch Snake DQN without blocking Tkinter."""
    BG_MAIN = "#0b1220"
    PANEL_BG = "#111a2a"
    PANEL_ALT = "#182235"
    TEXT = "#e6eef7"
    TEXT_MUTED = "#9db0c7"
    ACCENT = "#4fc3f7"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Snake DQN Training Dashboard")
        self.root.geometry("1760x1060")
        self.root.configure(bg=self.BG_MAIN)

        self.msg_queue: queue.Queue[dict] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.cfg_lock = threading.Lock()
        self.runtime_cfg: TrainConfig | None = None
        self.active_training_immutable: tuple[int, int, int] | None = None  # board, apples, episodes
        self.pending_visual_changes = False
        self.pending_epsilon_changes = False

        self.cfg = TrainConfig()
        self.agent = SnakeDQNAgent(self.cfg)

        self.scores: list[float] = []

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(80, self._poll_queue)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=0, minsize=560)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = tk.Frame(self.root, bg=self.BG_MAIN)
        left.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        left.rowconfigure(2, weight=1)
        left.columnconfigure(0, weight=1)

        right = tk.Frame(self.root, bg=self.BG_MAIN)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        controls = tk.LabelFrame(
            left,
            text="Training Controls",
            bg=self.PANEL_BG,
            fg=self.TEXT,
            font=("Helvetica", 12, "bold"),
            padx=12,
            pady=10,
            bd=1,
            relief="groove",
        )
        controls.grid(row=0, column=0, sticky="ew")

        self.board_var = tk.StringVar(value=str(self.cfg.board_size))
        self.apple_var = tk.StringVar(value=str(self.cfg.apples))
        self.hidden_layer_count_var = tk.StringVar(value=str(len(self.cfg.hidden_layers)))
        self.neurons_var = tk.StringVar(value=self._format_hidden_layers_for_ui(self.cfg.hidden_layers))
        self.epsilon_start_var = tk.StringVar(value=f"{self.cfg.epsilon_start:.2f}")
        self.epsilon_min_var = tk.StringVar(value=f"{self.cfg.epsilon_min:.4f}")
        self.epsilon_decay_rate_var = tk.StringVar(value=f"{self.cfg.epsilon_decay_rate:.2f}")
        self.print_every_var = tk.StringVar(value="25")
        self.anim_delay_var = tk.DoubleVar(value=0.0)
        self.snake_head_color_var = tk.StringVar(value="#45d483")
        self.snake_body_color_var = tk.StringVar(value="#1fb86b")
        self.apple_color_var = tk.StringVar(value="#ff5c74")
        self.grid_color_var = tk.StringVar(value="#2a3340")
        self.board_bg_color_var = tk.StringVar(value="#1c2229")
        self.border_color_var = tk.StringVar(value="#7f8b99")
        self.epsilon_start_var.trace_add("write", lambda *_: self._mark_epsilon_pending())
        self.epsilon_min_var.trace_add("write", lambda *_: self._mark_epsilon_pending())
        self.epsilon_decay_rate_var.trace_add("write", lambda *_: self._mark_epsilon_pending())
        self.snake_head_color_var.trace_add("write", lambda *_: self._mark_visual_pending())
        self.snake_body_color_var.trace_add("write", lambda *_: self._mark_visual_pending())
        self.apple_color_var.trace_add("write", lambda *_: self._mark_visual_pending())
        self.grid_color_var.trace_add("write", lambda *_: self._mark_visual_pending())
        self.board_bg_color_var.trace_add("write", lambda *_: self._mark_visual_pending())
        self.border_color_var.trace_add("write", lambda *_: self._mark_visual_pending())

        columns = tk.Frame(controls, bg=self.PANEL_BG)
        columns.pack(fill="x")
        left_col = tk.Frame(columns, bg=self.PANEL_BG)
        right_col = tk.Frame(columns, bg=self.PANEL_BG)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right_col.pack(side="left", fill="both", expand=True, padx=(8, 0))

        self._add_dropdown(left_col, "Board", self.board_var, [str(v) for v in BOARD_SIZES])
        self._add_dropdown(left_col, "Apples", self.apple_var, [str(v) for v in APPLE_CHOICES])
        self._add_entry(left_col, "Hidden layers", self.hidden_layer_count_var)
        self._add_entry(left_col, "Neurons/layer", self.neurons_var)
        self._add_entry(left_col, "Eps start", self.epsilon_start_var)
        self._add_entry(left_col, "Eps min", self.epsilon_min_var)
        self._add_entry(left_col, "Eps decay rate", self.epsilon_decay_rate_var)
        epsilon_btn_row = tk.Frame(left_col, bg=self.PANEL_BG)
        epsilon_btn_row.pack(fill="x", pady=(2, 4))
        self._make_btn(epsilon_btn_row, "Apply Epsilon", self._apply_epsilon_settings, w=12).pack(side="left", padx=2)
        self._add_entry(left_col, "Chunk episodes", self.print_every_var)
        self._add_info(right_col, "Policy", "Backend phase scheduler")
        self._add_info(right_col, "Gamma/N-step", "Automatic curriculum")
        self._add_info(right_col, "Rewards/LR/Epsilon", "Automatic curriculum")
        self._add_info(right_col, "Target updates", "Hard -> Soft auto switch")
        self._add_slider(right_col, "Anim delay (ms)", self.anim_delay_var, 0, 150, 5)

        visuals = tk.LabelFrame(
            controls,
            text="Visual Theme",
            bg=self.PANEL_ALT,
            fg=self.TEXT,
            font=("Helvetica", 10, "bold"),
            padx=10,
            pady=8,
            bd=1,
            relief="groove",
        )
        visuals.pack(fill="x", pady=(8, 0))
        self._add_color_row(visuals, "Snake head", self.snake_head_color_var)
        self._add_color_row(visuals, "Snake body", self.snake_body_color_var)
        self._add_color_row(visuals, "Apple", self.apple_color_var)
        self._add_color_row(visuals, "Grid", self.grid_color_var)
        self._add_color_row(visuals, "Board bg", self.board_bg_color_var)
        self._add_color_row(visuals, "Border", self.border_color_var)
        visual_btn_row = tk.Frame(visuals, bg=self.PANEL_ALT)
        visual_btn_row.pack(fill="x", pady=(4, 0))
        self._make_btn(visual_btn_row, "Apply Colors", self._apply_colors, w=12).pack(side="left", padx=2)
        self._make_btn(visual_btn_row, "Reset Colors", self._reset_colors, w=12).pack(side="left", padx=2)

        btn_row = tk.Frame(controls, bg=self.PANEL_BG)
        btn_row.pack(fill="x", pady=(8, 0))

        self._make_btn(btn_row, "Train", self.start_training, w=10).pack(side="left", padx=2)
        self._make_btn(btn_row, "Watch", self.start_watch, w=10).pack(side="left", padx=2)
        self._make_btn(btn_row, "Stop", self.stop_worker, w=10).pack(side="left", padx=2)
        self._make_btn(btn_row, "Load", self.load_model, w=10).pack(side="left", padx=2)
        self._make_btn(btn_row, "Save", self.save_model, w=10).pack(side="left", padx=2)
        cfg_btn_row = tk.Frame(controls, bg=self.PANEL_BG)
        cfg_btn_row.pack(fill="x", pady=(6, 0))
        self._make_btn(cfg_btn_row, "Save Config", self.save_config, w=12).pack(side="left", padx=2)
        self._make_btn(cfg_btn_row, "Load Config", self.load_config, w=12).pack(side="left", padx=2)
        help_label = tk.Label(
            controls,
            text="Shortcuts: Ctrl+W Watch, Ctrl+S Stop, Ctrl+L Load, Ctrl+Shift+S Save",
            fg=self.TEXT_MUTED,
            bg=self.PANEL_BG,
            font=("Helvetica", 8),
            wraplength=500,
            justify="left",
        )
        help_label.pack(anchor="w", pady=(6, 0))

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(
            left,
            textvariable=self.status_var,
            fg=self.TEXT,
            bg=self.BG_MAIN,
            font=("Helvetica", 12, "bold"),
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", pady=(10, 8), padx=2)

        board_frame = tk.LabelFrame(
            right,
            text="Live Snake Board",
            bg=self.PANEL_BG,
            fg=self.TEXT,
            font=("Helvetica", 11, "bold"),
            padx=8,
            pady=8,
            bd=1,
            relief="groove",
        )
        board_frame.grid(row=0, column=0, sticky="nsew")
        board_frame.rowconfigure(0, weight=1)
        board_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            board_frame,
            bg=self.board_bg_color_var.get(),
            width=1080,
            height=920,
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        state_frame = tk.LabelFrame(
            left,
            text="Current State Features (Last Step)",
            bg=self.PANEL_BG,
            fg=self.TEXT,
            font=("Helvetica", 10, "bold"),
            padx=8,
            pady=6,
            bd=1,
            relief="groove",
        )
        state_frame.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        state_frame.rowconfigure(0, weight=1)
        state_frame.columnconfigure(0, weight=1)
        self.state_text = tk.Text(
            state_frame,
            height=12,
            width=62,
            bg=self.PANEL_ALT,
            fg=self.TEXT,
            font=("Courier", 9),
            wrap="none",
            bd=0,
            relief="flat",
        )
        self.state_text.pack(fill="both", expand=True)
        self.state_text.insert("1.0", "State features will appear here during training/watch...")
        self.state_text.config(state="disabled")

        self.root.bind("<Control-t>", lambda _e: self.start_training())
        self.root.bind("<Control-w>", lambda _e: self.start_watch())
        self.root.bind("<Control-s>", lambda _e: self.stop_worker())
        self.root.bind("<Control-l>", lambda _e: self.load_model())
        self.root.bind("<Control-Shift-S>", lambda _e: self.save_model())
        self.root.bind("<F5>", lambda _e: self.start_training())
        self._apply_visual_settings()

    def _add_entry(self, parent: tk.Widget, label: str, var: tk.StringVar) -> None:
        row = tk.Frame(parent, bg=self.PANEL_BG)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, bg=self.PANEL_BG, fg=self.TEXT_MUTED, width=12, anchor="w").pack(side="left")
        tk.Entry(
            row,
            textvariable=var,
            width=12,
            justify="center",
            relief="flat",
            bd=0,
            bg="#e8eef5",
            fg="#112033",
            insertbackground="#112033",
        ).pack(side="left", fill="x", expand=True)

    def _add_dropdown(self, parent: tk.Widget, label: str, var: tk.StringVar, options: list[str]) -> None:
        row = tk.Frame(parent, bg=self.PANEL_BG)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, bg=self.PANEL_BG, fg=self.TEXT_MUTED, width=12, anchor="w").pack(side="left")
        dropdown = tk.OptionMenu(row, var, *options)
        dropdown.config(
            bg="#e8eef5",
            fg="#112033",
            activebackground="#d7e4f1",
            activeforeground="#112033",
            relief="flat",
            bd=0,
            highlightthickness=0,
            width=9,
        )
        dropdown["menu"].config(bg="#e8eef5", fg="#112033", activebackground="#d7e4f1", activeforeground="#112033")
        dropdown.pack(side="left", fill="x", expand=True)

    def _add_info(self, parent: tk.Widget, label: str, value: str) -> None:
        row = tk.Frame(parent, bg=self.PANEL_BG)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, bg=self.PANEL_BG, fg=self.TEXT_MUTED, width=12, anchor="w").pack(side="left")
        tk.Label(row, text=value, bg=self.PANEL_BG, fg=self.TEXT, anchor="w").pack(side="left")

    def _add_slider(
        self,
        parent: tk.Widget,
        label: str,
        var: tk.DoubleVar,
        min_value: float,
        max_value: float,
        resolution: float,
    ) -> None:
        row = tk.Frame(parent, bg=self.PANEL_BG)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, bg=self.PANEL_BG, fg=self.TEXT_MUTED, width=12, anchor="w").pack(side="left")
        tk.Scale(
            row,
            variable=var,
            from_=min_value,
            to=max_value,
            resolution=resolution,
            orient="horizontal",
            showvalue=True,
            length=180,
            bg=self.PANEL_BG,
            fg=self.TEXT,
            highlightthickness=0,
            troughcolor="#263349",
            activebackground=self.ACCENT,
        ).pack(side="left", fill="x", expand=True)

    def _add_color_row(self, parent: tk.Widget, label: str, var: tk.StringVar) -> None:
        row = tk.Frame(parent, bg=self.PANEL_ALT)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, bg=self.PANEL_ALT, fg=self.TEXT_MUTED, width=12, anchor="w").pack(side="left")
        tk.Entry(
            row,
            textvariable=var,
            width=10,
            justify="center",
            relief="flat",
            bd=0,
            bg="#e8eef5",
            fg="#112033",
            insertbackground="#112033",
        ).pack(side="left", padx=(0, 4))
        tk.Button(
            row,
            text="Pick",
            command=lambda v=var: self._pick_color(v),
            width=6,
            relief="flat",
            bd=0,
            bg=self.ACCENT,
            fg="#0b1220",
            activebackground="#7ad7ff",
            activeforeground="#0b1220",
        ).pack(side="left")

    def _pick_color(self, var: tk.StringVar) -> None:
        color = colorchooser.askcolor(color=var.get(), title="Choose color")[1]
        if color:
            var.set(color)
            self.pending_visual_changes = True

    def _make_btn(self, parent: tk.Widget, text: str, command, w: int = 10) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            width=w,
            relief="flat",
            bd=0,
            bg=self.ACCENT,
            fg="#0b1220",
            activebackground="#7ad7ff",
            activeforeground="#0b1220",
            font=("Helvetica", 10, "bold"),
            cursor="hand2",
        )

    def _mark_visual_pending(self) -> None:
        self.pending_visual_changes = True

    def _mark_epsilon_pending(self) -> None:
        self.pending_epsilon_changes = True

    def _apply_colors(self) -> None:
        try:
            self._apply_visual_settings()
            self.pending_visual_changes = False
            self.status_var.set("Colors applied")
        except ValueError as exc:
            messagebox.showerror("Invalid Color", str(exc))

    def _reset_colors(self) -> None:
        self.snake_head_color_var.set("#45d483")
        self.snake_body_color_var.set("#1fb86b")
        self.apple_color_var.set("#ff5c74")
        self.grid_color_var.set("#2a3340")
        self.board_bg_color_var.set("#1c2229")
        self.border_color_var.set("#7f8b99")
        self._apply_visual_settings()
        self.pending_visual_changes = False
        self.status_var.set("Colors reset to defaults")

    def _apply_epsilon_settings(self) -> None:
        try:
            epsilon_start = float(self.epsilon_start_var.get().strip())
            epsilon_min = float(self.epsilon_min_var.get().strip())
            epsilon_decay_rate = float(self.epsilon_decay_rate_var.get().strip())
            if not (0.0 <= epsilon_min <= epsilon_start <= 1.0):
                raise ValueError("Require 0 <= epsilon_min <= epsilon_start <= 1")
            if epsilon_decay_rate <= 0.0:
                raise ValueError("Epsilon decay rate must be > 0")
        except ValueError as exc:
            messagebox.showerror("Invalid Epsilon Settings", str(exc))
            return

        with self.cfg_lock:
            base_cfg = self.runtime_cfg if self.runtime_cfg is not None else self.cfg
            updated = replace(
                base_cfg,
                epsilon_start=epsilon_start,
                epsilon_min=epsilon_min,
                epsilon_decay_rate=epsilon_decay_rate,
            )
            self.cfg = updated
            self.runtime_cfg = updated
            self.agent.cfg = updated
            if self.worker is None or not self.worker.is_alive():
                self.agent.epsilon = epsilon_start
        self.pending_epsilon_changes = False
        self.status_var.set(
            f"Epsilon applied: start={epsilon_start:.2f}, min={epsilon_min:.4f}, decay={epsilon_decay_rate:.2f}"
        )

    @staticmethod
    def _format_hidden_layers_for_ui(hidden_layers: tuple[int, ...]) -> str:
        if len(set(hidden_layers)) == 1:
            return str(hidden_layers[0])
        return ",".join(str(width) for width in hidden_layers)

    @staticmethod
    def _parse_int_in_range(raw: str, name: str, min_value: int, max_value: int) -> int:
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"{name} must be an integer.")
        if not (min_value <= value <= max_value):
            raise ValueError(f"{name} must be between {min_value} and {max_value}.")
        return value

    def _validate_hex_color(self, value: str, name: str) -> str:
        if len(value) == 7 and value.startswith("#"):
            hex_part = value[1:]
            if all(ch in "0123456789abcdefABCDEF" for ch in hex_part):
                return value
        raise ValueError(f"{name} must be a hex color like #45d483.")

    def _apply_visual_settings(self) -> None:
        self.snake_head_color = self._validate_hex_color(self.snake_head_color_var.get().strip(), "Snake head color")
        self.snake_body_color = self._validate_hex_color(self.snake_body_color_var.get().strip(), "Snake body color")
        self.apple_color = self._validate_hex_color(self.apple_color_var.get().strip(), "Apple color")
        self.grid_color = self._validate_hex_color(self.grid_color_var.get().strip(), "Grid color")
        self.board_bg_color = self._validate_hex_color(self.board_bg_color_var.get().strip(), "Board background color")
        self.border_color = self._validate_hex_color(self.border_color_var.get().strip(), "Border color")
        self.canvas.configure(bg=self.board_bg_color)

    def _read_cfg_from_ui(self) -> TrainConfig:
        board_size = self._parse_int_in_range(self.board_var.get().strip(), "Board size", min(BOARD_SIZES), max(BOARD_SIZES))
        apples = self._parse_int_in_range(self.apple_var.get().strip(), "Apples", min(APPLE_CHOICES), max(APPLE_CHOICES))
        hidden_layer_count = self._parse_int_in_range(
            self.hidden_layer_count_var.get().strip(),
            "Hidden layers",
            MIN_HIDDEN_LAYERS,
            MAX_HIDDEN_LAYERS,
        )
        hidden_layers = parse_hidden_layer_widths(hidden_layer_count, self.neurons_var.get())
        try:
            anim_delay_ms = float(self.anim_delay_var.get())
        except (TypeError, ValueError):
            raise ValueError("Animation delay must be a number.")
        try:
            epsilon_start = float(self.epsilon_start_var.get().strip())
            epsilon_min = float(self.epsilon_min_var.get().strip())
            epsilon_decay_rate = float(self.epsilon_decay_rate_var.get().strip())
            print_every = int(self.print_every_var.get().strip())
        except ValueError:
            raise ValueError("Epsilon settings and chunk episodes must be numeric.")

        if board_size not in BOARD_SIZES:
            raise ValueError("Board size must be 10, 20, 30, or 40.")
        if apples not in APPLE_CHOICES:
            raise ValueError("Apples must be 1, 3, 5, or 10.")
        if not (0 <= anim_delay_ms <= 1000):
            raise ValueError("Animation delay must be between 0 and 1000 ms.")
        if not (0.0 <= epsilon_min <= epsilon_start <= 1.0):
            raise ValueError("Require 0 <= epsilon_min <= epsilon_start <= 1.")
        if epsilon_decay_rate <= 0.0:
            raise ValueError("Epsilon decay rate must be > 0.")
        if print_every <= 0:
            raise ValueError("Chunk episodes must be > 0.")

        return replace(
            TrainConfig(),
            board_size=board_size,
            apples=apples,
            hidden_layers=hidden_layers,
            epsilon_start=epsilon_start,
            epsilon_min=epsilon_min,
            epsilon_decay_rate=epsilon_decay_rate,
            # Preserve currently loaded model encoding so Watch does not rebuild a fresh agent
            # with a mismatched architecture (e.g., integer12 vs board_cnn).
            state_encoding=self.cfg.state_encoding,
            step_delay=anim_delay_ms / 1000.0,
        )

    def _sync_agent_to_cfg(self, cfg: TrainConfig) -> None:
        if (
            cfg.board_size != self.cfg.board_size
            or cfg.state_encoding != self.cfg.state_encoding
            or cfg.hidden_layers != self.cfg.hidden_layers
        ):
            self.cfg = cfg
            self.agent = SnakeDQNAgent(cfg)
            return

        # Same model shape; keep weights and update runtime config.
        self.cfg = cfg
        self.agent.cfg = cfg

    def _set_runtime_cfg(self, cfg: TrainConfig) -> None:
        with self.cfg_lock:
            self.runtime_cfg = cfg
            self.cfg = cfg
            self.agent.cfg = cfg
    def _get_runtime_cfg(self) -> TrainConfig:
        with self.cfg_lock:
            if self.runtime_cfg is None:
                return self.cfg
            return replace(self.runtime_cfg)

    def _draw_snapshot(self, snapshot: dict) -> None:
        size = snapshot["size"]
        snake = snapshot["snake"]
        apples = snapshot["apples"]

        canvas_w = max(self.canvas.winfo_width(), 200)
        canvas_h = max(self.canvas.winfo_height(), 200)
        cell = max(8, min(canvas_w, canvas_h) // size)

        board_w = cell * size
        board_h = cell * size
        x_off = (canvas_w - board_w) // 2
        y_off = (canvas_h - board_h) // 2

        self.canvas.delete("all")

        for i in range(size + 1):
            pos = i * cell
            self.canvas.create_line(x_off, y_off + pos, x_off + board_w, y_off + pos, fill=self.grid_color)
            self.canvas.create_line(x_off + pos, y_off, x_off + pos, y_off + board_h, fill=self.grid_color)

        self.canvas.create_rectangle(x_off, y_off, x_off + board_w, y_off + board_h, outline=self.border_color, width=2)

        for x, y in apples:
            self.canvas.create_oval(
                x_off + x * cell + 3,
                y_off + y * cell + 3,
                x_off + (x + 1) * cell - 3,
                y_off + (y + 1) * cell - 3,
                fill=self.apple_color,
                outline="",
            )

        for idx, (x, y) in enumerate(snake):
            color = self.snake_head_color if idx == 0 else self.snake_body_color
            self.canvas.create_rectangle(
                x_off + x * cell + 2,
                y_off + y * cell + 2,
                x_off + (x + 1) * cell - 2,
                y_off + (y + 1) * cell - 2,
                fill=color,
                outline="",
            )

    def _update_state_display(self, state: np.ndarray) -> None:
        lines: list[str]
        if state.ndim == 1 and state.shape[0] == 32:
            danger = state[0:4]
            direc = state[4:8]
            food_d = state[8:12]
            food_pos = state[12:15]
            flood = state[15:19]
            tail = state[19:23]
            body_d = state[23:27]
            length = state[27]
            quads = state[28:32]
            lines = [
                "Danger   [U  D  L  R]: " + "  ".join(f"{v:.2f}" for v in danger),
                "Dir      [U  D  L  R]: " + "  ".join(f"{v:.2f}" for v in direc),
                "Food dir [U  D  L  R]: " + "  ".join(f"{v:.2f}" for v in food_d),
                "Food pos [dx dy dist]: " + "  ".join(f"{v:.2f}" for v in food_pos),
                "Flood    [U  D  L  R]: " + "  ".join(f"{v:.2f}" for v in flood),
                "Flood bars           : " + " ".join("#" * int(max(0.0, min(1.0, v)) * 8) for v in flood),
                "Tail [reach dx dy d]: " + "  ".join(f"{v:.2f}" for v in tail),
                "Body dist[U  D  L  R]: " + "  ".join(f"{v:.2f}" for v in body_d),
                f"Length norm: {length:.3f}   Quads [NW NE SW SE]: " + "  ".join(f"{v:.2f}" for v in quads),
            ]
        elif state.ndim == 3 and state.shape[0] >= 8:
            hmap = state[0]
            tmap = state[2]
            amap = state[3]
            occupied = state[4]
            free = state[5]
            dir_x = float(np.mean(state[6]))
            dir_y = float(np.mean(state[7]))
            hy, hx = np.unravel_index(int(np.argmax(hmap)), hmap.shape)
            ty, tx = np.unravel_index(int(np.argmax(tmap)), tmap.shape)
            ay, ax = np.unravel_index(int(np.argmax(amap)), amap.shape) if float(amap.max()) > 0.0 else (hy, hx)
            occ_ratio = float(np.mean(occupied))
            free_ratio = float(np.mean(free))
            lines = [
                f"CNN board state: shape={tuple(state.shape)}",
                f"Head(x,y)=({hx},{hy})  Tail(x,y)=({tx},{ty})  Apple(x,y)=({ax},{ay})",
                f"Dir planes: dx={dir_x:.2f} dy={dir_y:.2f}",
                f"Occupied ratio={occ_ratio:.3f}  Free ratio={free_ratio:.3f}",
                "Channels: [head, body, tail, apple, occupied, free, dir_x, dir_y]",
            ]
        else:
            lines = [f"Unsupported state shape: {tuple(state.shape)}"]

        self.state_text.config(state="normal")
        self.state_text.delete("1.0", "end")
        self.state_text.insert("1.0", "\n".join(lines))
        self.state_text.config(state="disabled")

    def _update_plot(self) -> None:
        return

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                mtype = msg.get("type")

                if mtype == "step":
                    self._draw_snapshot(msg["snapshot"])
                    state = msg.get("state")
                    if state is not None:
                        self._update_state_display(np.asarray(state, dtype=np.float32))

                elif mtype == "episode":
                    score = float(msg["score"])
                    avg = float(msg["avg"])
                    epsilon = float(msg["epsilon"])
                    episode = int(msg["episode"])
                    total = int(msg["total"])
                    elapsed_total = float(msg.get("elapsed_total_sec", 0.0))
                    elapsed_chunk = float(msg.get("elapsed_chunk_sec", 0.0))

                    self.scores.append(score)
                    self.status_var.set(
                        f"Episode {episode}/{total} | Length: {score:.0f} | Avg10: {avg:.2f} | "
                        f"Epsilon: {epsilon:.4f} | Total: {elapsed_total:.1f}s | Chunk: {elapsed_chunk:.1f}s"
                    )

                elif mtype == "done":
                    self.worker = None
                    done_text = msg.get("text", "Finished")
                    self.status_var.set(done_text)
                    if msg.get("ask_save", False):
                        if messagebox.askyesno("Save Model", f"{done_text}\n\nSave model now?"):
                            self.save_model()

                elif mtype == "error":
                    self.worker = None
                    self.status_var.set("Error")
                    messagebox.showerror("Training Error", msg.get("text", "Unknown error"))
        except queue.Empty:
            pass

        self.root.after(80, self._poll_queue)

    def _launch_worker(self, fn: Callable[[], None]) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "Training or watch mode is already running.")
            return

        self.stop_event.clear()
        self.worker = threading.Thread(target=fn, daemon=True)
        self.worker.start()

    def _clear_series(self) -> None:
        self.scores.clear()

    def start_training(self) -> None:
        messagebox.showinfo(
            "Offline Training Only",
            "Run training with train_offline.py.\nThis dashboard is now for loading models/checkpoints and watch mode.",
        )
        self.status_var.set("Training disabled in dashboard. Use train_offline.py")

    def start_watch(self) -> None:
        if self.pending_epsilon_changes:
            messagebox.showinfo("Pending Settings", "Epsilon settings changed. Click Apply Epsilon before watch mode.")
            return
        if self.pending_visual_changes:
            messagebox.showinfo("Pending Settings", "Color settings changed. Click Apply Colors before watch mode.")
            return
        try:
            cfg = self._read_cfg_from_ui()
        except ValueError as exc:
            messagebox.showerror("Invalid Config", str(exc))
            return

        self._sync_agent_to_cfg(cfg)
        self._set_runtime_cfg(cfg)
        self._clear_series()

        def worker() -> None:
            saved_epsilon = self.agent.epsilon
            try:
                watch_cfg = cfg
                self.agent.epsilon = 0.0

                recent_scores: deque[float] = deque(maxlen=10)
                max_watch_episodes = 100000
                episode_game = make_game(watch_cfg)

                def on_step(game: SnakeGame, _step: int, _length: int, _eps: float) -> None:
                    if self.stop_event.is_set():
                        return
                    current_cfg = self._get_runtime_cfg()
                    state = encode_state(game, current_cfg)
                    self.msg_queue.put(
                        {
                            "type": "step",
                            "snapshot": {
                                "size": game.config.grid_size,
                                "snake": list(game.snake),
                                "apples": list(game.apples),
                            },
                            "state": state.tolist(),
                        }
                    )

                for episode in range(1, max_watch_episodes + 1):
                    if self.stop_event.is_set():
                        break
                    current_cfg = self._get_runtime_cfg()
                    watch_cfg = replace(current_cfg, step_delay=current_cfg.step_delay)

                    score, _, _, _ = run_episode(
                        self.agent,
                        watch_cfg,
                        episode_index=episode,
                        train=False,
                        render_step=on_step,
                        stop_flag=self.stop_event,
                        game=episode_game,
                    )

                    recent_scores.append(float(score))
                    avg = float(np.mean(recent_scores))

                    self.msg_queue.put(
                        {
                            "type": "episode",
                            "episode": episode,
                            "total": max_watch_episodes,
                            "score": score,
                            "avg": avg,
                            "epsilon": self.agent.epsilon,
                        }
                    )

                done_text = "Watch stopped" if self.stop_event.is_set() else "Watch complete"
                self.msg_queue.put({"type": "done", "text": done_text, "ask_save": True})
            except Exception as exc:
                self.msg_queue.put({"type": "error", "text": str(exc)})
            finally:
                self.agent.epsilon = saved_epsilon

        self._launch_worker(worker)

    def stop_worker(self) -> None:
        self.stop_event.set()
        self.status_var.set("Stopping...")

    def save_model(self) -> None:
        default_name = os.path.basename(default_model_path(self.cfg.board_size, self.cfg.state_encoding)).replace(".pt", ".ckpt")
        path = filedialog.asksaveasfilename(
            title="Save model",
            initialdir=MODELS_DIR,
            initialfile=default_name,
            defaultextension=".ckpt",
            filetypes=[
                ("Training checkpoint", "*.ckpt"),
                ("PyTorch weights model", "*.pt"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            if path.lower().endswith(".ckpt"):
                self.agent.save_checkpoint(path, episode_index=0, replay_buffer=self.agent.memory)
                self.status_var.set(f"Saved checkpoint: {path}")
            else:
                self.agent.save(path)
                self.status_var.set(f"Saved weights model: {path}")
        except Exception as exc:
            messagebox.showerror("Save Failed", str(exc))

    def save_config(self) -> None:
        try:
            cfg = self._read_cfg_from_ui()
            print_every = int(self.print_every_var.get().strip())
            runtime_cfg = self._get_runtime_cfg()
        except ValueError as exc:
            messagebox.showerror("Invalid Config", str(exc))
            return

        path = filedialog.asksaveasfilename(
            title="Save training config",
            initialdir=MODELS_DIR,
            defaultextension=".json",
            filetypes=[("JSON config", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        config_dict = {
            "board_size": cfg.board_size,
            "apples": cfg.apples,
            "episodes": runtime_cfg.episodes,
            "hidden_layers": list(cfg.hidden_layers),
            "epsilon_start": cfg.epsilon_start,
            "epsilon_min": cfg.epsilon_min,
            "epsilon_decay_rate": cfg.epsilon_decay_rate,
            "food_reward_base": cfg.food_reward_base,
            "food_reward_min": cfg.food_reward_min,
            "survival_reward_start": cfg.survival_reward_start,
            "survival_reward_end": cfg.survival_reward_end,
            "death_reward": cfg.death_reward,
            "terminal_bonus_alpha_start": cfg.terminal_bonus_alpha_start,
            "terminal_bonus_alpha_end": cfg.terminal_bonus_alpha_end,
            "terminal_bonus_power": cfg.terminal_bonus_power,
            "anim_delay_ms": int(round(cfg.step_delay * 1000.0)),
            "chunk_episodes": print_every,
            "colors": {
                "snake_head": self.snake_head_color_var.get().strip(),
                "snake_body": self.snake_body_color_var.get().strip(),
                "apple": self.apple_color_var.get().strip(),
                "grid": self.grid_color_var.get().strip(),
                "board_bg": self.board_bg_color_var.get().strip(),
                "border": self.border_color_var.get().strip(),
            },
        }
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(config_dict, handle, indent=2)
            self.status_var.set(f"Config saved: {path}")
        except Exception as exc:
            messagebox.showerror("Save Failed", str(exc))

    def load_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Load training config",
            initialdir=MODELS_DIR,
            filetypes=[("JSON config", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as handle:
                config_dict = json.load(handle)

            self.board_var.set(str(config_dict.get("board_size", self.cfg.board_size)))
            self.apple_var.set(str(config_dict.get("apples", self.cfg.apples)))
            hidden_layers_raw = config_dict.get("hidden_layers", list(self.cfg.hidden_layers))
            hidden_layers = tuple(int(width) for width in hidden_layers_raw)
            self.hidden_layer_count_var.set(str(len(hidden_layers)))
            self.neurons_var.set(self._format_hidden_layers_for_ui(hidden_layers))
            self.epsilon_start_var.set(f"{float(config_dict.get('epsilon_start', self.cfg.epsilon_start)):.2f}")
            self.epsilon_min_var.set(f"{float(config_dict.get('epsilon_min', self.cfg.epsilon_min)):.4f}")
            self.epsilon_decay_rate_var.set(
                f"{float(config_dict.get('epsilon_decay_rate', self.cfg.epsilon_decay_rate)):.2f}"
            )
            self.anim_delay_var.set(float(config_dict.get("anim_delay_ms", int(round(self.cfg.step_delay * 1000.0)))))
            self.print_every_var.set(str(int(config_dict.get("chunk_episodes", 25))))

            colors = config_dict.get("colors", {})
            self.snake_head_color_var.set(str(colors.get("snake_head", self.snake_head_color_var.get())))
            self.snake_body_color_var.set(str(colors.get("snake_body", self.snake_body_color_var.get())))
            self.apple_color_var.set(str(colors.get("apple", self.apple_color_var.get())))
            self.grid_color_var.set(str(colors.get("grid", self.grid_color_var.get())))
            self.board_bg_color_var.set(str(colors.get("board_bg", self.board_bg_color_var.get())))
            self.border_color_var.set(str(colors.get("border", self.border_color_var.get())))
            self._apply_colors()
            self._apply_epsilon_settings()

            cfg = self._read_cfg_from_ui()
            episodes_raw = config_dict.get("episodes", cfg.episodes)
            try:
                episodes = max(1, int(episodes_raw))
            except (TypeError, ValueError):
                episodes = cfg.episodes
            cfg = replace(cfg, episodes=episodes)
            self._sync_agent_to_cfg(cfg)
            self._set_runtime_cfg(cfg)
            self.status_var.set(f"Config loaded: {path}")
        except Exception as exc:
            messagebox.showerror("Load Failed", str(exc))

    def load_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Load model",
            initialdir=MODELS_DIR,
            filetypes=[
                ("Model or checkpoint", "*.pt *.ckpt"),
                ("Training checkpoint", "*.ckpt"),
                ("PyTorch weights model", "*.pt"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            try:
                metadata = SnakeDQNAgent.load_metadata(path)
            except Exception as meta_exc:
                if path.lower().endswith(".ckpt"):
                    # Fallback path for legacy/non-standard checkpoint metadata.
                    cfg = self._read_cfg_from_ui()
                    tried_errors: list[str] = []
                    for encoding in (cfg.state_encoding, STATE_ENCODING_INTEGER, STATE_ENCODING_BOARD):
                        try:
                            self._sync_agent_to_cfg(replace(cfg, state_encoding=encoding))
                            loaded_episode, _ = self.agent.load_checkpoint(path)
                            self.status_var.set(f"Loaded checkpoint: {path} (episode {loaded_episode})")
                            return
                        except Exception as load_exc:
                            tried_errors.append(f"{encoding}: {load_exc}")
                    raise ValueError(
                        f"Could not load checkpoint metadata ({meta_exc}). "
                        f"Fallback attempts failed -> {' | '.join(tried_errors)}"
                    )
                raise meta_exc
            board_size = metadata.get("board_size", self.cfg.board_size)
            if board_size not in BOARD_SIZES:
                raise ValueError(f"Unsupported board size in model: {board_size}")

            self.board_var.set(str(board_size))

            cfg_data = metadata.get("cfg", {})
            state_encoding = str(metadata.get("state_encoding", cfg_data.get("state_encoding", STATE_ENCODING_INTEGER)))
            if state_encoding == "compact11":
                state_encoding = STATE_ENCODING_INTEGER
            if state_encoding not in SUPPORTED_STATE_ENCODINGS:
                raise ValueError(f"Unsupported state encoding in model: {state_encoding}")
            apples = int(cfg_data.get("apples", self.apple_var.get()))
            if apples in APPLE_CHOICES:
                self.apple_var.set(str(apples))
            hidden_layers = metadata.get("hidden_layers", self.cfg.hidden_layers)
            if not isinstance(hidden_layers, tuple):
                hidden_layers = tuple(int(v) for v in hidden_layers)
            self.hidden_layer_count_var.set(str(len(hidden_layers)))
            self.neurons_var.set(self._format_hidden_layers_for_ui(hidden_layers))

            cfg = replace(self._read_cfg_from_ui(), state_encoding=state_encoding)
            self._sync_agent_to_cfg(cfg)
            if path.lower().endswith(".ckpt"):
                loaded_episode, _ = self.agent.load_checkpoint(path)
                self.status_var.set(f"Loaded checkpoint: {path} (episode {loaded_episode})")
            else:
                self.agent.load(path)
                self.status_var.set(f"Loaded model: {path}")

        except Exception as exc:
            messagebox.showerror("Load Failed", str(exc))

    def _on_close(self) -> None:
        self.stop_event.set()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=0.3)
        self.root.destroy()


def run_training_dashboard() -> None:
    """Entry point for the training/watch GUI."""
    root = tk.Tk()
    TrainingDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    run_training_dashboard()
