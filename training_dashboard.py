# Tkinter training dashboard with live board view + live matplotlib training graph.
from __future__ import annotations

from collections import deque
from dataclasses import replace
import os
import queue
import threading
from typing import Callable

# Keep matplotlib cache local for environments without writable home config.
LOCAL_MPLCONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig")
os.makedirs(LOCAL_MPLCONFIG, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", LOCAL_MPLCONFIG)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox

try:
    from .agent import SnakeDQNAgent
    from .game_logic import SnakeGame
    from .utils import (
        APPLE_CHOICES,
        BOARD_SIZES,
        MODELS_DIR,
        TrainConfig,
        chunked_mean,
        default_model_path,
        make_game,
        run_episode,
    )
except ImportError:
    from agent import SnakeDQNAgent
    from game_logic import SnakeGame
    from utils import (
        APPLE_CHOICES,
        BOARD_SIZES,
        MODELS_DIR,
        TrainConfig,
        chunked_mean,
        default_model_path,
        make_game,
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
        self.root.geometry("1640x980")
        self.root.configure(bg=self.BG_MAIN)

        self.msg_queue: queue.Queue[dict] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.cfg_lock = threading.Lock()
        self.runtime_cfg: TrainConfig | None = None
        self.active_training_immutable: tuple[int, int, int] | None = None  # board, apples, episodes

        self.cfg = TrainConfig()
        self.agent = SnakeDQNAgent(self.cfg)

        self.scores: list[float] = []

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(80, self._poll_queue)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)

        left = tk.Frame(self.root, bg=self.BG_MAIN)
        left.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        left.rowconfigure(2, weight=1)
        left.columnconfigure(0, weight=1)

        right = tk.Frame(self.root, bg=self.BG_MAIN)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)

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
        self.anim_delay_var = tk.DoubleVar(value=0.0)
        self.snake_head_color_var = tk.StringVar(value="#45d483")
        self.snake_body_color_var = tk.StringVar(value="#1fb86b")
        self.apple_color_var = tk.StringVar(value="#ff5c74")
        self.grid_color_var = tk.StringVar(value="#2a3340")
        self.board_bg_color_var = tk.StringVar(value="#1c2229")
        self.border_color_var = tk.StringVar(value="#7f8b99")

        columns = tk.Frame(controls, bg=self.PANEL_BG)
        columns.pack(fill="x")
        left_col = tk.Frame(columns, bg=self.PANEL_BG)
        right_col = tk.Frame(columns, bg=self.PANEL_BG)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right_col.pack(side="left", fill="both", expand=True, padx=(8, 0))

        self._add_dropdown(left_col, "Board", self.board_var, [str(v) for v in BOARD_SIZES])
        self._add_dropdown(left_col, "Apples", self.apple_var, [str(v) for v in APPLE_CHOICES])
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

        btn_row = tk.Frame(controls, bg=self.PANEL_BG)
        btn_row.pack(fill="x", pady=(8, 0))

        self._make_btn(btn_row, "Train", self.start_training, w=10).pack(side="left", padx=2)
        self._make_btn(btn_row, "Watch", self.start_watch, w=10).pack(side="left", padx=2)
        self._make_btn(btn_row, "Stop", self.stop_worker, w=10).pack(side="left", padx=2)
        self._make_btn(btn_row, "Load", self.load_model, w=10).pack(side="left", padx=2)
        self._make_btn(btn_row, "Save", self.save_model, w=10).pack(side="left", padx=2)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(
            left,
            textvariable=self.status_var,
            fg=self.TEXT,
            bg=self.BG_MAIN,
            font=("Helvetica", 12, "bold"),
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", pady=(10, 8))

        self.canvas = tk.Canvas(left, bg=self.board_bg_color_var.get(), width=940, height=760, highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew")

        fig = plt.Figure(figsize=(7, 7), dpi=100) #type: ignore
        self.ax_trend = fig.add_subplot(211)
        self.ax_hist = fig.add_subplot(212)
        fig.subplots_adjust(hspace=0.4)

        self.plot_canvas = FigureCanvasTkAgg(fig, master=right)
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)
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
            self._apply_visual_settings()

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
        board_size = int(self.board_var.get())
        apples = int(self.apple_var.get())
        anim_delay_ms = float(self.anim_delay_var.get())

        if board_size not in BOARD_SIZES:
            raise ValueError("Board size must be 10, 20, 30, or 40.")
        if apples not in APPLE_CHOICES:
            raise ValueError("Apples must be 1, 3, 5, or 10.")
        if not (0 <= anim_delay_ms <= 1000):
            raise ValueError("Animation delay must be between 0 and 1000 ms.")

        return replace(
            TrainConfig(),
            board_size=board_size,
            apples=apples,
            step_delay=anim_delay_ms / 1000.0,
        )

    def _sync_agent_to_cfg(self, cfg: TrainConfig) -> None:
        if cfg.board_size != self.cfg.board_size or cfg.state_encoding != self.cfg.state_encoding:
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

    def _update_plot(self) -> None:
        self.ax_trend.clear()
        self.ax_trend.set_title("Training Trend (Average per 10 Episodes)")
        self.ax_trend.set_xlabel("Episode")
        self.ax_trend.set_ylabel("Length")
        self.ax_trend.grid(alpha=0.25)

        self.ax_hist.clear()
        self.ax_hist.set_title("Episode Length Distribution")
        self.ax_hist.set_xlabel("Length")
        self.ax_hist.set_ylabel("Count")
        self.ax_hist.grid(alpha=0.2)

        if not self.scores:
            self.plot_canvas.draw_idle()
            return

        x10, mean10 = chunked_mean(self.scores, chunk_size=10)
        if x10.size > 0:
            self.ax_trend.plot(
                x10,
                mean10,
                color="#1f77b4",
                linewidth=2.2,
                marker="o",
                markersize=3,
                label="Average length (per 10 episodes)",
            )
        handles, labels = self.ax_trend.get_legend_handles_labels()
        if handles:
            self.ax_trend.legend(loc="upper left")

        max_score = int(max(self.scores))
        bins = np.arange(0.5, max_score + 1.5, 1.0)
        self.ax_hist.hist(self.scores, bins=bins, color="#44b5a4", alpha=0.85, edgecolor="#17323a") #type: ignore
        mean_all = float(np.mean(self.scores))
        median_all = float(np.median(self.scores))
        self.ax_hist.axvline(mean_all, color="#1f77b4", linestyle="--", linewidth=1.6, label=f"Mean: {mean_all:.2f}")
        self.ax_hist.axvline(median_all, color="#ff7f0e", linestyle="-", linewidth=1.6, label=f"Median: {median_all:.2f}")
        handles, labels = self.ax_hist.get_legend_handles_labels()
        if handles:
            self.ax_hist.legend(loc="upper right")

        self.plot_canvas.draw_idle()

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                mtype = msg.get("type")

                if mtype == "step":
                    self._draw_snapshot(msg["snapshot"])

                elif mtype == "episode":
                    score = float(msg["score"])
                    avg = float(msg["avg"])
                    epsilon = float(msg["epsilon"])
                    episode = int(msg["episode"])
                    total = int(msg["total"])

                    self.scores.append(score)
                    self._update_plot()
                    self.status_var.set(
                        f"Episode {episode}/{total} | Length: {score:.0f} | Avg10: {avg:.2f} | Epsilon: {epsilon:.4f}"
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
        self._update_plot()

    def start_training(self) -> None:
        try:
            cfg = self._read_cfg_from_ui()
        except ValueError as exc:
            messagebox.showerror("Invalid Config", str(exc))
            return

        self._sync_agent_to_cfg(cfg)
        self._set_runtime_cfg(cfg)
        self.active_training_immutable = (cfg.board_size, cfg.apples, cfg.episodes)
        print(f"\nUsing device: {self.agent.device}\n")
        self.status_var.set(f"Ready | Device: {self.agent.device}")
        self._clear_series()

        def worker() -> None:
            try:
                recent_scores: deque[float] = deque(maxlen=10)
                episode_game = make_game(cfg)

                def on_step(game: SnakeGame, _step: int, _length: int, _eps: float) -> None:
                    if self.stop_event.is_set():
                        return
                    self.msg_queue.put(
                        {
                            "type": "step",
                            "snapshot": {
                                "size": cfg.board_size,
                                "snake": list(game.snake),
                                "apples": list(game.apples),
                            },
                        }
                    )

                total_episodes = cfg.episodes
                for episode in range(1, total_episodes + 1):
                    if self.stop_event.is_set():
                        break
                    current_cfg = self._get_runtime_cfg()

                    score, _, _ = run_episode(
                        self.agent,
                        current_cfg,
                        episode_index=episode,
                        train=True,
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
                            "total": total_episodes,
                            "score": score,
                            "avg": avg,
                            "epsilon": self.agent.epsilon,
                        }
                    )

                done_text = "Training stopped" if self.stop_event.is_set() else "Training complete"
                self.msg_queue.put({"type": "done", "text": done_text, "ask_save": True})
            except Exception as exc:
                self.msg_queue.put({"type": "error", "text": str(exc)})

        self._launch_worker(worker)

    def start_watch(self) -> None:
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
                    self.msg_queue.put(
                        {
                            "type": "step",
                            "snapshot": {
                                "size": watch_cfg.board_size,
                                "snake": list(game.snake),
                                "apples": list(game.apples),
                            },
                        }
                    )

                for episode in range(1, max_watch_episodes + 1):
                    if self.stop_event.is_set():
                        break
                    current_cfg = self._get_runtime_cfg()
                    watch_cfg = replace(current_cfg, step_delay=current_cfg.step_delay)

                    score, _, _ = run_episode(
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
        default_name = os.path.basename(default_model_path(self.cfg.board_size, self.cfg.state_encoding))
        path = filedialog.asksaveasfilename(
            title="Save model",
            initialdir=MODELS_DIR,
            initialfile=default_name,
            defaultextension=".pt",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self.agent.save(path)
            self.status_var.set(f"Saved model: {path}")
        except Exception as exc:
            messagebox.showerror("Save Failed", str(exc))

    def load_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Load model",
            initialdir=MODELS_DIR,
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            metadata = SnakeDQNAgent.load_metadata(path)
            board_size = metadata.get("board_size", self.cfg.board_size)
            if board_size not in BOARD_SIZES:
                raise ValueError(f"Unsupported board size in model: {board_size}")

            self.board_var.set(str(board_size))

            cfg_data = metadata.get("cfg", {})
            state_encoding = str(metadata.get("state_encoding", cfg_data.get("state_encoding", self.cfg.state_encoding)))
            if state_encoding not in {"compact11", "board"}:
                raise ValueError(f"Unsupported state encoding in model: {state_encoding}")
            apples = int(cfg_data.get("apples", self.apple_var.get()))
            if apples in APPLE_CHOICES:
                self.apple_var.set(str(apples))

            cfg = self._read_cfg_from_ui()
            cfg = replace(cfg, state_encoding=state_encoding)
            self._sync_agent_to_cfg(cfg)
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
