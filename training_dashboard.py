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
from tkinter import filedialog, messagebox

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

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Snake DQN Training Dashboard")
        self.root.geometry("1500x980")
        self.root.configure(bg="#101418")

        self.msg_queue: queue.Queue[dict] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None

        self.cfg = TrainConfig()
        self.agent = SnakeDQNAgent(self.cfg)

        self.scores: list[float] = []

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(80, self._poll_queue)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = tk.Frame(self.root, bg="#101418")
        left.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        right = tk.Frame(self.root, bg="#101418")
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)

        controls = tk.LabelFrame(
            left,
            text="Training Controls",
            bg="#0f1720",
            fg="#e6eef7",
            font=("Helvetica", 12, "bold"),
            padx=12,
            pady=10,
        )
        controls.pack(fill="x")

        self.board_var = tk.StringVar(value=str(self.cfg.board_size))
        self.apple_var = tk.StringVar(value=str(self.cfg.apples))
        self.episodes_var = tk.StringVar(value=str(self.cfg.episodes))
        self.max_steps_var = tk.StringVar(value=str(self.cfg.max_steps))
        self.eps_decay_var = tk.StringVar(value=str(self.cfg.epsilon_decay))
        self.eps_min_var = tk.StringVar(value=str(self.cfg.epsilon_min))
        self.lr_var = tk.StringVar(value=str(self.cfg.lr))

        self._add_dropdown(controls, "Board", self.board_var, [str(v) for v in BOARD_SIZES])
        self._add_dropdown(controls, "Apples", self.apple_var, [str(v) for v in APPLE_CHOICES])
        self._add_entry(controls, "Episodes", self.episodes_var)
        self._add_entry(controls, "Max steps", self.max_steps_var)
        self._add_entry(controls, "Epsilon decay", self.eps_decay_var)
        self._add_entry(controls, "Epsilon min", self.eps_min_var)
        self._add_entry(controls, "Learning rate", self.lr_var)

        btn_row = tk.Frame(controls, bg="#0f1720")
        btn_row.pack(fill="x", pady=(8, 0))

        tk.Button(btn_row, text="Train", command=self.start_training, width=10).pack(side="left", padx=2)
        tk.Button(btn_row, text="Watch", command=self.start_watch, width=10).pack(side="left", padx=2)
        tk.Button(btn_row, text="Stop", command=self.stop_worker, width=10).pack(side="left", padx=2)
        tk.Button(btn_row, text="Load", command=self.load_model, width=10).pack(side="left", padx=2)
        tk.Button(btn_row, text="Save", command=self.save_model, width=10).pack(side="left", padx=2)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(
            left,
            textvariable=self.status_var,
            fg="#d9e3ef",
            bg="#101418",
            font=("Helvetica", 12, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(10, 8))

        self.canvas = tk.Canvas(left, bg="#1c2229", width=700, height=700, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        fig = plt.Figure(figsize=(7, 7), dpi=100) #type: ignore
        self.ax_trend = fig.add_subplot(211)
        self.ax_hist = fig.add_subplot(212)
        fig.subplots_adjust(hspace=0.4)

        self.plot_canvas = FigureCanvasTkAgg(fig, master=right)
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _add_entry(self, parent: tk.Widget, label: str, var: tk.StringVar) -> None:
        row = tk.Frame(parent, bg="#0f1720")
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, bg="#0f1720", fg="#dbe7f3", width=12, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=var, width=12, justify="center").pack(side="left")

    def _add_dropdown(self, parent: tk.Widget, label: str, var: tk.StringVar, options: list[str]) -> None:
        row = tk.Frame(parent, bg="#0f1720")
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, bg="#0f1720", fg="#dbe7f3", width=12, anchor="w").pack(side="left")
        tk.OptionMenu(row, var, *options).pack(side="left")

    def _read_cfg_from_ui(self) -> TrainConfig:
        board_size = int(self.board_var.get())
        apples = int(self.apple_var.get())
        episodes = int(self.episodes_var.get())
        max_steps = int(self.max_steps_var.get())
        eps_decay = float(self.eps_decay_var.get())
        eps_min = float(self.eps_min_var.get())
        lr = float(self.lr_var.get())

        if board_size not in BOARD_SIZES:
            raise ValueError("Board size must be 10, 20, 30, or 40.")
        if apples not in APPLE_CHOICES:
            raise ValueError("Apples must be 1, 3, 5, or 10.")
        if episodes <= 0 or max_steps <= 0:
            raise ValueError("Episodes and max steps must be > 0.")
        if not (0.9 <= eps_decay <= 0.99999):
            raise ValueError("Epsilon decay must be between 0.9 and 0.99999.")
        if not (0 < eps_min <= 1.0):
            raise ValueError("Epsilon min must be in (0, 1].")
        if lr <= 0:
            raise ValueError("Learning rate must be > 0.")

        return TrainConfig(
            board_size=board_size,
            apples=apples,
            episodes=episodes,
            max_steps=max_steps,
            epsilon_decay=eps_decay,
            epsilon_min=eps_min,
            lr=lr,
        )

    def _sync_agent_to_cfg(self, cfg: TrainConfig) -> None:
        if cfg.board_size != self.cfg.board_size or cfg.state_encoding != self.cfg.state_encoding:
            self.cfg = cfg
            self.agent = SnakeDQNAgent(cfg)
            return

        # Same model shape; keep weights, update training hyperparameters.
        self.cfg = cfg
        self.agent.cfg = cfg
        for group in self.agent.optimizer.param_groups:
            group["lr"] = cfg.lr

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
            self.canvas.create_line(x_off, y_off + pos, x_off + board_w, y_off + pos, fill="#2a3340")
            self.canvas.create_line(x_off + pos, y_off, x_off + pos, y_off + board_h, fill="#2a3340")

        self.canvas.create_rectangle(x_off, y_off, x_off + board_w, y_off + board_h, outline="#7f8b99", width=2)

        for x, y in apples:
            self.canvas.create_oval(
                x_off + x * cell + 3,
                y_off + y * cell + 3,
                x_off + (x + 1) * cell - 3,
                y_off + (y + 1) * cell - 3,
                fill="#ff5c74",
                outline="",
            )

        for idx, (x, y) in enumerate(snake):
            color = "#45d483" if idx == 0 else "#1fb86b"
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
                    self.status_var.set(msg.get("text", "Finished"))

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

                for episode in range(1, cfg.episodes + 1):
                    if self.stop_event.is_set():
                        break

                    score, _, _ = run_episode(
                        self.agent,
                        cfg,
                        train=True,
                        render_step=on_step,
                        stop_flag=self.stop_event,
                        game=episode_game,
                    )
                    self.agent.decay_epsilon()

                    recent_scores.append(float(score))
                    avg = float(np.mean(recent_scores))

                    self.msg_queue.put(
                        {
                            "type": "episode",
                            "episode": episode,
                            "total": cfg.episodes,
                            "score": score,
                            "avg": avg,
                            "epsilon": self.agent.epsilon,
                        }
                    )

                done_text = "Training stopped" if self.stop_event.is_set() else "Training complete"
                self.msg_queue.put({"type": "done", "text": done_text})
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
        self._clear_series()

        def worker() -> None:
            saved_epsilon = self.agent.epsilon
            try:
                watch_cfg = replace(cfg, step_delay=0.05)
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

                    score, _, _ = run_episode(
                        self.agent,
                        watch_cfg,
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
                self.msg_queue.put({"type": "done", "text": done_text})
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

            if "epsilon_decay" in cfg_data:
                self.eps_decay_var.set(str(cfg_data["epsilon_decay"]))
            if "epsilon_min" in cfg_data:
                self.eps_min_var.set(str(cfg_data["epsilon_min"]))
            if "lr" in cfg_data:
                self.lr_var.set(str(cfg_data["lr"]))

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
