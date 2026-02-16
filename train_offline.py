# Headless/offline training entrypoint with optional live matplotlib graph.
from __future__ import annotations

import argparse
import os
import sys
import threading
from typing import Callable

# Keep matplotlib cache local for environments without writable home config.
LOCAL_MPLCONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig")
os.makedirs(LOCAL_MPLCONFIG, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", LOCAL_MPLCONFIG)

import matplotlib.pyplot as plt
import numpy as np
from collections import deque

try:
    from .agent import SnakeDQNAgent
    from .utils import (
        APPLE_CHOICES,
        BOARD_SIZES,
        TrainConfig,
        chunked_mean,
        default_model_path,
        make_game,
        run_episode,
    )
except ImportError:
    from agent import SnakeDQNAgent
    from utils import (
        APPLE_CHOICES,
        BOARD_SIZES,
        TrainConfig,
        chunked_mean,
        default_model_path,
        make_game,
        run_episode,
    )


def _update_progress_plots(ax_trend: plt.Axes, ax_hist: plt.Axes, scores: list[float]) -> None: #type: ignore
    ax_trend.clear()
    ax_trend.set_title("Training Trend (Average per 10 Episodes)")
    ax_trend.set_xlabel("Episode")
    ax_trend.set_ylabel("Length")
    ax_trend.grid(alpha=0.25)

    x10, mean10 = chunked_mean(scores, chunk_size=10)
    if x10.size > 0:
        ax_trend.plot(
            x10,
            mean10,
            color="#1f77b4",
            linewidth=2.2,
            marker="o",
            markersize=3,
            label="Average length (per 10 episodes)",
        )
    handles, labels = ax_trend.get_legend_handles_labels()
    if handles:
        ax_trend.legend(loc="upper left")

    ax_hist.clear()
    ax_hist.set_title("Episode Length Distribution")
    ax_hist.set_xlabel("Length")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(alpha=0.2)

    if scores:
        max_score = int(max(scores))
        bins = np.arange(0.5, max_score + 1.5, 1.0)
        ax_hist.hist(scores, bins=bins, color="#44b5a4", alpha=0.85, edgecolor="#17323a") #type: ignore
        mean_all = float(np.mean(scores))
        median_all = float(np.median(scores))
        ax_hist.axvline(mean_all, color="#1f77b4", linestyle="--", linewidth=1.6, label=f"Mean: {mean_all:.2f}")
        ax_hist.axvline(median_all, color="#ff7f0e", linestyle="-", linewidth=1.6, label=f"Median: {median_all:.2f}")
        handles, labels = ax_hist.get_legend_handles_labels()
        if handles:
            ax_hist.legend(loc="upper right")


def train_offline(
    cfg: TrainConfig,
    load_path: str | None = None,
    save_path: str | None = None,
    show_plot: bool = True,
    print_every: int = 25,
    show_final_trend_plot: bool = True,
    stop_flag: threading.Event | None = None,
    episode_callback: Callable[[int, float, float, float], None] | None = None,
) -> tuple[SnakeDQNAgent, list[float], list[float]]:
    """Train agent without Tkinter; optionally show a live matplotlib chart."""
    agent = SnakeDQNAgent(cfg)
    if load_path:
        agent.load(load_path)

    scores: list[float] = []
    avg10_scores: list[float] = []
    recent_10: deque[float] = deque(maxlen=10)
    log_chunk_size = max(1, int(print_every))
    recent_chunk: deque[float] = deque(maxlen=log_chunk_size)
    chunk_episode_ends: list[int] = []
    chunk_avg_scores: list[float] = []
    chunk_median_scores: list[float] = []

    print(f"\nUsing device: {agent.device}\n")

    header = (
        f"{'Episodes':^18}"
        f"{'Last':^10}"
        f"{'Avg':^10}"
        f"{'Median':^10}"
        f"{'Max':^10}"
        f"{'Epsilon':^12}"
    )
    print(header)
    print("-" * len(header))

    if show_plot:
        plt.ion()
        fig, (ax_trend, ax_hist) = plt.subplots(2, 1, figsize=(10, 8))
        fig.subplots_adjust(hspace=0.35)

    episode_game = make_game(cfg)

    for episode in range(1, cfg.episodes + 1):
        if stop_flag and stop_flag.is_set():
            break

        score, _, _ = run_episode(
            agent,
            cfg,
            episode_index=episode,
            train=True,
            stop_flag=stop_flag,
            game=episode_game,
        )
        scores.append(float(score))
        recent_10.append(float(score))
        recent_chunk.append(float(score))
        avg10 = float(np.mean(recent_10))
        avg10_scores.append(avg10)

        if episode_callback:
            episode_callback(episode, float(score), avg10, float(agent.epsilon))

        if show_plot and (episode == 1 or episode % 10 == 0 or episode == cfg.episodes):
            _update_progress_plots(ax_trend, ax_hist, scores) #type: ignore
            fig.canvas.draw_idle() #type: ignore
            fig.canvas.flush_events() #type: ignore
            plt.pause(0.001)

        if episode % log_chunk_size == 0 or episode == cfg.episodes:
            avg_chunk = float(np.mean(recent_chunk))
            median_chunk = float(np.median(recent_chunk))
            max_chunk = float(np.max(recent_chunk))
            chunk_episode_ends.append(episode)
            chunk_avg_scores.append(avg_chunk)
            chunk_median_scores.append(median_chunk)
            range_start = episode - len(recent_chunk) + 1
            episode_label = f"{range_start}-{episode}/{cfg.episodes}"
            row = (
                f"{episode_label:^18}"
                f"{score:^10.0f}"
                f"{avg_chunk:^10.2f}"
                f"{median_chunk:^10.2f}"
                f"{max_chunk:^10.0f}"
                f"{agent.epsilon:^12.4f}"
            )
            print(row)
            print()

    if save_path:
        agent.save(save_path)

    if show_plot:
        _update_progress_plots(ax_trend, ax_hist, scores) #type: ignore
        plt.ioff()
        plt.show()
    elif show_final_trend_plot and chunk_episode_ends:
        plt.figure(figsize=(9, 5))
        plt.plot(
            chunk_episode_ends,
            chunk_avg_scores,
            color="#1f77b4",
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=f"Average (per {log_chunk_size})",
        )
        plt.plot(
            chunk_episode_ends,
            chunk_median_scores,
            color="#ff7f0e",
            linewidth=2.0,
            marker="s",
            markersize=4,
            label=f"Median (per {log_chunk_size})",
        )
        plt.title("Training Plateau Trend")
        plt.xlabel("Episode")
        plt.ylabel("Length")
        plt.grid(alpha=0.25)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    return agent, scores, avg10_scores


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Offline Snake DQN training")
    parser.add_argument("--board-size", type=int, default=defaults.board_size, choices=BOARD_SIZES)
    parser.add_argument("--apples", type=int, default=defaults.apples, choices=APPLE_CHOICES)
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--save", type=str, default="")
    parser.add_argument(
        "--print-every",
        type=int,
        default=25,
        help="Print training stats every N episodes (also used for chunk average/median trend).",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable matplotlib live plot")
    parser.add_argument(
        "--no-final-trend-plot",
        action="store_true",
        help="Disable end-of-training average/median trend plot when live plot is off.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for settings in the terminal (board/apples/plot/load).",
    )
    return parser.parse_args()


def _prompt_int(label: str, default: int, choices: tuple[int, ...] | None = None, min_value: int | None = None) -> int:
    while True:
        options = f" {choices}" if choices else ""
        raw = input(f"{label}{options} [{default}]: ").strip()
        raw_l = raw.lower()
        if raw == "" or raw_l == "default":
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Enter a valid integer.")
            continue
        if choices and value not in choices:
            print(f"Choose one of: {choices}")
            continue
        if min_value is not None and value < min_value:
            print(f"Value must be >= {min_value}")
            continue
        return value


def _prompt_float(label: str, default: float, min_value: float | None = None, max_value: float | None = None) -> float:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        raw_l = raw.lower()
        if raw == "" or raw_l == "default":
            return default
        try:
            value = float(raw)
        except ValueError:
            print("Enter a valid number.")
            continue
        if min_value is not None and value < min_value:
            print(f"Value must be >= {min_value}")
            continue
        if max_value is not None and value > max_value:
            print(f"Value must be <= {max_value}")
            continue
        return value


def _prompt_bool(label: str, default: bool) -> bool:
    default_hint = "y" if default else "n"
    while True:
        raw = input(f"{label} (y/n) [default: {default_hint}]: ").strip()
        raw_l = raw.lower()
        if raw == "" or raw_l == "default":
            return default
        if raw_l in {"y", "yes"}:
            return True
        if raw_l in {"n", "no"}:
            return False
        print("Enter y or n.")


def prompt_train_config() -> tuple[TrainConfig, str | None, bool, int, bool]:
    default_cfg = TrainConfig()
    print("\nSnake offline training setup")
    print("Press Enter to keep defaults.\n")
    print("Tip: type 'default' at the first prompt to skip all setup.\n")

    first = input("Quick start: press Enter to configure, or type 'default' to run with all defaults: ").strip().lower()
    if first == "default":
        # Default quick-start prefers speed: no live plotting.
        return default_cfg, None, False, 25, True

    board_size = _prompt_int("Board size", default_cfg.board_size, choices=BOARD_SIZES)
    apples = _prompt_int("Apples", default_cfg.apples, choices=APPLE_CHOICES)
    show_plot = _prompt_bool("Show live matplotlib plot", False)
    print_every = _prompt_int("Print stats every N episodes", 25, min_value=1)
    show_final_trend_plot = _prompt_bool("Show end trend plot when live plot is off", True)

    load_raw = input("Model to load (.pt), blank for none: ").strip()
    cfg = TrainConfig(
        board_size=board_size,
        apples=apples,
        state_encoding=default_cfg.state_encoding,
    )
    load_path = None if load_raw == "" or load_raw.lower() == "default" else load_raw
    return cfg, load_path, show_plot, print_every, show_final_trend_plot


def _prompt_save_after_training(agent: SnakeDQNAgent, cfg: TrainConfig) -> None:
    while True:
        choice = input("\nSave model now? (y/n): ").strip().lower()
        if choice in {"y", "yes"}:
            default_path = default_model_path(cfg.board_size, cfg.state_encoding)
            raw = input(f"Save path [{default_path}]: ").strip()
            path = default_path if raw == "" else raw
            try:
                agent.save(path)
                print(f"Saved model to: {path}")
            except Exception as exc:
                print(f"Save failed: {exc}")
            return
        if choice in {"n", "no", ""}:
            print("Model not saved.")
            return
        print("Enter y or n.")


def run_offline_training_cli() -> None:
    args = parse_args()
    interactive = args.interactive or len(sys.argv) == 1
    if interactive:
        cfg, load_path, show_plot, print_every, show_final_trend_plot = prompt_train_config()
        save_path = None
    else:
        cfg = TrainConfig(
            board_size=args.board_size,
            apples=args.apples,
            state_encoding=TrainConfig().state_encoding,
        )
        load_path = args.load if args.load else None
        save_path = args.save if args.save else None
        show_plot = not args.no_plot
        print_every = args.print_every
        show_final_trend_plot = not args.no_final_trend_plot

    agent, _, _ = train_offline(
        cfg,
        load_path=load_path,
        save_path=save_path,
        show_plot=show_plot,
        print_every=print_every,
        show_final_trend_plot=show_final_trend_plot,
    )

    if interactive and not save_path:
        _prompt_save_after_training(agent, cfg)


if __name__ == "__main__":
    run_offline_training_cli()
