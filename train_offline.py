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

try:
    from .agent import SnakeDQNAgent
    from .utils import (
        APPLE_CHOICES,
        BOARD_SIZES,
        TrainConfig,
        chunked_median,
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
        chunked_median,
        default_model_path,
        make_game,
        run_episode,
    )


def _update_progress_plots(ax_trend: plt.Axes, ax_hist: plt.Axes, scores: list[float]) -> None: #type: ignore
    ax_trend.clear()
    ax_trend.set_title("Training Trend (Median per 25 Episodes)")
    ax_trend.set_xlabel("Episode")
    ax_trend.set_ylabel("Length")
    ax_trend.grid(alpha=0.25)

    x25, median25 = chunked_median(scores, chunk_size=25)
    if x25.size > 0:
        ax_trend.plot(
            x25,
            median25,
            color="#1f77b4",
            linewidth=2.2,
            marker="o",
            markersize=3,
            label="Median length (per 25 episodes)",
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
    stop_flag: threading.Event | None = None,
    episode_callback: Callable[[int, float, float, float], None] | None = None,
) -> tuple[SnakeDQNAgent, list[float], list[float]]:
    """Train agent without Tkinter; optionally show a live matplotlib chart."""
    agent = SnakeDQNAgent(cfg)
    if load_path:
        agent.load(load_path)

    scores: list[float] = []
    med25_scores: list[float] = []

    if show_plot:
        plt.ion()
        fig, (ax_trend, ax_hist) = plt.subplots(2, 1, figsize=(10, 8))
        fig.subplots_adjust(hspace=0.35)

    episode_game = make_game(cfg)

    for episode in range(1, cfg.episodes + 1):
        if stop_flag and stop_flag.is_set():
            break

        score, _, _ = run_episode(agent, cfg, train=True, stop_flag=stop_flag, game=episode_game)
        scores.append(float(score))
        med25 = float(np.median(scores[-25:]))
        med25_scores.append(med25)

        agent.decay_epsilon()

        if episode_callback:
            episode_callback(episode, float(score), med25, float(agent.epsilon))

        if show_plot and (episode == 1 or episode % 25 == 0 or episode == cfg.episodes):
            _update_progress_plots(ax_trend, ax_hist, scores) #type: ignore
            fig.canvas.draw_idle() #type: ignore
            fig.canvas.flush_events() #type: ignore
            plt.pause(0.001)

        if episode % 50 == 0:
            print(
                f"Episode {episode}/{cfg.episodes} | "
                f"Length: {score:.0f} | Median25: {med25:.2f} | Epsilon: {agent.epsilon:.4f}"
            )

    out_path = save_path or default_model_path(cfg.board_size)
    agent.save(out_path)

    if show_plot:
        _update_progress_plots(ax_trend, ax_hist, scores) #type: ignore
        plt.ioff()
        plt.show()

    return agent, scores, med25_scores


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Offline Snake DQN training")
    parser.add_argument("--board-size", type=int, default=defaults.board_size, choices=BOARD_SIZES)
    parser.add_argument("--apples", type=int, default=defaults.apples, choices=APPLE_CHOICES)
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--epsilon-decay", type=float, default=defaults.epsilon_decay)
    parser.add_argument("--epsilon-min", type=float, default=defaults.epsilon_min)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--no-plot", action="store_true", help="Disable matplotlib live plot")
    parser.add_argument(
        "--no-distance-shaping",
        action="store_true",
        help="Disable distance-based reward shaping for slightly faster training steps",
    )
    parser.add_argument("--interactive", action="store_true", help="Prompt for settings in the terminal")
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
    default_text = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{label} ({default_text}): ").strip()
        raw_l = raw.lower()
        if raw == "" or raw_l == "default":
            return default
        if raw_l in {"y", "yes"}:
            return True
        if raw_l in {"n", "no"}:
            return False
        print("Enter y or n.")


def prompt_train_config() -> tuple[TrainConfig, str | None, str | None, bool]:
    default_cfg = TrainConfig()
    print("\nSnake offline training setup")
    print("Press Enter to keep defaults.\n")
    print("Tip: type 'default' at the first prompt to skip all setup.\n")

    first = input("Quick start: press Enter to configure, or type 'default' to run with all defaults: ").strip().lower()
    if first == "default":
        # Default quick-start prefers speed: no live plotting.
        return default_cfg, None, None, False

    board_size = _prompt_int("Board size", default_cfg.board_size, choices=BOARD_SIZES)
    apples = _prompt_int("Apples", default_cfg.apples, choices=APPLE_CHOICES)
    episodes = _prompt_int("Episodes", default_cfg.episodes, min_value=1)
    max_steps = _prompt_int("Max steps per episode", default_cfg.max_steps, min_value=1)
    epsilon_decay = _prompt_float("Epsilon decay", default_cfg.epsilon_decay, min_value=0.9, max_value=0.99999)
    epsilon_min = _prompt_float("Epsilon min", default_cfg.epsilon_min, min_value=0.0, max_value=1.0)
    lr = _prompt_float("Learning rate", default_cfg.lr, min_value=1e-8)
    use_distance_shaping = _prompt_bool("Use distance-based reward shaping", default_cfg.distance_reward_shaping)
    show_plot = _prompt_bool("Show live matplotlib plot", False)

    load_raw = input("Model to load (.pt), blank for none: ").strip()
    save_raw = input("Model save path (.pt), blank for default: ").strip()

    cfg = TrainConfig(
        board_size=board_size,
        apples=apples,
        episodes=episodes,
        max_steps=max_steps,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        lr=lr,
        distance_reward_shaping=use_distance_shaping,
    )
    load_path = None if load_raw == "" or load_raw.lower() == "default" else load_raw
    save_path = None if save_raw == "" or save_raw.lower() == "default" else save_raw
    return cfg, load_path, save_path, show_plot


def run_offline_training_cli() -> None:
    args = parse_args()
    interactive = args.interactive or len(sys.argv) == 1
    if interactive:
        cfg, load_path, save_path, show_plot = prompt_train_config()
    else:
        cfg = TrainConfig(
            board_size=args.board_size,
            apples=args.apples,
            episodes=args.episodes,
            max_steps=args.max_steps,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min,
            lr=args.lr,
            distance_reward_shaping=not args.no_distance_shaping,
        )
        load_path = args.load if args.load else None
        save_path = args.save if args.save else None
        show_plot = not args.no_plot

    train_offline(cfg, load_path=load_path, save_path=save_path, show_plot=show_plot)


if __name__ == "__main__":
    run_offline_training_cli()
