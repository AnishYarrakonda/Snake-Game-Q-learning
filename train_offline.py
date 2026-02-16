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
    from .utils import APPLE_CHOICES, BOARD_SIZES, TrainConfig, default_model_path, run_episode
except ImportError:
    from agent import SnakeDQNAgent
    from utils import APPLE_CHOICES, BOARD_SIZES, TrainConfig, default_model_path, run_episode


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
    avg_scores: list[float] = []

    if show_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(9, 5))
        current_line, = ax.plot([], [], label="Current Length", color="#1f77b4")
        avg_line, = ax.plot([], [], label="Average Length", color="#ff7f0e")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Length")
        ax.set_title("Snake Training")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.25)

    for episode in range(1, cfg.episodes + 1):
        if stop_flag and stop_flag.is_set():
            break

        score, _, _ = run_episode(agent, cfg, train=True, stop_flag=stop_flag)
        scores.append(float(score))
        avg = float(np.mean(scores))
        avg_scores.append(avg)

        agent.decay_epsilon()

        if episode_callback:
            episode_callback(episode, float(score), avg, float(agent.epsilon))

        if show_plot and (episode == 1 or episode % 10 == 0 or episode == cfg.episodes):
            x = np.arange(1, len(scores) + 1)
            current_line.set_data(x, scores) #type: ignore
            avg_line.set_data(x, avg_scores) #type: ignore
            ax.relim() #type: ignore
            ax.autoscale_view() #type: ignore
            fig.canvas.draw_idle() #type: ignore
            fig.canvas.flush_events() #type: ignore
            plt.pause(0.001)

        if episode % 50 == 0:
            print(
                f"Episode {episode}/{cfg.episodes} | "
                f"Length: {score:.0f} | Avg: {avg:.2f} | Epsilon: {agent.epsilon:.4f}"
            )

    out_path = save_path or default_model_path(cfg.board_size)
    agent.save(out_path)

    if show_plot:
        plt.ioff()
        plt.show()

    return agent, scores, avg_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Snake DQN training")
    parser.add_argument("--board-size", type=int, default=20, choices=BOARD_SIZES)
    parser.add_argument("--apples", type=int, default=3, choices=APPLE_CHOICES)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--epsilon-decay", type=float, default=0.997)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.0008)
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
        if raw == "":
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
        if raw == "":
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
        raw = input(f"{label} ({default_text}): ").strip().lower()
        if raw == "":
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Enter y or n.")


def prompt_train_config() -> tuple[TrainConfig, str | None, str | None, bool]:
    print("\nSnake offline training setup")
    print("Press Enter to keep defaults.\n")

    board_size = _prompt_int("Board size", 20, choices=BOARD_SIZES)
    apples = _prompt_int("Apples", 3, choices=APPLE_CHOICES)
    episodes = _prompt_int("Episodes", 3000, min_value=1)
    max_steps = _prompt_int("Max steps per episode", 1200, min_value=1)
    epsilon_decay = _prompt_float("Epsilon decay", 0.997, min_value=0.9, max_value=0.99999)
    epsilon_min = _prompt_float("Epsilon min", 0.05, min_value=0.0, max_value=1.0)
    lr = _prompt_float("Learning rate", 0.0008, min_value=1e-8)
    use_distance_shaping = _prompt_bool("Use distance-based reward shaping", True)
    show_plot = _prompt_bool("Show live matplotlib plot", True)

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
    load_path = load_raw or None
    save_path = save_raw or None
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
