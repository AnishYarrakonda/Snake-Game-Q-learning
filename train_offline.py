# Headless/offline training entrypoint with optional live matplotlib graph.
from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
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
        MAX_HIDDEN_LAYERS,
        MIN_HIDDEN_LAYERS,
        STATE_ENCODING_INTEGER,
        TrainConfig,
        clear_escape_cache,
        chunked_mean,
        default_model_path,
        make_game,
        parse_hidden_layer_widths,
        run_episode,
    )
except ImportError:
    from agent import SnakeDQNAgent
    from utils import (
        APPLE_CHOICES,
        BOARD_SIZES,
        MAX_HIDDEN_LAYERS,
        MIN_HIDDEN_LAYERS,
        STATE_ENCODING_INTEGER,
        TrainConfig,
        clear_escape_cache,
        chunked_mean,
        default_model_path,
        make_game,
        parse_hidden_layer_widths,
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


def _default_checkpoint_path(cfg: TrainConfig) -> str:
    base_model = default_model_path(cfg.board_size, cfg.state_encoding)
    root, _ = os.path.splitext(base_model)
    return f"{root}.ckpt"


def _periodic_checkpoint_path(base_path: str, episode: int) -> str:
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".ckpt"
    # If base already ends with an episode suffix, replace it.
    root = re.sub(r"([_-](?:ep)?\d+)$", "", root, flags=re.IGNORECASE)
    return f"{root}_{episode}{ext}"


def _print_progress_bar(episode: int, total: int, bar_length: int = 50) -> None:
    """Print a compact progress bar in the terminal."""
    total_safe = max(1, int(total))
    percent = min(1.0, max(0.0, episode / total_safe))
    filled = int(bar_length * percent)
    bar = "#" * filled + "-" * (bar_length - filled)
    print(f"\rProgress: |{bar}| {episode}/{total_safe} ({percent * 100:.1f}%)", end="", flush=True)


def train_offline(
    cfg: TrainConfig,
    load_path: str | None = None,
    save_path: str | None = None,
    resume_path: str | None = None,
    save_weights_path: str | None = None,
    save_checkpoint_every: int = 0,
    checkpoint_base_path: str | None = None,
    show_plot: bool = True,
    print_every: int = 25,
    show_final_trend_plot: bool = True,
    early_stop_patience: int = 500,
    early_stop_threshold: float = 0.02,
    stop_flag: threading.Event | None = None,
    episode_callback: Callable[[int, float, float, float], None] | None = None,
) -> tuple[SnakeDQNAgent, list[float], list[float]]:
    """Train agent without Tkinter; optionally show a live matplotlib chart."""
    agent = SnakeDQNAgent(cfg)
    start_episode = 1

    if load_path and resume_path:
        raise ValueError("Use only one of load_path (weights-only) or resume_path (full checkpoint).")

    if resume_path:
        loaded_episode, _ = agent.load_checkpoint(resume_path)
        if agent.loaded_legacy_payload:
            print(
                "Loaded legacy weights-only payload from resume path; optimizer/replay/episode state unavailable. "
                "Starting a fresh training run from episode 1."
            )
            print(f"Resuming training from episode 0, epsilon={agent.epsilon:.6f}, best_score={agent.best_score:.0f}")
            start_episode = 1
        else:
            start_episode = loaded_episode + 1
            print(
                f"Resuming training from episode {loaded_episode}, "
                f"epsilon={agent.epsilon:.6f}, best_score={agent.best_score:.0f}"
            )

    if load_path:
        agent.load(load_path)
        start_episode = 1

    if start_episode > cfg.episodes:
        print(
            f"Checkpoint already at episode {start_episode - 1}, "
            f"which is >= configured total episodes ({cfg.episodes}). No new episodes to run."
        )

    scores: list[float] = []
    avg10_scores: list[float] = []
    recent_10: deque[float] = deque(maxlen=10)
    log_chunk_size = max(1, int(print_every))
    recent_chunk: deque[float] = deque(maxlen=log_chunk_size)
    recent_steps: deque[float] = deque(maxlen=log_chunk_size)
    recent_q_mean: deque[float] = deque(maxlen=log_chunk_size)
    recent_td_error: deque[float] = deque(maxlen=log_chunk_size)
    recent_food_r: deque[float] = deque(maxlen=log_chunk_size)
    recent_survival_r: deque[float] = deque(maxlen=log_chunk_size)
    recent_safety_bonus: deque[float] = deque(maxlen=log_chunk_size)
    recent_terminal_bonus: deque[float] = deque(maxlen=log_chunk_size)
    recent_loss: deque[float] = deque(maxlen=log_chunk_size)
    recent_total_reward: deque[float] = deque(maxlen=log_chunk_size)
    recent_q_explodes: deque[int] = deque(maxlen=log_chunk_size)
    rolling_scores_500: deque[float] = deque(maxlen=500)
    chunk_episode_ends: list[int] = []
    chunk_avg_scores: list[float] = []
    chunk_median_scores: list[float] = []
    checkpoint_every = max(0, int(save_checkpoint_every))
    if checkpoint_every == 0:
        checkpoint_every = max(1, int(cfg.episodes * 0.10))
    checkpoint_seed_path = checkpoint_base_path or save_path or resume_path or _default_checkpoint_path(cfg)
    last_completed_episode = start_episode - 1
    last_periodic_checkpoint_path: str | None = None
    best_rolling_avg = 0.0
    patience_counter = 0
    early_stop_triggered = False

    print(f"\nUsing device: {agent.device}\n")
    print(f"Checkpoint cadence: every {checkpoint_every} episodes")

    header = (
        f"{'Episodes':<18}"
        f"{'TotSec':>9}"
        f"{'ChunkSec':>10}"
        f"{'Last':>6}"
        f"{'Avg':>8}"
        f"{'Med':>8}"
        f"{'Max':>6}"
        f"{'Max500':>8}"
        f"{'Steps':>8}"
        f"{'Eps':>8}"
        f"{'MeanQ':>9}"
        f"{'TDerr':>9}"
        f"{'FoodR':>9}"
        f"{'SurvR':>9}"
        f"{'SafeB':>9}"
        f"{'Bonus':>9}"
        f"{'Loss':>10}"
        f"{'TotR':>9}"
        f"{'QExp':>6}"
    )

    print(header)
    print("-" * len(header))

    if show_plot:
        plt.ion()
        fig, (ax_trend, ax_hist) = plt.subplots(2, 1, figsize=(10, 8))
        fig.subplots_adjust(hspace=0.35)

    episode_game = make_game(cfg)
    train_start_t = time.perf_counter()
    chunk_start_t = train_start_t

    for episode in range(start_episode, cfg.episodes + 1):
        if stop_flag and stop_flag.is_set():
            break
        if episode % 100 == 0:
            clear_escape_cache()
        _print_progress_bar(episode, cfg.episodes)

        score, total_reward, steps_taken, episode_stats = run_episode(
            agent,
            cfg,
            episode_index=episode,
            train=True,
            stop_flag=stop_flag,
            game=episode_game,
        )
        scores.append(float(score))
        rolling_scores_500.append(float(score))
        recent_10.append(float(score))
        recent_chunk.append(float(score))
        recent_steps.append(float(steps_taken))
        recent_q_mean.append(float(episode_stats["mean_q"]))
        recent_td_error.append(float(episode_stats["td_error_mean"]))
        recent_food_r.append(float(episode_stats["avg_food_reward"]))
        recent_survival_r.append(float(episode_stats["avg_survival_reward"]))
        recent_safety_bonus.append(float(episode_stats["avg_safety_bonus"]))
        recent_terminal_bonus.append(float(episode_stats["terminal_bonus"]))
        recent_loss.append(float(episode_stats["loss"]))
        recent_total_reward.append(float(total_reward))
        recent_q_explodes.append(int(abs(float(episode_stats["max_abs_q"])) > cfg.q_explosion_threshold))
        avg10 = float(np.mean(recent_10))
        avg10_scores.append(avg10)
        agent.best_score = max(agent.best_score, float(score))
        last_completed_episode = episode
        if early_stop_patience > 0 and episode % 25 == 0 and len(rolling_scores_500) >= 100:
            current_avg = float(np.mean(rolling_scores_500))
            if current_avg > best_rolling_avg + early_stop_threshold:
                best_rolling_avg = current_avg
                patience_counter = 0
            else:
                patience_counter += 25
            if patience_counter >= early_stop_patience:
                early_stop_triggered = True
                print()
                print(f"Early stopping at episode {episode}: no improvement for {patience_counter} episodes")
                print(f"Best rolling average: {best_rolling_avg:.2f}")
                break

        if episode_callback:
            episode_callback(episode, float(score), avg10, float(agent.epsilon))

        if show_plot and (episode == start_episode or episode % 10 == 0 or episode == cfg.episodes):
            _update_progress_plots(ax_trend, ax_hist, scores) #type: ignore
            fig.canvas.draw_idle() #type: ignore
            fig.canvas.flush_events() #type: ignore
            plt.pause(0.001)

        if episode % log_chunk_size == 0 or episode == cfg.episodes:
            print()
            total_elapsed = time.perf_counter() - train_start_t
            chunk_elapsed = time.perf_counter() - chunk_start_t
            avg_chunk = float(np.mean(recent_chunk))
            median_chunk = float(np.median(recent_chunk))
            max_chunk = float(np.max(recent_chunk))
            max_rolling_500 = float(np.max(rolling_scores_500)) if rolling_scores_500 else 0.0
            avg_steps = float(np.mean(recent_steps)) if recent_steps else 0.0
            avg_q = float(np.mean(recent_q_mean)) if recent_q_mean else 0.0
            avg_td = float(np.mean(recent_td_error)) if recent_td_error else 0.0
            avg_food = float(np.mean(recent_food_r)) if recent_food_r else 0.0
            avg_survival = float(np.mean(recent_survival_r)) if recent_survival_r else 0.0
            avg_safety_bonus = float(np.mean(recent_safety_bonus)) if recent_safety_bonus else 0.0
            avg_bonus = float(np.mean(recent_terminal_bonus)) if recent_terminal_bonus else 0.0
            avg_loss = float(np.mean(recent_loss)) if recent_loss else 0.0
            avg_reward = float(np.mean(recent_total_reward)) if recent_total_reward else 0.0
            chunk_q_explodes = int(np.sum(recent_q_explodes)) if recent_q_explodes else 0
            chunk_episode_ends.append(episode)
            chunk_avg_scores.append(avg_chunk)
            chunk_median_scores.append(median_chunk)
            range_start = episode - len(recent_chunk) + 1
            episode_label = f"{range_start}-{episode}/{cfg.episodes}"
            
            row = (
                f"{episode_label:<18}"
                f"{total_elapsed:>9.1f}"
                f"{chunk_elapsed:>10.1f}"
                f"{score:>6.0f}"
                f"{avg_chunk:>8.2f}"
                f"{median_chunk:>8.2f}"
                f"{max_chunk:>6.0f}"
                f"{max_rolling_500:>8.0f}"
                f"{avg_steps:>8.1f}"
                f"{agent.epsilon:>8.4f}"
                f"{avg_q:>9.2f}"
                f"{avg_td:>9.3f}"
                f"{avg_food:>9.3f}"
                f"{avg_survival:>9.3f}"
                f"{avg_safety_bonus:>9.3f}"
                f"{avg_bonus:>9.3f}"
                f"{avg_loss:>10.4f}"
                f"{avg_reward:>9.3f}"
                f"{chunk_q_explodes:>6}"
            )

            print(row)
            print()
            chunk_start_t = time.perf_counter()

        if checkpoint_every > 0 and (episode % checkpoint_every == 0):
            print()
            periodic_path = _periodic_checkpoint_path(checkpoint_seed_path, episode)
            agent.save_checkpoint(periodic_path, episode_index=episode, replay_buffer=agent.memory)
            last_periodic_checkpoint_path = periodic_path
            print(f"Saved checkpoint: {periodic_path}")

    print()

    if save_path:
        if (
            last_periodic_checkpoint_path
            and os.path.abspath(last_periodic_checkpoint_path) == os.path.abspath(save_path)
        ):
            print(f"Final checkpoint already saved at: {save_path}")
        else:
            agent.save_checkpoint(save_path, episode_index=last_completed_episode, replay_buffer=agent.memory)
            print(f"Saved checkpoint: {save_path}")
    if save_weights_path:
        agent.save(save_weights_path)
        print(f"Saved weights-only model: {save_weights_path}")

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

    if early_stop_triggered:
        print(f"Training stopped early at episode {last_completed_episode}/{cfg.episodes}")
    setattr(agent, "last_completed_episode", int(last_completed_episode))

    return agent, scores, avg10_scores


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    default_hidden_neurons = str(defaults.hidden_layers[0])
    parser = argparse.ArgumentParser(description="Offline Snake DQN training")
    parser.add_argument("--board-size", type=int, default=defaults.board_size, choices=BOARD_SIZES)
    parser.add_argument("--apples", type=int, default=defaults.apples, choices=APPLE_CHOICES)
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--epsilon-start", type=float, default=defaults.epsilon_start)
    parser.add_argument("--epsilon-min", type=float, default=defaults.epsilon_min)
    parser.add_argument("--epsilon-decay-rate", type=float, default=defaults.epsilon_decay_rate)
    parser.add_argument(
        "--hidden-layers",
        type=int,
        default=len(defaults.hidden_layers),
        help=f"Number of hidden layers ({MIN_HIDDEN_LAYERS}-{MAX_HIDDEN_LAYERS}).",
    )
    parser.add_argument(
        "--neurons",
        type=str,
        default=default_hidden_neurons,
        help="Neurons per layer: one integer for all layers, or comma-separated widths matching --hidden-layers.",
    )
    parser.add_argument(
        "--load",
        type=str,
        default="",
        help="Load weights-only model and start a fresh run (no replay/optimizer/episode resume).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume from full checkpoint and continue training with optimizer/replay/schedules intact.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Save full checkpoint at the end of training.",
    )
    parser.add_argument(
        "--save-weights",
        type=str,
        default="",
        help="Optional weights-only save path (.pt).",
    )
    parser.add_argument(
        "--save-checkpoint-every",
        type=int,
        default=0,
        help="If > 0, save a full checkpoint every N episodes; if 0, auto-uses 10% of total episodes.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=25,
        help="Print training stats every N episodes (also used for chunk average/median trend).",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=500,
        help="Stop after this many no-improvement episodes (0 disables early stopping).",
    )
    parser.add_argument(
        "--early-stop-threshold",
        type=float,
        default=0.02,
        help="Minimum rolling-average improvement to reset early-stop patience.",
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
        help="Prompt for settings in the terminal (board/apples/architecture/plot/load).",
    )
    return parser.parse_args()


def _prompt_int(
    label: str,
    default: int,
    choices: tuple[int, ...] | None = None,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
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
        if max_value is not None and value > max_value:
            print(f"Value must be <= {max_value}")
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


def _format_hidden_layers_for_prompt(hidden_layers: tuple[int, ...]) -> str:
    if len(set(hidden_layers)) == 1:
        return str(hidden_layers[0])
    return ",".join(str(width) for width in hidden_layers)


def _prompt_hidden_layers(default_layers: tuple[int, ...]) -> tuple[int, ...]:
    layer_count = _prompt_int(
        "Hidden layers",
        len(default_layers),
        min_value=MIN_HIDDEN_LAYERS,
        max_value=MAX_HIDDEN_LAYERS,
    )
    single_default = default_layers[0]
    if len(default_layers) == layer_count:
        neurons_default = _format_hidden_layers_for_prompt(default_layers)
    else:
        neurons_default = str(single_default)

    while True:
        raw = input(
            f"Neurons per layer (single int or {layer_count} comma-separated ints) [{neurons_default}]: "
        ).strip()
        raw_l = raw.lower()
        if raw == "" or raw_l == "default":
            raw = neurons_default
        try:
            return parse_hidden_layer_widths(layer_count, raw)
        except ValueError as exc:
            print(exc)


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
    episodes = _prompt_int("Episodes", default_cfg.episodes, min_value=1)
    hidden_layers = _prompt_hidden_layers(default_cfg.hidden_layers)
    epsilon_start = _prompt_float("Epsilon start", default_cfg.epsilon_start, min_value=0.0, max_value=1.0)
    epsilon_min = _prompt_float("Epsilon min", default_cfg.epsilon_min, min_value=0.0, max_value=1.0)
    epsilon_decay_rate = _prompt_float("Epsilon decay rate", default_cfg.epsilon_decay_rate, min_value=1e-6)
    if epsilon_min > epsilon_start:
        print("Epsilon min cannot exceed epsilon start; swapping values.")
        epsilon_start, epsilon_min = epsilon_min, epsilon_start
    show_plot = _prompt_bool("Show live matplotlib plot", False)
    print_every = _prompt_int("Print stats every N episodes", 25, min_value=1)
    show_final_trend_plot = _prompt_bool("Show end trend plot when live plot is off", True)

    load_raw = input("Model to load (.pt), blank for none: ").strip()
    cfg = TrainConfig(
        board_size=board_size,
        apples=apples,
        episodes=episodes,
        hidden_layers=hidden_layers,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay_rate=epsilon_decay_rate,
        state_encoding=STATE_ENCODING_INTEGER,
    )
    load_path = None if load_raw == "" or load_raw.lower() == "default" else load_raw
    return cfg, load_path, show_plot, print_every, show_final_trend_plot


def _prompt_save_after_training(agent: SnakeDQNAgent, cfg: TrainConfig) -> None:
    last_episode = int(getattr(agent, "last_completed_episode", cfg.episodes))
    default_path = _default_checkpoint_path(cfg)
    raw = input(
        f"\nSave final checkpoint to [{default_path}] (press Enter for default, type 'skip' to skip): "
    ).strip()
    raw_l = raw.lower()
    if raw_l not in {"skip", "s"}:
        final_path = default_path if raw == "" else raw
        try:
            agent.save_checkpoint(final_path, episode_index=last_episode, replay_buffer=agent.memory)
            print(f"Saved final checkpoint to: {final_path}")
        except Exception as exc:
            print(f"Final save failed: {exc}")
    else:
        print("Skipped final checkpoint save.")

    extra = input("Save checkpoint 1 to (leave blank if you don't want to save it): ").strip()
    if extra:
        try:
            agent.save_checkpoint(extra, episode_index=last_episode, replay_buffer=agent.memory)
            print(f"Saved checkpoint 1 to: {extra}")
        except Exception as exc:
            print(f"Checkpoint 1 save failed: {exc}")


def run_offline_training_cli() -> None:
    args = parse_args()
    interactive = args.interactive or len(sys.argv) == 1
    if args.resume and args.load:
        raise SystemExit("Use only one of --resume or --load.")
    if args.save_checkpoint_every < 0:
        raise SystemExit("--save-checkpoint-every must be >= 0.")
    if args.early_stop_patience < 0:
        raise SystemExit("--early-stop-patience must be >= 0.")
    if args.early_stop_threshold < 0.0:
        raise SystemExit("--early-stop-threshold must be >= 0.")
    if not (0.0 <= args.epsilon_min <= args.epsilon_start <= 1.0):
        raise SystemExit("Require 0 <= --epsilon-min <= --epsilon-start <= 1.")
    if args.epsilon_decay_rate <= 0.0:
        raise SystemExit("--epsilon-decay-rate must be > 0.")

    if interactive:
        cfg, load_path, show_plot, print_every, show_final_trend_plot = prompt_train_config()
        resume_path = None
        save_path = None
        save_weights_path = None
        save_checkpoint_every = max(1, int(cfg.episodes * 0.10))
        checkpoint_base_path = None
    else:
        resume_path = args.resume if args.resume else None
        if resume_path:
            metadata = SnakeDQNAgent.load_metadata(resume_path)
            cfg_data = metadata.get("cfg", {})
            hidden_layers_raw = metadata.get("hidden_layers", TrainConfig().hidden_layers)
            hidden_layers = tuple(int(width) for width in hidden_layers_raw)
            cfg = TrainConfig(
                board_size=int(metadata.get("board_size", args.board_size)),
                apples=int(cfg_data.get("apples", args.apples)),
                episodes=args.episodes,
                hidden_layers=hidden_layers,
                epsilon_start=args.epsilon_start,
                epsilon_min=args.epsilon_min,
                epsilon_decay_rate=args.epsilon_decay_rate,
                state_encoding=STATE_ENCODING_INTEGER,
            )
        else:
            try:
                hidden_layers = parse_hidden_layer_widths(args.hidden_layers, args.neurons)
            except ValueError as exc:
                raise SystemExit(f"Invalid hidden layer settings: {exc}")
            cfg = TrainConfig(
                board_size=args.board_size,
                apples=args.apples,
                episodes=args.episodes,
                hidden_layers=hidden_layers,
                epsilon_start=args.epsilon_start,
                epsilon_min=args.epsilon_min,
                epsilon_decay_rate=args.epsilon_decay_rate,
                state_encoding=STATE_ENCODING_INTEGER,
            )

        load_path = args.load if args.load else None
        save_path = args.save if args.save else None
        save_weights_path = args.save_weights if args.save_weights else None
        save_checkpoint_every = int(args.save_checkpoint_every)
        checkpoint_base_path = save_path or resume_path or _default_checkpoint_path(cfg)
        show_plot = not args.no_plot
        print_every = args.print_every
        show_final_trend_plot = not args.no_final_trend_plot

    agent, _, _ = train_offline(
        cfg,
        load_path=load_path,
        save_path=save_path,
        resume_path=resume_path,
        save_weights_path=save_weights_path,
        save_checkpoint_every=save_checkpoint_every,
        checkpoint_base_path=checkpoint_base_path,
        show_plot=show_plot,
        print_every=print_every,
        show_final_trend_plot=show_final_trend_plot,
        early_stop_patience=args.early_stop_patience,
        early_stop_threshold=args.early_stop_threshold,
    )

    if interactive and not save_path:
        _prompt_save_after_training(agent, cfg)


if __name__ == "__main__":
    run_offline_training_cli()
