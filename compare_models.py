"""Compare two trained Snake DQN models."""
from __future__ import annotations

import argparse
import numpy as np

try:
    from .agent import SnakeDQNAgent
    from .utils import TrainConfig, make_game, run_episode
except ImportError:
    from agent import SnakeDQNAgent
    from utils import TrainConfig, make_game, run_episode


def _cfg_from_metadata(meta: dict) -> TrainConfig:
    cfg_data = meta.get("cfg", {})
    hidden_layers_raw = meta.get("hidden_layers", cfg_data.get("hidden_layers", [256, 256, 128]))
    hidden_layers = tuple(int(width) for width in hidden_layers_raw)
    return TrainConfig(
        board_size=int(meta.get("board_size", cfg_data.get("board_size", 10))),
        apples=int(cfg_data.get("apples", 5)),
        hidden_layers=hidden_layers,
    )


def _compare_metric(name: str, value1: float, value2: float) -> None:
    if value1 > value2:
        winner = "Model 1"
    elif value2 > value1:
        winner = "Model 2"
    else:
        winner = "Tie"
    print(f"{name:<20} {value1:>15.2f} {value2:>15.2f} {winner:>10}")


def compare_models(model1_path: str, model2_path: str, num_episodes: int = 100) -> None:
    """Run both models and compare their performance."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0")

    meta1 = SnakeDQNAgent.load_metadata(model1_path)
    meta2 = SnakeDQNAgent.load_metadata(model2_path)

    cfg1 = _cfg_from_metadata(meta1)
    cfg2 = _cfg_from_metadata(meta2)
    if cfg1.board_size != cfg2.board_size:
        print(
            f"Warning: board sizes differ ({cfg1.board_size} vs {cfg2.board_size}); "
            "results are not directly comparable."
        )

    print(f"Loading model 1: {model1_path}")
    agent1 = SnakeDQNAgent(cfg1)
    agent1.load(model1_path)
    agent1.epsilon = 0.0

    print(f"Loading model 2: {model2_path}")
    agent2 = SnakeDQNAgent(cfg2)
    agent2.load(model2_path)
    agent2.epsilon = 0.0

    scores1: list[float] = []
    scores2: list[float] = []
    game1 = make_game(cfg1)
    game2 = make_game(cfg2)

    print(f"\nRunning {num_episodes} episodes for each model...")
    for episode in range(1, num_episodes + 1):
        if episode % 10 == 0 or episode == num_episodes:
            print(f"Episode {episode}/{num_episodes}", end="\r", flush=True)

        score1, _, _, _ = run_episode(agent1, cfg1, episode_index=episode, train=False, game=game1)
        score2, _, _, _ = run_episode(agent2, cfg2, episode_index=episode, train=False, game=game2)
        scores1.append(float(score1))
        scores2.append(float(score2))
    print()

    scores1_arr = np.asarray(scores1, dtype=np.float32)
    scores2_arr = np.asarray(scores2, dtype=np.float32)

    print("=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Metric':<20} {'Model 1':>15} {'Model 2':>15} {'Winner':>10}")
    print("-" * 60)
    _compare_metric("Mean score", float(scores1_arr.mean()), float(scores2_arr.mean()))
    _compare_metric("Median score", float(np.median(scores1_arr)), float(np.median(scores2_arr)))
    _compare_metric("Max score", float(scores1_arr.max()), float(scores2_arr.max()))
    _compare_metric("Min score", float(scores1_arr.min()), float(scores2_arr.min()))
    _compare_metric("Std dev", float(scores1_arr.std()), float(scores2_arr.std()))
    _compare_metric("25th percentile", float(np.percentile(scores1_arr, 25)), float(np.percentile(scores2_arr, 25)))
    _compare_metric("75th percentile", float(np.percentile(scores1_arr, 75)), float(np.percentile(scores2_arr, 75)))
    print("=" * 60)

    wins1 = int(np.sum(scores1_arr > scores2_arr))
    wins2 = int(np.sum(scores2_arr > scores1_arr))
    ties = int(np.sum(scores1_arr == scores2_arr))
    print(f"Head-to-head: Model 1 wins {wins1}, Model 2 wins {wins2}, Ties {ties}")
    print(f"Win rate: Model 1 {wins1 / num_episodes * 100:.1f}%, Model 2 {wins2 / num_episodes * 100:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two Snake DQN models")
    parser.add_argument("model1", help="Path to first model (.pt)")
    parser.add_argument("model2", help="Path to second model (.pt)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes per model")
    args = parser.parse_args()
    compare_models(args.model1, args.model2, args.episodes)


if __name__ == "__main__":
    main()
