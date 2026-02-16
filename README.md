# Snake Game (DQN with Phase Curriculum)

This project has three main ways to run Snake:
- train offline (headless, optional live matplotlib plot)
- train/watch from a Tkinter dashboard
- play Snake manually in a GUI

## Setup

From the repo root:

```bash
cd reinforcement_learning/snake_game_q_learning
python3 -m pip install torch numpy matplotlib
```

`tkinter` is also required for GUI modes (usually included with standard Python installs).

## Which file should I run?

### 1) Offline training (fastest training loop)

Run:

```bash
python3 train_offline.py
```

Running without flags opens a simple interactive console setup (press Enter to keep defaults).
You can also type `default` at the very first prompt to skip all setup instantly.
During interactive setup, pressing Enter on any individual prompt keeps that parameter's default value.
Quick-start `default` runs with plotting disabled for faster training.

Useful options:

```bash
python3 train_offline.py --board-size 20 --apples 3
python3 train_offline.py --no-plot
python3 train_offline.py --load models/snake_dqn_board_20x20.pt --save models/my_model.pt
```

This trains the model and saves a `.pt` file
(default: `models/snake_dqn_compact11_<board>x<board>.pt`).
Backend defaults now control strategy automatically (gamma, n-step returns, reward weighting,
epsilon schedule, learning-rate schedule, and target update behavior).

Plot behavior:
- top subplot: average length per 10 episodes (simple line)
- bottom subplot: histogram of episode lengths (distribution)
- console prints every 25 episodes with one table header (`Episodes`, `Last`, `Avg`, `Median`, `Max`, `Epsilon`)
- episode ends as a win when the board is fully filled by the snake

### 2) Watch training / watch model play (dashboard)

Run:

```bash
python3 training_dashboard.py
```

Inside the dashboard:
- `Train`: trains the agent and shows board + live episode graph
- `Watch`: runs the model in play mode (`epsilon=0`) so you can watch it play
- `Anim delay (ms)` slider: controls animation speed in training/watch playback
- Most learning hyperparameters are backend-controlled by curriculum (not exposed in UI)
- `Visual Theme`: customize snake/apple/grid/background/border colors using hex or color picker
- `Load`: load a saved `.pt` model
- `Save`: save current model
- `Stop`: stop current run

### 3) Manual Snake game (you control the snake)

Run:

```bash
python3 snake_gui.py
```

Controls:
- movement: arrow keys or `W A S D`
- pause/resume: `Space`

## Files at a glance

- `train_offline.py`: offline/headless training entry point
- `training_dashboard.py`: GUI dashboard for train/watch/load/save
- `snake_gui.py`: manual playable Snake
- `models/`: saved DQN checkpoints (`.pt`)
