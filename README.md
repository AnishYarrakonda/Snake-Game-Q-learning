# Snake Game (Q-Learning / DQN)

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
python3 train_offline.py --board-size 20 --apples 3 --episodes 3000 --max-steps 600
python3 train_offline.py --no-plot
python3 train_offline.py --state-encoding compact11
python3 train_offline.py --no-distance-shaping
python3 train_offline.py --load models/snake_dqn_compact11_20x20.pt --save models/my_model.pt
```

This trains the model and saves a `.pt` file
(default: `models/snake_dqn_compact11_<board>x<board>.pt`).
`compact11` is the Patrick-style state representation and is the default.

Plot behavior:
- top subplot: average length per 10 episodes (simple line)
- bottom subplot: histogram of episode lengths (distribution)
- console prints every 50 episodes with episode range, Avg50, Median50, Max50, and epsilon
- episode ends as a win when the board is fully filled by the snake

### 2) Watch training / watch model play (dashboard)

Run:

```bash
python3 training_dashboard.py
```

Or the compatibility launcher:

```bash
python3 qlearning_agent.py
```

Inside the dashboard:
- `Train`: trains the agent and shows board + live episode graph
- `Watch`: runs the model in play mode (`epsilon=0`) so you can watch it play
- `Load`: load a saved `.pt` model
- `Save`: save current model
- `Stop`: stop current run

### 3) Manual Snake game (you control the snake)

Run:

```bash
python3 snake_gui.py
```

Or the compatibility launcher:

```bash
python3 gui.py
```

Controls:
- movement: arrow keys or `W A S D`
- pause/resume: `Space`

## Files at a glance

- `train_offline.py`: offline/headless training entry point
- `training_dashboard.py`: GUI dashboard for train/watch/load/save
- `qlearning_agent.py`: launcher for dashboard
- `snake_gui.py`: manual playable Snake
- `gui.py`: launcher for manual Snake GUI
- `models/`: saved DQN checkpoints (`.pt`)
