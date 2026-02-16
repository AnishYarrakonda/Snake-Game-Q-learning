# Snake Game (Q-Learning / DQN)

This project has three main ways to run Snake:
- train offline (headless, optional live matplotlib plot)
- train/watch from a Tkinter dashboard
- play Snake manually in a GUI

## Setup

From the repo root:

```bash
cd reinforcement_learning/snake_game_q_learning
python -m pip install torch numpy matplotlib
```

`tkinter` is also required for GUI modes (usually included with standard Python installs).

## Which file should I run?

### 1) Offline training (fastest training loop)

Run:

```bash
python train_offline.py
```

Useful options:

```bash
python train_offline.py --board-size 20 --apples 3 --episodes 3000 --max-steps 1200
python train_offline.py --no-plot
python train_offline.py --load models/snake_dqn_20x20.pt --save models/my_model.pt
```

This trains the model and saves a `.pt` file (default: `models/snake_dqn_<board>x<board>.pt`).

### 2) Watch training / watch model play (dashboard)

Run:

```bash
python training_dashboard.py
```

Or the compatibility launcher:

```bash
python qlearning_agent.py
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
python snake_gui.py
```

Or the compatibility launcher:

```bash
python gui.py
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
