# Backward-compatible launcher for the manual Snake player GUI.
from __future__ import annotations

try:
    from .snake_gui import run_player_gui
except ImportError:
    from snake_gui import run_player_gui


if __name__ == "__main__":
    run_player_gui()
