# Backward-compatible launcher that opens the training dashboard by default.
from __future__ import annotations

try:
    from .training_dashboard import run_training_dashboard
except ImportError:
    from training_dashboard import run_training_dashboard


if __name__ == "__main__":
    run_training_dashboard()
