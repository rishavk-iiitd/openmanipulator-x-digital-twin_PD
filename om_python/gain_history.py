"""Persists the Kp/Kd values used for each PD run, across sessions."""
import json
import time
from pathlib import Path

HISTORY_FILE = Path(__file__).resolve().parent.parent / "pd_gain_history.json"


def load_history(path=HISTORY_FILE):
    if Path(path).exists():
        with open(path, "r") as f:
            return json.load(f)
    return []


def append_history(kp, kd, plot_path=None, mode="simulated", path=HISTORY_FILE, **extra):
    history = load_history(path)
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kp": list(kp),
        "kd": list(kd),
        "mode": mode,
        "plot": str(plot_path) if plot_path else None,
    }
    entry.update(extra)
    history.append(entry)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    return history
