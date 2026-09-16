"""Standalone utility: re-renders a saved run's telemetry (a CSV produced by
om_python/plotting.py) as a PNG with everything the original 6-panel +
end-effector grid has, plus a new "tracking error vs time" panel -- the
per-joint angle error e_j(t) = target_j - angle_j(t), i.e. exactly the
signal the PD law (tau = Kp*e + Kd*edot) is driving to zero.

This is a separate script rather than a change to om_python/plotting.py's
save_run() (which every lab uses) -- it re-reads an already-saved CSV and
produces a new, additional image, it doesn't change what future runs save.

Usage:
    venv\\Scripts\\python plot_with_error.py plots\\run_20260706_174940.csv

    # override defaults (all optional -- the defaults below match the
    # PD-lab "Run to Basic Pose" hardware run this script was built for):
    venv\\Scripts\\python plot_with_error.py plots\\run_20260706_174940.csv ^
        --target-deg 0,-60,20,40 --kp 301.5,400.5,408,303.5 ^
        --kd 23.5,27.8,31.1,20 --source hardware --title "PD run to Basic pose"
"""
import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from om_python.plotting import save_run_with_error

# Matches pd_panel.py's BASIC_POSE -- the target this script was built to
# analyze (pd_lab.py's "Run to Basic Pose", hardware mode).
DEFAULT_TARGET_DEG = (0.0, -60.0, 20.0, 40.0)
DEFAULT_KP = (301.5, 400.5, 408.0, 303.5)
DEFAULT_KD = (23.5, 27.8, 31.1, 20.0)


def load_run(csv_path):
    """Inverse of plotting.py's CSV writer: reconstructs the same `run`
    dict of quantity -> list[time][joint] that save_run() takes."""
    run = {"t": [], "angle": [], "position": [], "velocity": [], "angular_velocity": [],
           "torque": [], "jerk": [], "end_effector": []}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            run["t"].append(float(row["t"]))
            for key in ("angle", "position", "velocity", "angular_velocity", "torque", "jerk"):
                run[key].append([float(row[f"{key}_j{j + 1}"]) for j in range(4)])
            run["end_effector"].append([
                float(row["end_effector_x"]), float(row["end_effector_y"]), float(row["end_effector_z"]),
            ])
    return run


def _parse_floats(s):
    return tuple(float(v) for v in s.split(","))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", type=Path, help="run_<timestamp>.csv saved by om_python/plotting.py")
    parser.add_argument("--target-deg", type=_parse_floats, default=DEFAULT_TARGET_DEG,
                         help="target joint angles, degrees, comma-separated (default: Basic Pose)")
    parser.add_argument("--kp", type=_parse_floats, default=DEFAULT_KP)
    parser.add_argument("--kd", type=_parse_floats, default=DEFAULT_KD)
    parser.add_argument("--source", choices=("simulated", "hardware"), default="hardware")
    parser.add_argument("--title", default="PD run to Basic pose")
    parser.add_argument("-o", "--out", type=Path, default=None,
                         help="output PNG path (default: <csv_path stem>_with_error.png next to the CSV)")
    args = parser.parse_args()

    target = tuple(math.radians(d) for d in args.target_deg)
    run = load_run(args.csv_path)
    out_path = args.out or args.csv_path.with_name(args.csv_path.stem + "_with_error.png")

    save_run_with_error(run, args.kp, args.kd, target, args.source, args.title, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
