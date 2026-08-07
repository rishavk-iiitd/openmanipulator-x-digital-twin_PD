"""Renders one PD run's recorded telemetry to a PNG (+ a CSV of the raw
numbers), using the non-interactive Agg backend so this can run from a
background thread without touching any GUI event loop (DearPyGui/GLFW)."""
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"

def _panels(torque_label):
    return [
        ("angle", "Angle (rad)"),
        ("position", "Position (mm, dist. from base)"),
        ("velocity", "Velocity (mm/s)"),
        ("angular_velocity", "Angular velocity (rad/s)"),
        ("torque", torque_label),
        ("jerk", "Jerk (rad/s^3)"),
    ]


def _fmt_gains(values):
    return "[" + ", ".join(f"{v:g}" for v in values) + "]"


def save_run(run, kp, kd, source="simulated", title="PD run to Basic pose", commanded_path=None):
    """run: dict of quantity -> list[time][joint] (4 joints), plus run["t"].
    kp/kd: per-joint (length-4) gain sequences. source: "simulated" or
    "hardware" -- controls the torque axis label (N*m estimate vs. real mA).
    title: prefix for the figure's suptitle, e.g. "PD run to XYZ target".
    commanded_path: optional list of XYZ (mm) -- the *reference* end-effector
    path being tracked (e.g. path_follow_lab.py's circle/square), overlaid on
    the actual end-effector trajectory for a visual/quantitative tracking-
    error comparison. None (the default) reproduces the previous behavior."""
    PLOTS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    torque_label = "Torque (N*m, simulated)" if source == "simulated" else "Motor current (mA, measured)"
    panels = _panels(torque_label)

    fig = plt.figure(figsize=(11, 16))
    fig.suptitle(f"{title} ({source}) -- Kp={_fmt_gains(kp)}, Kd={_fmt_gains(kd)}", y=0.995)
    gs = fig.add_gridspec(4, 2)

    t = run["t"]
    for idx, (key, panel_title) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        data = run[key]
        for j in range(4):
            ax.plot(t, [row[j] for row in data], label=f"Joint {j + 1}")
        ax.set_title(panel_title)
        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    _add_end_effector_panels(fig, gs, run.get("end_effector", []), commanded_path)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png_path = PLOTS_DIR / f"run_{stamp}.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)

    csv_path = PLOTS_DIR / f"run_{stamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["t"]
        for key, _ in panels:
            header += [f"{key}_j{j + 1}" for j in range(4)]
        header += ["end_effector_x", "end_effector_y", "end_effector_z"]
        write_commanded = bool(commanded_path) and len(commanded_path) == len(t)
        if write_commanded:
            header += ["commanded_x", "commanded_y", "commanded_z"]
        writer.writerow(header)
        ee = run.get("end_effector", [])
        for i, ti in enumerate(t):
            row = [ti]
            for key, _ in panels:
                row += list(run[key][i])
            row += list(ee[i]) if i < len(ee) else [None, None, None]
            if write_commanded:
                row += list(commanded_path[i])
            writer.writerow(row)

    return png_path, csv_path


def _add_end_effector_panels(fig, gs, end_effector, commanded=None):
    """3D XYZ trajectory of the end effector (with a real grid), plus a text
    panel next to it giving the exact final coordinate. `commanded`, if
    given, is the reference path being tracked (path_follow_lab.py) -- drawn
    as a dashed overlay for a direct visual comparison against the actual
    trajectory, with an RMS tracking-error figure in the text panel."""
    ax3d = fig.add_subplot(gs[3, 0], projection="3d")
    if end_effector:
        xs = [p[0] for p in end_effector]
        ys = [p[1] for p in end_effector]
        zs = [p[2] for p in end_effector]
        ax3d.plot(xs, ys, zs, color="tab:blue", linewidth=1.5, label="actual")
        ax3d.scatter([xs[0]], [ys[0]], [zs[0]], color="tab:green", s=40, label="start")
        ax3d.scatter([xs[-1]], [ys[-1]], [zs[-1]], color="tab:red", s=40, label="end")
    if commanded:
        cxs = [p[0] for p in commanded]
        cys = [p[1] for p in commanded]
        czs = [p[2] for p in commanded]
        ax3d.plot(cxs, cys, czs, color="tab:orange", linewidth=1.2, linestyle="--", label="commanded")
    if end_effector or commanded:
        ax3d.legend(fontsize=8)
    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")
    ax3d.set_title("End-effector trajectory")
    ax3d.grid(True)

    ax_text = fig.add_subplot(gs[3, 1])
    ax_text.axis("off")
    if end_effector:
        sx, sy, sz = end_effector[0]
        ex, ey, ez = end_effector[-1]
        text = (
            "End-effector position (mm)\n\n"
            f"Start:  X={sx:8.2f}  Y={sy:8.2f}  Z={sz:8.2f}\n"
            f"Final:  X={ex:8.2f}  Y={ey:8.2f}  Z={ez:8.2f}\n\n"
            f"Net displacement: {((ex - sx) ** 2 + (ey - sy) ** 2 + (ez - sz) ** 2) ** 0.5:.2f} mm"
        )
        if commanded:
            n = min(len(end_effector), len(commanded))
            sq_err = [
                sum((end_effector[i][k] - commanded[i][k]) ** 2 for k in range(3))
                for i in range(n)
            ]
            rms_mm = (sum(sq_err) / n) ** 0.5 if n else 0.0
            text += (
                f"\n\nPath tracking RMS error: {rms_mm:.2f} mm\n"
                f"(index-aligned actual vs. commanded, n={n})"
            )
    else:
        text = "No end-effector data recorded."
    ax_text.text(0.02, 0.7, text, fontsize=12, family="monospace", va="top", transform=ax_text.transAxes)
