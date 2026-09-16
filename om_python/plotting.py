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


def _finite_diff_list(t, values):
    """values: list[N] of length-4 sequences. Returns d(values)/dt, same
    shape, via one-sided backward difference (first entry left at 0) --
    same convention pd_panel.py's _finite_diff uses for hardware
    velocity/jerk."""
    n = len(values)
    out = [[0.0, 0.0, 0.0, 0.0] for _ in range(n)]
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            continue
        out[i] = [(values[i][k] - values[i - 1][k]) / dt for k in range(4)]
    return out


def _tracking_error(t, angle, target):
    """e_j(t) = target_j(t) - angle_j(t), rad. target is either a single
    length-4 sequence (fixed-point regulation -- broadcast to every t) or
    a list of length-4 sequences already aligned 1:1 with t (a moving
    reference trajectory, e.g. trajectory.py's smoothed target). Returns
    a list[N] of length-4 lists, N = min(len(t), len(target))."""
    if len(target) > 0 and not hasattr(target[0], "__len__"):
        n = len(t)
        return [[target[j] - angle[i][j] for j in range(4)] for i in range(n)]
    n = min(len(t), len(target), len(angle))
    return [[target[i][j] - angle[i][j] for j in range(4)] for i in range(n)]


def save_run(run, kp, kd, source="simulated", title="PD run to Basic pose", commanded_path=None, target=None):
    """run: dict of quantity -> list[time][joint] (4 joints), plus run["t"].
    kp/kd: per-joint (length-4) gain sequences. source: "simulated" or
    "hardware" -- controls the torque axis label (N*m estimate vs. real mA).
    title: prefix for the figure's suptitle, e.g. "PD run to XYZ target".
    commanded_path: optional list of XYZ (mm) -- the *reference* end-effector
    path being tracked (e.g. path_follow_lab.py's circle/square), overlaid on
    the actual end-effector trajectory for a visual/quantitative tracking-
    error comparison. None (the default) reproduces the previous behavior.
    target: optional per-joint tracking target, radians -- a single length-4
    sequence (fixed-point regulation) or a list of length-4 sequences
    matching run["t"] (a moving reference trajectory). When given, adds
    "tracking error vs time" (e = target - angle) and "error rate vs time"
    (de/dt) panels -- e is exactly the signal the PD law (tau = Kp*e +
    Kd*edot) is driving to zero, and edot is exactly what its Kd term
    reacts to, so together they're the most direct explanation for what
    the other panels are reacting to. None (the default) omits them."""
    PLOTS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    torque_label = "Torque (N*m, simulated)" if source == "simulated" else "Motor current (mA, measured)"
    panels = _panels(torque_label)
    t = run["t"]
    error = _tracking_error(t, run["angle"], target) if target is not None else None

    n_rows = 5 if error is not None else 4
    fig = plt.figure(figsize=(11, 19) if error is not None else (11, 16))
    suptitle_y = 0.997 if error is not None else 0.995
    fig.suptitle(f"{title} ({source}) -- Kp={_fmt_gains(kp)}, Kd={_fmt_gains(kd)}", y=suptitle_y)
    gs = fig.add_gridspec(n_rows, 2)

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
    error_dot = None
    if error is not None:
        error_dot = _finite_diff_list(t[:len(error)], error)
        _add_error_panels(fig, gs, 4, t[:len(error)], error, error_dot)

    fig.tight_layout(rect=(0, 0, 1, 0.98 if error is not None else 0.97))
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
        if error is not None:
            header += [f"error_j{j + 1}" for j in range(4)]
            header += [f"error_dot_j{j + 1}" for j in range(4)]
        # Any extra per-joint series the caller recorded (e.g. the hardware
        # run's reference angle and the feedforward the firmware reports
        # actually applying) go into the CSV too. They are not plotted, but
        # they are what lets a saved run be checked after the fact against
        # the control law that was supposed to produce it -- the analysis
        # that revealed gravity compensation had never been reaching the
        # motors had to reconstruct these; now they are simply recorded.
        extra_keys = [k for k in ("target_angle", "feedforward") if run.get(k)]
        for key in extra_keys:
            header += [f"{key}_j{j + 1}" for j in range(4)]
        writer.writerow(header)
        ee = run.get("end_effector", [])
        for i, ti in enumerate(t):
            row = [ti]
            for key, _ in panels:
                row += list(run[key][i])
            row += list(ee[i]) if i < len(ee) else [None, None, None]
            if write_commanded:
                row += list(commanded_path[i])
            if error is not None:
                if i < len(error):
                    row += list(error[i]) + list(error_dot[i])
                else:
                    row += [None] * 8
            for key in extra_keys:
                series = run[key]
                row += list(series[i]) if i < len(series) else [None] * 4
            writer.writerow(row)

    return png_path, csv_path


def save_run_with_error(run, kp, kd, target, source, title, out_path):
    """Same layout as save_run(), plus the tracking-error/error-rate panels
    -- kept for plot_with_error.py's CLI reload of CSVs saved before
    save_run() itself started plotting them. Unlike save_run(), this
    doesn't pick its own filename/timestamp or write a CSV -- callers pass
    an explicit out_path alongside a run they've already saved."""
    torque_label = "Torque (N*m, simulated)" if source == "simulated" else "Motor current (mA, measured)"
    panels = _panels(torque_label)

    fig = plt.figure(figsize=(11, 19))
    fig.suptitle(f"{title} ({source}) -- Kp={_fmt_gains(kp)}, Kd={_fmt_gains(kd)}", y=0.997)
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 0.8])

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

    _add_end_effector_panels(fig, gs, run.get("end_effector", []))
    error = _tracking_error(t, run["angle"], target)
    error_dot = _finite_diff_list(t[:len(error)], error)
    _add_error_panels(fig, gs, 4, t[:len(error)], error, error_dot)

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _add_error_panels(fig, gs, row, t, error, error_dot):
    """e_j(t) = target_j - angle_j(t) (left) and de_j/dt (right), rad and
    rad/s -- e is exactly the signal the PD law (tau = Kp*e + Kd*edot) is
    driving to zero, and edot is exactly what its Kd term reacts to."""
    ax_e = fig.add_subplot(gs[row, 0])
    for j in range(4):
        ax_e.plot(t, [r[j] for r in error], label=f"Joint {j + 1}")
    ax_e.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_e.set_title("Tracking error vs. time  --  e = target - angle")
    ax_e.set_xlabel("time (s)")
    ax_e.set_ylabel("error (rad)")
    ax_e.grid(True, alpha=0.3)
    ax_e.legend(fontsize=8)

    ax_ed = fig.add_subplot(gs[row, 1])
    for j in range(4):
        ax_ed.plot(t, [r[j] for r in error_dot], label=f"Joint {j + 1}")
    ax_ed.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_ed.set_title("Error rate vs. time  --  edot = d(error)/dt")
    ax_ed.set_xlabel("time (s)")
    ax_ed.set_ylabel("error rate (rad/s)")
    ax_ed.grid(True, alpha=0.3)
    ax_ed.legend(fontsize=8)


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
