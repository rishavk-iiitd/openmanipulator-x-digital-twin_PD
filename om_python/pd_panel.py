"""Control panel for the PD-control lab: per-joint Kp/Kd matrix + gain
history, Simulate vs. Real Hardware mode, Run/Reset/Emergency-stop.

In Real Hardware mode this drives firmware/open_manipulator_torque_pd.ino
over torque_link.py and keeps the 3D viewer's "goal" arm synced to the real
arm's live telemetry at all times (not just during a recorded run). In
Simulate mode it drives the placeholder JointDynamics model from
dynamics.py instead -- no hardware involved.
"""
import math
import os
import threading
import time

import dearpygui.dearpygui as dpg

from . import gain_history, kinematics, plotting
from .dynamics import JointDynamics
from .torque_link import TorqueLink

BASIC_POSE = (0.0, math.radians(-60.0), math.radians(20.0), math.radians(40.0))
HOME_POSE = (0.0, 0.0, 0.0, 0.0)

SIM_DT = 0.01
SIM_DURATION = 6.0
HW_DURATION = 6.0

SIM_DEFAULT_KP = [5.0, 5.0, 5.0, 5.0]
SIM_DEFAULT_KD = [0.6, 0.6, 0.6, 0.6]
# Real current-mode gains (mA/rad, mA/(rad/s)). Joints 2/3 carry real load
# against gravity -- 150 mA/rad (~150 mA at a 1 rad error) turned out to be
# below their breakaway friction/gravity torque and produced no visible
# motion at all, so these are pushed higher, closer to (but still under)
# the firmware's +-400 mA ceiling. This has no integral/gravity-compensation
# term, so expect some steady-state droop under gravity -- that's an
# inherent limit of plain PD, not a bug. Tune per-joint from here.
#
# Joint 3 (elbow) specifically needs far more than this pattern would
# suggest: at kp3=350 (in line with the other joints) it does NOT just
# droop, it sustains a ~0.4 rad stick-slip oscillation on the real arm for
# the entire run (see plots/run_20260810_171540_with_error.png) -- classic
# signature of commanded current sitting too close to breakaway friction,
# so it repeatedly sticks, builds error, jerks past it, overshoots, and
# repeats. Pushing kp3 far higher (well beyond where the other joints need
# to go) keeps commanded current clear of that stiction band, so it moves
# smoothly and settles instead of cycling -- confirmed by an actual live
# tuning session on 2026-08-10 sweeping kp3 up to ~1600-1625 and kd3 back
# down through 5-50 (see ik_gain_history.json), converging on kp3=1600,
# kd3=24 as the last, non-oscillating configuration tried (plots/
# run_20260810_170810_with_error.png: joint 3 settles in ~0.5s, current
# steady around -170 mA, well inside the ceiling). It still settles with a
# modest steady-state offset (plain PD, no integral term).
#
# These remain the PLAIN-PD numbers, for this lab and the panels that stream
# their own reference trajectory. They are not what the compensated labs
# should use -- see HW_GC_* below for why kp3=1600 stops being necessary the
# moment friction is fed forward instead of overpowered.
HW_DEFAULT_KP = [300.0, 350.0, 1600.0, 300.0]
HW_DEFAULT_KD = [20.0, 25.0, 24.0, 20.0]

# ---------------------------------------------------------------------------
# Fixed per-joint gains for the gravity-compensated PD lab (ik_gravity_panel).
# These are the values chosen on the real arm. Run
# `python -m om_python.gain_schedule` to see what any pair implies, or to
# recompute from a target response rather than guessing new numbers by hand.
#
# For a joint driven by torque, the pair that places its closed loop at a
# chosen natural frequency and damping ratio is
#     Kp = omega^2 * J * SCALE          Kd = 2 * zeta * omega * J * SCALE
# with SCALE = 560.98 mA/N*m (the XM430-W350 torque constant) and J the
# effective inertia the motor actually feels.
#
# J is the one number that had to be measured. It is NOT just M(q) from
# rigid_body_dynamics.py: each joint also carries its DYNAMIXEL's rotor
# inertia reflected through a 353.5:1 gearbox, and reflected inertia scales
# with the SQUARE of the ratio, so that term dominates. Across 216 ringing
# joint traces in plots/, a joint oscillating under a known Kp gives its
# inertia directly (omega^2 = Kp/(SCALE*J)); subtracting the link-only M_ii
# leaves a common 0.011 kg*m^2, which also matches an independent estimate
# from the motor itself (~6e-8 kg*m^2 rotor x 353.5^2 ~ 0.008).
#
#     J = mean over the workspace of diag(M(q)) + 0.011
#       = [0.0175, 0.0224, 0.0164, 0.0114] kg*m^2
#
# The pair below works out to omega ~ 7-8 rad/s (about 1.2 Hz) at zeta ~ 0.5
# to 0.7, i.e. a moderately damped response with a little overshoot. That is
# a deliberately softer setting than the omega = 12, zeta = 0.9 the formula
# above suggests: on this arm, gentler gains matter more than the last degree
# of steady-state accuracy, because a geared joint that is pushed hard near
# its target stick-slips rather than settling.
#
# A single fixed pair is enough. The achieved damping ratio only drifts
# between 0.47 (arm reaching out, where inertia is highest) and 0.69 (wrist)
# across the whole workspace -- nothing underdamped enough to ring. An
# earlier revision streamed pose-scheduled gains 20x a second to hold zeta
# exactly constant; measuring that spread showed it was not worth the extra
# moving parts.
#
# What these DON'T fix is steady-state error. The gearbox absorbs 40-200 mA
# of stiction before the output shaft moves (measured the same way, by
# differencing G(q) against the holding current), so the arm parks within
# roughly stiction/Kp of the target. Lower Kp means a wider park band --
# that is the trade being made here. Raising Kp shrinks it; so would an
# integral term, which is deliberately not present.
#
# Note on Kd and loop rate: Kd*T/J = 2 is the discrete stability limit, so
# these are safe down to about 25 Hz. ik_gravity_panel checks the rate the
# firmware reports achieving and scales Kd down if the loop is slower than
# the gains assume.
HW_GC_KP = [500.0, 700.0, 600.0, 400.0]
HW_GC_KD = [80.0, 100.0, 90.0, 70.0]


def _fmt_list(values):
    return "[" + ", ".join(f"{v:g}" for v in values) + "]"


def _finite_diff(t, values):
    """values: list[N] of length-4 lists. Returns the derivative, same shape."""
    n = len(values)
    out = [[0.0, 0.0, 0.0, 0.0] for _ in range(n)]
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            continue
        out[i] = [(values[i][k] - values[i - 1][k]) / dt for k in range(4)]
    return out


class PDLabPanel:
    def __init__(self, state):
        self.state = state
        self.dynamics = JointDynamics()
        self.torque_link = None
        self._running = False
        self._recording = False
        self._connecting = False
        self._hw_buffer = []
        self._history = gain_history.load_history()

    def run(self):
        dpg.create_context()
        dpg.create_viewport(title="PD Control Lab", width=480, height=820)
        dpg.setup_dearpygui()

        with dpg.window(tag="pd_main", no_close=True, no_collapse=True):
            dpg.add_text("Torque-based PD control -> Basic pose", color=(255, 204, 102))

            dpg.add_radio_button(
                ("Simulate", "Real Hardware"), tag="mode_radio", default_value="Simulate",
                horizontal=True, callback=self._on_mode_change,
            )

            with dpg.group(tag="hw_group", show=False):
                with dpg.group(horizontal=True):
                    dpg.add_input_text(label="COM port", tag="hw_port", default_value="COM3", width=100)
                    dpg.add_button(label="Connect", callback=self._on_connect)
                    dpg.add_button(
                        label="EMERGENCY TORQUE OFF", callback=self._on_emergency_stop,
                        width=200,
                    )
                dpg.add_text("Not connected.", tag="hw_status", color=(255, 140, 140))

            dpg.add_separator()
            dpg.add_text("Per-joint gains:")
            with dpg.table(header_row=True):
                dpg.add_table_column(label="Joint")
                dpg.add_table_column(label="Kp")
                dpg.add_table_column(label="Kd")
                for j in range(4):
                    with dpg.table_row():
                        dpg.add_text(f"Joint {j + 1}")
                        dpg.add_input_float(tag=f"kp_{j}", default_value=SIM_DEFAULT_KP[j], step=0.5, width=-1)
                        dpg.add_input_float(tag=f"kd_{j}", default_value=SIM_DEFAULT_KD[j], step=0.1, width=-1)
            dpg.add_text("", tag="gain_units", color=(160, 160, 160), wrap=440)

            with dpg.group(horizontal=True):
                dpg.add_button(label="Run to Basic Pose", width=230, height=40, callback=self._on_run)
                dpg.add_button(label="Reset / Torque Off", width=210, height=40, callback=self._on_reset)

            dpg.add_text("", tag="pd_status", color=(140, 220, 140))
            dpg.add_separator()

            dpg.add_text("Gain history (double-click to reload into the table):")
            dpg.add_listbox([], tag="history_list", num_items=10, width=-1, callback=self._on_history_select)

        self._update_gain_units_label()
        self._refresh_history_widget()

        dpg.set_primary_window("pd_main", True)
        dpg.show_viewport()
        dpg.start_dearpygui()

        if self.torque_link:
            self.torque_link.close()
        dpg.destroy_context()

    # ---------------- mode / gains ----------------
    def _on_mode_change(self, sender, value):
        defaults_kp = SIM_DEFAULT_KP if value == "Simulate" else HW_DEFAULT_KP
        defaults_kd = SIM_DEFAULT_KD if value == "Simulate" else HW_DEFAULT_KD
        self._set_gains(defaults_kp, defaults_kd)
        dpg.configure_item("hw_group", show=(value == "Real Hardware"))
        self._update_gain_units_label()

    def _update_gain_units_label(self):
        if dpg.get_value("mode_radio") == "Simulate":
            dpg.set_value(
                "gain_units",
                "Units: Kp in N*m/rad, Kd in N*m/(rad/s) -- placeholder simulated dynamics, no hardware involved.",
            )
        else:
            dpg.set_value(
                "gain_units",
                "Units: Kp in mA/rad, Kd in mA/(rad/s) -- real current commanded to the "
                "servos, hard-clamped to +-400 mA in firmware regardless of these values. "
                "No gravity compensation/integral term, so joints 2/3 will settle short of "
                "the target under gravity (steady-state droop) -- raise Kp or accept the offset.",
            )

    def _get_gains(self):
        kp = [dpg.get_value(f"kp_{j}") for j in range(4)]
        kd = [dpg.get_value(f"kd_{j}") for j in range(4)]
        return kp, kd

    def _set_gains(self, kp, kd):
        for j in range(4):
            dpg.set_value(f"kp_{j}", kp[j])
            dpg.set_value(f"kd_{j}", kd[j])

    # ---------------- history ----------------
    def _refresh_history_widget(self):
        items = [
            f"[{h.get('mode', 'simulated')}] Kp={_fmt_list(h['kp'])} Kd={_fmt_list(h['kd'])}  ({h['timestamp']})"
            for h in reversed(self._history)
        ]
        dpg.configure_item("history_list", items=items)

    def _on_history_select(self, sender, value):
        for entry in self._history:
            label = (
                f"[{entry.get('mode', 'simulated')}] Kp={_fmt_list(entry['kp'])} "
                f"Kd={_fmt_list(entry['kd'])}  ({entry['timestamp']})"
            )
            if label == value:
                self._set_gains(entry["kp"], entry["kd"])
                return

    # ---------------- hardware connection ----------------
    def _on_connect(self):
        if self._connecting:
            dpg.set_value("hw_status", "Already connecting -- give it a few seconds (OpenCR reboot + handshake).")
            return

        # Release any previous connection first -- otherwise this same
        # process tries to re-open a port it's already holding open, which
        # Windows rejects with PermissionError("Access is denied").
        if self.torque_link is not None:
            self.torque_link.close()
            self.torque_link = None

        self._connecting = True
        port = dpg.get_value("hw_port")
        dpg.set_value("hw_status", f"Opening {port} (OpenCR reboots on connect, waiting for firmware)...")
        threading.Thread(target=self._connect_thread, args=(port,), daemon=True).start()

    def _connect_thread(self, port):
        try:
            link = TorqueLink(on_state=self._on_hw_state)
            ready = link.connect(port)
        except Exception as exc:
            dpg.set_value("hw_status", f"Connect failed: {exc}")
            self.torque_link = None
            self._connecting = False
            return

        self.torque_link = link
        self._connecting = False
        if ready:
            dpg.set_value("hw_status", f"Connected to {port} -- firmware ready.")
        else:
            dpg.set_value(
                "hw_status",
                f"Connected to {port} but never saw 'torque_pd_ready' -- "
                f"boot log: {link.boot_log or '(nothing received)'}. Check firmware is flashed "
                f"and nothing else (Serial Monitor, main.py) has the port open.",
            )

    def _on_hw_state(self, t, angles, velocities, currents):
        # Keep the 3D viewer's "goal" arm synced to the real arm at all times,
        # not just while a run is being recorded.
        with self.state.lock:
            self.state.receive_joint_angle[:] = angles
        if self._recording:
            self._hw_buffer.append((t, list(angles), list(velocities), list(currents)))

    def _on_emergency_stop(self):
        self._recording = False
        self._running = False
        if self.torque_link:
            self.torque_link.torque_off()
            dpg.set_value("hw_status", "TORQUE OFF sent -- holding position.")
        dpg.set_value("pd_status", "Emergency stop.")

    # ---------------- run / reset ----------------
    def _on_reset(self):
        if self._running:
            return
        if dpg.get_value("mode_radio") == "Simulate":
            self.dynamics.reset(HOME_POSE)
            with self.state.lock:
                self.state.receive_joint_angle[:] = list(HOME_POSE)
                self.state.ctrl_joint_angle[:] = list(BASIC_POSE)
            dpg.set_value("pd_status", "Reset to home pose (simulated).")
        else:
            if self.torque_link:
                self.torque_link.torque_off()
                dpg.set_value("hw_status", "Torque off -- holding position.")
            else:
                dpg.set_value("hw_status", "Not connected.")

    def _on_run(self):
        if self._running:
            dpg.set_value("pd_status", "A run is already in progress...")
            return

        mode = dpg.get_value("mode_radio")
        kp, kd = self._get_gains()
        self._running = True

        if mode == "Simulate":
            dpg.set_value("pd_status", f"Running simulated PD (Kp={_fmt_list(kp)}, Kd={_fmt_list(kd)})...")
            threading.Thread(target=self._simulate, args=(kp, kd), daemon=True).start()
        else:
            if not self.torque_link or not self.torque_link.connected:
                dpg.set_value("pd_status", "Connect to hardware first.")
                self._running = False
                return
            if not self.torque_link.ready:
                dpg.set_value(
                    "pd_status",
                    "Firmware never confirmed ready (no 'torque_pd_ready' seen) -- "
                    "reconnect before running, otherwise this records nothing.",
                )
                self._running = False
                return
            dpg.set_value("pd_status", f"Running on real hardware (Kp={_fmt_list(kp)}, Kd={_fmt_list(kd)})...")
            threading.Thread(target=self._run_hardware, args=(kp, kd), daemon=True).start()

    # ---------------- simulate ----------------
    def _simulate(self, kp, kd):
        self.dynamics.reset(HOME_POSE)
        with self.state.lock:
            self.state.ctrl_joint_angle[:] = list(BASIC_POSE)

        n_steps = int(SIM_DURATION / SIM_DT)
        run = {
            "t": [], "angle": [], "position": [], "velocity": [],
            "angular_velocity": [], "torque": [], "jerk": [], "end_effector": [],
        }
        prev_positions = kinematics.joint_positions(HOME_POSE)

        for step in range(n_steps):
            t = step * SIM_DT
            torques, jerks = self.dynamics.step(SIM_DT, kp, kd, BASIC_POSE)

            theta = list(self.dynamics.theta)
            positions = kinematics.joint_positions(theta)
            pos_mag = [math.dist((0, 0, 0), p) for p in positions]
            prev_mag = [math.dist((0, 0, 0), p) for p in prev_positions]
            velocity = [(a - b) / SIM_DT for a, b in zip(pos_mag, prev_mag)]
            prev_positions = positions

            run["t"].append(t)
            run["angle"].append(theta)
            run["position"].append(pos_mag)
            run["velocity"].append(velocity)
            run["angular_velocity"].append(list(self.dynamics.theta_dot))
            run["torque"].append(torques)
            run["jerk"].append(jerks)
            run["end_effector"].append(kinematics.gripper_center(theta, (0, 0, 0), 0.0))

            with self.state.lock:
                self.state.receive_joint_angle[:] = theta

            time.sleep(SIM_DT)

        self._finish_run(run, kp, kd, source="simulated")

    # ---------------- real hardware ----------------
    def _run_hardware(self, kp, kd):
        self._hw_buffer = []
        self._recording = True

        with self.state.lock:
            self.state.ctrl_joint_angle[:] = list(BASIC_POSE)

        self.torque_link.send_gains(kp, kd)
        self.torque_link.send_target(BASIC_POSE)
        self.torque_link.torque_on()

        time.sleep(HW_DURATION)

        self.torque_link.torque_off()
        self._recording = False

        run = self._process_hw_buffer(self._hw_buffer)
        self._finish_run(run, kp, kd, source="hardware")

    def _process_hw_buffer(self, buffer):
        run = {"t": [], "angle": [], "position": [], "angular_velocity": [], "torque": [], "end_effector": []}
        if not buffer:
            run["velocity"] = []
            run["jerk"] = []
            return run

        t0 = buffer[0][0]
        for t, angles, velocities, currents in buffer:
            run["t"].append(t - t0)
            run["angle"].append(angles)
            run["angular_velocity"].append(velocities)
            run["torque"].append(currents)
            run["position"].append([math.dist((0, 0, 0), p) for p in kinematics.joint_positions(angles)])
            run["end_effector"].append(kinematics.gripper_center(angles, (0, 0, 0), 0.0))

        run["velocity"] = _finite_diff(run["t"], run["position"])
        angular_accel = _finite_diff(run["t"], run["angular_velocity"])
        run["jerk"] = _finite_diff(run["t"], angular_accel)
        return run

    # ---------------- shared completion path ----------------
    def _finish_run(self, run, kp, kd, source):
        png_path, csv_path = plotting.save_run(run, kp, kd, source=source, target=BASIC_POSE)
        self._history = gain_history.append_history(kp, kd, png_path, mode=source)

        dpg.set_value("pd_status", f"Done ({source}). Saved {png_path.name} / {csv_path.name}")
        self._refresh_history_widget()
        try:
            os.startfile(str(png_path))
        except OSError:
            pass

        self._running = False
