"""Control panel for feedback-linearization CIRCULAR end-effector trajectory
tracking -- a trimmed-down path_follow_panel.py with the shape/square
machinery removed: the goal always traces om_python/paths.py's circle_path(),
never a fixed point.

Same computed-torque (feedback-linearization) controller as the other labs
-- rigid_body_dynamics.py's M(q)/C(q,qdot)/G(q) cancellation -- fed a
continuously moving reference instead of a static one. Per control tick:
  1. Sample the circle at the current time -> an instantaneous Cartesian
     point (paths.circle_path(t, center, radius, plane, period)).
  2. Solve IK for that point (warm-started from the previous tick's
     solution, so it converges in a handful of iterations and doesn't jump).
  3. Run that joint-space point through trajectory.py's critically-damped
     reference filter (same as smooth_trajectory_panel.py) -- this is what
     keeps the reference physically trackable instead of jumping every tick.
  4. Hand the filtered (q_ref, qdot_ref, qddot_ref) to the same
     RigidBodyDynamics.step() used everywhere else -- computed torque:
         tau = M(q)*qddot_ref + Kp*e + Kd*edot + C(q,qdot)*qdot + G(q)
     which cancels the arm's own nonlinear/gravity dynamics exactly, so
     Kp/Kd only have to correct the (now-linear, decoupled) tracking error.

See ik_gravity_panel.py / smooth_trajectory_panel.py for the point-target
versions, path_follow_panel.py for the general (circle-or-square,
shape-selectable) version this was trimmed from, and pd_lab.py /
torque_link.py / the torque-PD firmware for the hardware protocol, all
reused here unchanged.
"""
import math
import os
import threading
import time
from pathlib import Path

import dearpygui.dearpygui as dpg

from . import gain_history, inverse_kinematics, kinematics, paths, plotting, rigid_body_dynamics
from .pd_panel import HW_DEFAULT_KD, HW_DEFAULT_KP, _finite_diff
from .rigid_body_dynamics import RigidBodyDynamics
from .torque_link import TorqueLink
from .trajectory import ReferenceTrajectory

HOME_POSE = (0.0, 0.0, 0.0, 0.0)
DT = 0.01

SIM_DEFAULT_KP = [0.6, 0.6, 0.6, 0.6]
SIM_DEFAULT_KD = [0.15, 0.15, 0.15, 0.15]

DEFAULT_SMOOTHING_TIME = 0.4  # seconds -- see path_follow_panel.py's comment:
                               # measured steady-state tracking RMS on a
                               # 60mm-radius circle was ~28mm at
                               # smoothing_time=1.0, ~11mm at 0.4, ~2mm at
                               # 0.1 (PROJECT_REPORT.md). Smaller tracks
                               # tighter but raises jerk and can saturate
                               # MAX_ANGULAR_ACCEL; 0.4s is a reasonable
                               # middle ground out of the box.
DEFAULT_RADIUS_MM = 60.0  # mm -- verified fully reachable at the default
                           # center across multiple loops (PROJECT_REPORT.md)
DEFAULT_PERIOD = paths.DEFAULT_PERIOD["circle"]  # 6.0s / loop
DEFAULT_LOOPS = 2

GRAVITY_SCALE_DEFAULT = 560.98  # mA per N*m -- exact, see ik_gravity_panel.py
GRAVITY_UPDATE_PERIOD = 0.05    # 20 Hz
TARGET_UPDATE_PERIOD = 0.02     # 50 Hz -- how often the moving reference is
                                 # pushed to the firmware as its "target"

HISTORY_FILE = Path(__file__).resolve().parent.parent / "circular_gain_history.json"


def _fmt_list(values):
    return "[" + ", ".join(f"{v:g}" for v in values) + "]"


class CircularTrackingPanel:
    def __init__(self, state):
        self.state = state
        self.dynamics = RigidBodyDynamics()
        self.trajectory = ReferenceTrajectory()
        self.torque_link = None
        self._running = False
        self._connecting = False
        self._recording = False
        self._gravity_running = False
        self._path_running = False
        self._hw_buffer = []
        self._commanded_buffer = []
        self._q_goal_guess = list(HOME_POSE)
        self._last_q_ref = list(HOME_POSE)  # most recent trajectory setpoint, for
                                             # correlating with async hw telemetry
        self._history = gain_history.load_history(HISTORY_FILE)

    def run(self):
        dpg.create_context()
        dpg.create_viewport(title="Circular Trajectory Tracking (feedback linearization)", width=480, height=1020)
        dpg.setup_dearpygui()

        with dpg.window(tag="circular_main", no_close=True, no_collapse=True):
            dpg.add_text("Circular end-effector trajectory tracking", color=(255, 204, 102))
            dpg.add_text(
                "Same computed-torque controller as the other labs (M(q),\n"
                "C(q,qdot), G(q) cancellation), but the goal continuously\n"
                "traces a circle instead of sitting at one fixed point. IK is\n"
                "re-solved every tick (warm-started from the previous\n"
                "solution); the moving reference then passes through the same\n"
                "critically-damped filter as smooth_trajectory_lab.py before\n"
                "reaching the controller.",
                color=(160, 160, 160), wrap=440,
            )

            dpg.add_radio_button(
                ("Simulate", "Real Hardware"), tag="mode_radio", default_value="Simulate",
                horizontal=True, callback=self._on_mode_change,
            )

            with dpg.group(tag="hw_group", show=False):
                with dpg.group(horizontal=True):
                    dpg.add_input_text(label="COM port", tag="hw_port", default_value="COM3", width=100)
                    dpg.add_button(label="Connect", callback=self._on_connect)
                    dpg.add_button(label="EMERGENCY TORQUE OFF", callback=self._on_emergency_stop, width=200)
                dpg.add_text("Not connected.", tag="hw_status", color=(255, 140, 140), wrap=440)
                dpg.add_input_float(
                    label="Gravity comp scale (mA per N*m)", tag="gravity_scale",
                    default_value=GRAVITY_SCALE_DEFAULT, step=25.0, width=150,
                )

            dpg.add_separator()
            dpg.add_text("Circle:")
            dpg.add_combo(
                ("xz", "xy", "yz"), tag="plane_combo", default_value="xz", width=100,
                label="Plane (xz = vertical, forward-facing; xy = horizontal; yz = vertical, side-on)",
            )
            dpg.add_text("Circle center (mm):")
            with dpg.group(horizontal=True):
                dpg.add_input_float(label="Cx", tag="center_x", default_value=150.0, width=100)
                dpg.add_input_float(label="Cy", tag="center_y", default_value=0.0, width=100)
                dpg.add_input_float(label="Cz", tag="center_z", default_value=150.0, width=100)
            with dpg.group(horizontal=True):
                dpg.add_input_float(label="Radius (mm)", tag="path_radius", default_value=DEFAULT_RADIUS_MM, width=100)
                dpg.add_input_float(label="Period (s / loop)", tag="path_period", default_value=DEFAULT_PERIOD, width=100)
                dpg.add_input_int(label="Loops", tag="path_loops", default_value=DEFAULT_LOOPS, min_value=1, width=80)
            dpg.add_button(label="Preview circle start pose", width=-1, callback=self._on_preview)
            dpg.add_text("", tag="ik_status", color=(140, 220, 140), wrap=440)

            dpg.add_separator()
            dpg.add_input_float(
                label="Smoothing time (s)", tag="smoothing_time",
                default_value=DEFAULT_SMOOTHING_TIME, step=0.1, width=150,
            )
            dpg.add_text(
                "How much the reference lags the circle -- larger tracks\n"
                "smoother but cuts the radius short if it's not small\n"
                "relative to the loop period; smaller tracks tighter but\n"
                "raises jerk.",
                color=(160, 160, 160), wrap=440,
            )

            dpg.add_separator()
            dpg.add_text("Per-joint gains:")
            with dpg.table(header_row=True):
                dpg.add_table_column(label="Joint")
                dpg.add_table_column(label="Kp")
                dpg.add_table_column(label="Kd")
                for j in range(4):
                    with dpg.table_row():
                        dpg.add_text(f"Joint {j + 1}")
                        dpg.add_input_float(tag=f"kp_{j}", default_value=SIM_DEFAULT_KP[j], step=0.05, width=-1)
                        dpg.add_input_float(tag=f"kd_{j}", default_value=SIM_DEFAULT_KD[j], step=0.02, width=-1)
            dpg.add_text("", tag="gain_units", color=(160, 160, 160), wrap=440)

            with dpg.group(horizontal=True):
                dpg.add_button(label="Run Circle", width=230, height=40, callback=self._on_run)
                dpg.add_button(label="Reset / Torque Off", width=210, height=40, callback=self._on_reset)

            dpg.add_text("", tag="run_status", color=(140, 220, 140), wrap=440)
            dpg.add_separator()

            dpg.add_text("History (double-click to reload gains + circle):")
            dpg.add_listbox([], tag="history_list", num_items=10, width=-1, callback=self._on_history_select)

        self._update_gain_units_label()
        self._refresh_history_widget()
        dpg.set_primary_window("circular_main", True)
        dpg.show_viewport()
        dpg.start_dearpygui()

        self._gravity_running = False
        self._path_running = False
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
                "Units: Kp in N*m/rad, Kd in N*m/(rad/s) -- same model/scale as\n"
                "ik_gravity_lab.py/smooth_trajectory_lab.py. Angular acceleration\n"
                "is safety-clamped, so too-aggressive gains saturate rather than diverge.",
            )
        else:
            dpg.set_value(
                "gain_units",
                "Units: Kp in mA/rad, Kd in mA/(rad/s) -- same scale as pd_lab.py's\n"
                "hardware mode; gravity compensation and the moving reference are\n"
                "both added on top as separate feedforward/streamed-setpoint terms.",
            )

    def _get_gains(self):
        kp = [dpg.get_value(f"kp_{j}") for j in range(4)]
        kd = [dpg.get_value(f"kd_{j}") for j in range(4)]
        return kp, kd

    def _set_gains(self, kp, kd):
        for j in range(4):
            dpg.set_value(f"kp_{j}", kp[j])
            dpg.set_value(f"kd_{j}", kd[j])

    def _get_circle_params(self):
        plane = dpg.get_value("plane_combo")
        center = (dpg.get_value("center_x"), dpg.get_value("center_y"), dpg.get_value("center_z"))
        radius = dpg.get_value("path_radius")
        period = dpg.get_value("path_period")
        loops = dpg.get_value("path_loops")
        return plane, center, radius, period, loops

    def _set_circle_params(self, entry):
        dpg.set_value("plane_combo", entry.get("plane", "xz"))
        center = entry.get("center", [150.0, 0.0, 150.0])
        dpg.set_value("center_x", center[0])
        dpg.set_value("center_y", center[1])
        dpg.set_value("center_z", center[2])
        dpg.set_value("path_radius", entry.get("radius", DEFAULT_RADIUS_MM))
        dpg.set_value("path_period", entry.get("period", DEFAULT_PERIOD))
        dpg.set_value("path_loops", entry.get("loops", DEFAULT_LOOPS))
        dpg.set_value("smoothing_time", entry.get("smoothing_time", DEFAULT_SMOOTHING_TIME))

    # ---------------- history ----------------
    def _refresh_history_widget(self):
        dpg.configure_item("history_list", items=[self._format_entry(h) for h in reversed(self._history)])

    def _format_entry(self, h):
        center = h.get("center", [0, 0, 0])
        return (
            f"[{h.get('mode', 'simulated')}] {h.get('plane', 'xz')} plane "
            f"center=({center[0]:g},{center[1]:g},{center[2]:g}) radius={h.get('radius', 0):g}mm "
            f"T={h.get('period', 0):g}s x{h.get('loops', 1)} "
            f"Kp={_fmt_list(h['kp'])} Kd={_fmt_list(h['kd'])}  ({h['timestamp']})"
        )

    def _on_history_select(self, sender, value):
        for entry in self._history:
            if self._format_entry(entry) == value:
                self._set_gains(entry["kp"], entry["kd"])
                self._set_circle_params(entry)
                return

    # ---------------- preview ----------------
    def _on_preview(self):
        plane, center, radius, period, _loops = self._get_circle_params()
        xyz = paths.circle_path(0.0, center, radius, plane, period)
        q, reached, err_mm = inverse_kinematics.solve(xyz, initial_guess=self.dynamics.theta)

        with self.state.lock:
            self.state.ctrl_joint_angle[:] = q

        status = "reachable" if reached else "NOT fully reachable (showing closest pose)"
        dpg.set_value(
            "ik_status",
            f"Circle start {status}, error={err_mm:.2f} mm at XYZ={tuple(round(v, 1) for v in xyz)}. "
            "Ghost arm previews this pose -- during Run it'll track the moving reference instead.",
        )

    # ---------------- hardware connection ----------------
    def _on_connect(self):
        if self._connecting:
            dpg.set_value("hw_status", "Already connecting -- give it a few seconds.")
            return
        if self.torque_link is not None:
            self.torque_link.close()
            self.torque_link = None

        self._connecting = True
        port = dpg.get_value("hw_port")
        dpg.set_value("hw_status", f"Opening {port}, waiting for firmware...")
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
                f"Connected to {port} but never saw proof of life -- "
                f"boot log: {link.boot_log or '(nothing received)'}.",
            )

    def _on_hw_state(self, t, angles, velocities, currents):
        with self.state.lock:
            self.state.receive_joint_angle[:] = angles
        if self._recording:
            # self._last_q_ref: the reference _circle_updater_loop most
            # recently computed and sent -- not exactly synchronous with
            # this telemetry sample (they're separate threads/rates), but
            # it's the actual setpoint in effect at roughly this instant,
            # not an approximation of it.
            self._hw_buffer.append((t, list(angles), list(velocities), list(currents), list(self._last_q_ref)))

    def _on_emergency_stop(self):
        self._recording = False
        self._gravity_running = False
        self._path_running = False
        self._running = False
        if self.torque_link:
            self.torque_link.torque_off()
            dpg.set_value("hw_status", "TORQUE OFF sent -- holding position.")
        dpg.set_value("run_status", "Emergency stop.")

    # ---------------- run / reset ----------------
    def _on_reset(self):
        if self._running:
            return
        if dpg.get_value("mode_radio") == "Simulate":
            self.dynamics.reset(HOME_POSE)
            with self.state.lock:
                self.state.receive_joint_angle[:] = list(HOME_POSE)
            dpg.set_value("run_status", "Reset to home pose (simulated).")
        else:
            if self.torque_link:
                self.torque_link.torque_off()
                dpg.set_value("hw_status", "Torque off -- holding position.")
            else:
                dpg.set_value("hw_status", "Not connected.")

    def _on_run(self):
        if self._running:
            dpg.set_value("run_status", "A run is already in progress...")
            return

        mode = dpg.get_value("mode_radio")
        kp, kd = self._get_gains()
        smoothing_time = dpg.get_value("smoothing_time")
        plane, center, radius, period, loops = self._get_circle_params()
        self._running = True

        if mode == "Simulate":
            dpg.set_value(
                "run_status",
                f"Running circle/{plane}, radius={radius:g}mm, T={period:g}s x{loops} "
                f"(Kp={_fmt_list(kp)}, Kd={_fmt_list(kd)})...",
            )
            threading.Thread(
                target=self._simulate,
                args=(kp, kd, smoothing_time, plane, center, radius, period, loops),
                daemon=True,
            ).start()
        else:
            if not self.torque_link or not self.torque_link.connected:
                dpg.set_value("run_status", "Connect to hardware first.")
                self._running = False
                return
            if not self.torque_link.ready:
                dpg.set_value("run_status", "Firmware never confirmed ready -- reconnect before running.")
                self._running = False
                return
            dpg.set_value("run_status", f"Running circle/{plane} on real hardware...")
            threading.Thread(
                target=self._run_hardware,
                args=(kp, kd, smoothing_time, plane, center, radius, period, loops),
                daemon=True,
            ).start()

    # ---------------- simulate ----------------
    def _simulate(self, kp, kd, smoothing_time, plane, center, radius, period, loops):
        self.dynamics.reset(HOME_POSE)
        self.trajectory.reset(HOME_POSE)
        self._q_goal_guess = list(HOME_POSE)

        duration = period * loops
        n_steps = int(duration / DT)
        run = {
            "t": [], "angle": [], "position": [], "velocity": [],
            "angular_velocity": [], "torque": [], "jerk": [], "end_effector": [],
            "target_angle": [],
        }
        commanded = []
        prev_positions = kinematics.joint_positions(HOME_POSE)
        prev_ddot = [0.0, 0.0, 0.0, 0.0]

        for step in range(n_steps):
            t = step * DT
            xyz = paths.circle_path(t, center, radius, plane, period)
            q_goal, _reached, _err_mm = inverse_kinematics.solve(xyz, initial_guess=self._q_goal_guess)
            self._q_goal_guess = q_goal

            q_ref, qdot_ref, qddot_ref = self.trajectory.step(DT, q_goal, smoothing_time)
            tau, _G, _Cqd = self.dynamics.step(DT, kp, kd, q_ref, qdot_ref, qddot_ref)

            theta = list(self.dynamics.theta)
            ddot = list(self.dynamics.theta_ddot)
            jerk = [(ddot[i] - prev_ddot[i]) / DT for i in range(4)]
            prev_ddot = ddot

            positions = kinematics.joint_positions(theta)
            pos_mag = [math.dist((0, 0, 0), p) for p in positions]
            prev_mag = [math.dist((0, 0, 0), p) for p in prev_positions]
            velocity = [(a - b) / DT for a, b in zip(pos_mag, prev_mag)]
            prev_positions = positions

            run["t"].append(t)
            run["angle"].append(theta)
            run["position"].append(pos_mag)
            run["velocity"].append(velocity)
            run["angular_velocity"].append(list(self.dynamics.theta_dot))
            run["torque"].append(list(tau))
            run["jerk"].append(jerk)
            run["end_effector"].append(kinematics.gripper_center(theta, (0, 0, 0), 0.0))
            run["target_angle"].append(list(q_ref))
            commanded.append(xyz)

            with self.state.lock:
                self.state.receive_joint_angle[:] = theta
                self.state.ctrl_joint_angle[:] = list(q_ref)

            time.sleep(DT)

        self._finish_run(run, kp, kd, smoothing_time, plane, center, radius, period, loops, commanded, source="simulated")

    # ---------------- real hardware ----------------
    def _gravity_updater_loop(self):
        while self._gravity_running:
            if self.torque_link and self.torque_link.last_state:
                _, angles, _velocities, _currents = self.torque_link.last_state
                G = rigid_body_dynamics.gravity_vector(angles)
                scale = dpg.get_value("gravity_scale")
                self.torque_link.send_gravity([g * scale for g in G])
            time.sleep(GRAVITY_UPDATE_PERIOD)

    def _circle_updater_loop(self, plane, center, radius, period, smoothing_time):
        start_wall = time.time()
        last_time = start_wall
        while self._path_running:
            now = time.time()
            dt = now - last_time
            last_time = now
            t = now - start_wall

            xyz = paths.circle_path(t, center, radius, plane, period)
            q_goal, _reached, _err_mm = inverse_kinematics.solve(xyz, initial_guess=self._q_goal_guess)
            self._q_goal_guess = q_goal
            q_ref, _qdot_ref, _qddot_ref = self.trajectory.step(dt, q_goal, smoothing_time)
            self._last_q_ref = list(q_ref)

            if self.torque_link:
                self.torque_link.send_target(q_ref)
            with self.state.lock:
                self.state.ctrl_joint_angle[:] = list(q_ref)
            self._commanded_buffer.append((t, xyz))
            time.sleep(TARGET_UPDATE_PERIOD)

    def _run_hardware(self, kp, kd, smoothing_time, plane, center, radius, period, loops):
        self._hw_buffer = []
        self._commanded_buffer = []
        self._recording = True

        start = HOME_POSE
        if self.torque_link.last_state:
            start = self.torque_link.last_state[1]
        self.trajectory.reset(start)
        self._q_goal_guess = list(start)
        self._last_q_ref = list(start)

        self.torque_link.send_gains(kp, kd)

        self._gravity_running = True
        threading.Thread(target=self._gravity_updater_loop, daemon=True).start()
        self._path_running = True
        threading.Thread(
            target=self._circle_updater_loop, args=(plane, center, radius, period, smoothing_time), daemon=True,
        ).start()

        self.torque_link.torque_on()

        time.sleep(period * loops)

        self.torque_link.torque_off()
        self._recording = False
        self._gravity_running = False
        self._path_running = False

        run = self._process_hw_buffer(self._hw_buffer)
        commanded = [xyz for _t, xyz in self._commanded_buffer]
        self._finish_run(run, kp, kd, smoothing_time, plane, center, radius, period, loops, commanded, source="hardware")

    def _process_hw_buffer(self, buffer):
        run = {
            "t": [], "angle": [], "position": [], "angular_velocity": [], "torque": [],
            "end_effector": [], "target_angle": [],
        }
        if not buffer:
            run["velocity"] = []
            run["jerk"] = []
            return run

        t0 = buffer[0][0]
        for t, angles, velocities, currents, q_ref in buffer:
            run["t"].append(t - t0)
            run["angle"].append(angles)
            run["angular_velocity"].append(velocities)
            run["torque"].append(currents)
            run["position"].append([math.dist((0, 0, 0), p) for p in kinematics.joint_positions(angles)])
            run["end_effector"].append(kinematics.gripper_center(angles, (0, 0, 0), 0.0))
            run["target_angle"].append(q_ref)

        run["velocity"] = _finite_diff(run["t"], run["position"])
        angular_accel = _finite_diff(run["t"], run["angular_velocity"])
        run["jerk"] = _finite_diff(run["t"], angular_accel)
        return run

    # ---------------- shared completion path ----------------
    def _finish_run(self, run, kp, kd, smoothing_time, plane, center, radius, period, loops, commanded, source):
        png_path, csv_path = plotting.save_run(
            run, kp, kd, source=source,
            title=f"Circular tracking ({plane} plane, radius={radius:g}mm, T={period:g}s x{loops})",
            commanded_path=commanded,
            target=run.get("target_angle") or None,
        )
        self._history = gain_history.append_history(
            kp, kd, png_path, mode=source, path=HISTORY_FILE,
            plane=plane, center=list(center), radius=radius,
            period=period, loops=loops, smoothing_time=smoothing_time,
        )

        dpg.set_value("run_status", f"Done ({source}). Saved {png_path.name} / {csv_path.name}")
        self._refresh_history_widget()
        try:
            os.startfile(str(png_path))
        except OSError:
            pass

        self._running = False
