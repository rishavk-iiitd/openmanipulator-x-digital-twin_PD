"""Control panel for XYZ-target, gravity-compensated computed-torque PD
control tracking a SMOOTH, dynamically-generated reference trajectory --
a copy of ik_gravity_panel.py where the target fed to the PD controller is
no longer a fixed step input.

Stage 1: solve IK for a target end-effector XYZ (mm) -> target joint
angles.
Stage 2: trajectory.py integrates a critically-damped 2nd-order reference
ODE each tick, producing a smoothly evolving intermediate target
(position/velocity/acceleration) that gradually approaches the IK
solution, instead of handing the controller a step target immediately.
Stage 3: the same computed-torque control (mass matrix + Coriolis +
gravity compensation, rigid_body_dynamics.py) tracks that moving
reference -- either the simulated "virtual robot", or the real arm.

See ik_gravity_panel.py for the un-smoothed version and pd_lab.py /
torque_link.py / the torque-PD firmware for the hardware protocol, both
reused here unchanged.
"""
import math
import os
import threading
import time
from pathlib import Path

import dearpygui.dearpygui as dpg

from . import gain_history, inverse_kinematics, kinematics, plotting, rigid_body_dynamics
from .pd_panel import HW_DEFAULT_KD, HW_DEFAULT_KP, _finite_diff
from .rigid_body_dynamics import RigidBodyDynamics
from .torque_link import TorqueLink
from .trajectory import ReferenceTrajectory

HOME_POSE = (0.0, 0.0, 0.0, 0.0)
DT = 0.01
DURATION = 6.0

SIM_DEFAULT_KP = [0.6, 0.6, 0.6, 0.6]
SIM_DEFAULT_KD = [0.15, 0.15, 0.15, 0.15]

DEFAULT_SMOOTHING_TIME = 2.0  # seconds -- see trajectory.py

GRAVITY_SCALE_DEFAULT = 560.98  # mA per N*m -- exact, see ik_gravity_panel.py
GRAVITY_UPDATE_PERIOD = 0.05    # 20 Hz
TARGET_UPDATE_PERIOD = 0.02     # 50 Hz -- how often the moving reference is
                                 # pushed to the firmware as its "target"

HISTORY_FILE = Path(__file__).resolve().parent.parent / "smooth_gain_history.json"


def _fmt_list(values):
    return "[" + ", ".join(f"{v:g}" for v in values) + "]"


class SmoothTrajectoryPanel:
    def __init__(self, state):
        self.state = state
        self.dynamics = RigidBodyDynamics()
        self.trajectory = ReferenceTrajectory()
        self.torque_link = None
        self._running = False
        self._connecting = False
        self._recording = False
        self._gravity_running = False
        self._trajectory_running = False
        self._hw_buffer = []
        self._ik_solved = False
        self._target_angles = list(HOME_POSE)
        self._target_xyz = (0.0, 0.0, 0.0)
        self._last_q_ref = list(HOME_POSE)  # most recent trajectory setpoint, for
                                             # correlating with async hw telemetry
        self._history = gain_history.load_history(HISTORY_FILE)

    def run(self):
        dpg.create_context()
        dpg.create_viewport(title="Smooth Trajectory PD (XYZ target)", width=480, height=1020)
        dpg.setup_dearpygui()

        with dpg.window(tag="smooth_main", no_close=True, no_collapse=True):
            dpg.add_text("Smooth-trajectory gravity-compensated PD control", color=(255, 204, 102))
            dpg.add_text(
                "Stage 1: solve IK for a target XYZ. Stage 2: a critically-\n"
                "damped reference ODE (trajectory.py) smoothly ramps the PD\n"
                "target from the current pose to that solution over time,\n"
                "instead of a step input. Stage 3: computed-torque PD tracks\n"
                "that moving reference.",
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

            dpg.add_text("Target end-effector position (mm):")
            with dpg.group(horizontal=True):
                dpg.add_input_float(label="X", tag="target_x", default_value=150.0, width=100)
                dpg.add_input_float(label="Y", tag="target_y", default_value=0.0, width=100)
                dpg.add_input_float(label="Z", tag="target_z", default_value=150.0, width=100)
            dpg.add_button(label="Solve IK (preview target pose)", width=-1, callback=self._on_solve_ik)
            dpg.add_text("", tag="ik_status", color=(140, 220, 140), wrap=440)

            dpg.add_separator()
            dpg.add_input_float(
                label="Smoothing time (s)", tag="smoothing_time",
                default_value=DEFAULT_SMOOTHING_TIME, step=0.25, width=150,
            )
            dpg.add_text(
                "Roughly how long the reference target takes to glide from the\n"
                "current pose to the IK solution -- smaller is snappier (closer\n"
                "to a step input), larger is smoother/slower.",
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
                dpg.add_button(label="Run to Target", width=230, height=40, callback=self._on_run)
                dpg.add_button(label="Reset / Torque Off", width=210, height=40, callback=self._on_reset)

            dpg.add_text("", tag="run_status", color=(140, 220, 140), wrap=440)
            dpg.add_separator()

            dpg.add_text("History (double-click to reload gains + target):")
            dpg.add_listbox([], tag="history_list", num_items=10, width=-1, callback=self._on_history_select)

        self._update_gain_units_label()
        self._refresh_history_widget()
        dpg.set_primary_window("smooth_main", True)
        dpg.show_viewport()
        dpg.start_dearpygui()

        self._gravity_running = False
        self._trajectory_running = False
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
                "ik_gravity_lab.py. Angular acceleration is safety-clamped, same\n"
                "as there, so too-aggressive gains saturate rather than diverge.",
            )
        else:
            dpg.set_value(
                "gain_units",
                "Units: Kp in mA/rad, Kd in mA/(rad/s) -- same scale as pd_lab.py's\n"
                "hardware mode; gravity compensation and the smooth target are both\n"
                "added on top as separate feedforward/streamed-setpoint terms.",
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
        dpg.configure_item("history_list", items=[self._format_entry(h) for h in reversed(self._history)])

    def _format_entry(self, h):
        xyz = h.get("target_xyz", [0, 0, 0])
        st = h.get("smoothing_time", DEFAULT_SMOOTHING_TIME)
        return (
            f"[{h.get('mode', 'simulated')}] XYZ=({xyz[0]:g},{xyz[1]:g},{xyz[2]:g}) "
            f"smooth={st:g}s Kp={_fmt_list(h['kp'])} Kd={_fmt_list(h['kd'])}  ({h['timestamp']})"
        )

    def _on_history_select(self, sender, value):
        for entry in self._history:
            if self._format_entry(entry) == value:
                self._set_gains(entry["kp"], entry["kd"])
                xyz = entry.get("target_xyz", [0, 0, 0])
                dpg.set_value("target_x", xyz[0])
                dpg.set_value("target_y", xyz[1])
                dpg.set_value("target_z", xyz[2])
                dpg.set_value("smoothing_time", entry.get("smoothing_time", DEFAULT_SMOOTHING_TIME))
                return

    # ---------------- IK ----------------
    def _on_solve_ik(self):
        xyz = (dpg.get_value("target_x"), dpg.get_value("target_y"), dpg.get_value("target_z"))
        guess = self.dynamics.theta
        q, reached, err_mm = inverse_kinematics.solve(xyz, initial_guess=guess)
        self._target_angles = q
        self._target_xyz = xyz
        self._ik_solved = True

        with self.state.lock:
            self.state.ctrl_joint_angle[:] = q

        status = "reachable" if reached else "NOT fully reachable (showing closest pose)"
        dpg.set_value(
            "ik_status",
            f"IK {status}, error={err_mm:.2f} mm. "
            f"q=[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}] rad. "
            "Ghost arm previews the final pose -- during Run it'll show the moving "
            "reference instead.",
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
            # self._last_q_ref: the reference _trajectory_updater_loop most
            # recently computed and sent -- not exactly synchronous with
            # this telemetry sample (they're separate threads/rates), but
            # it's the actual setpoint in effect at roughly this instant,
            # not an approximation of it.
            self._hw_buffer.append((t, list(angles), list(velocities), list(currents), list(self._last_q_ref)))

    def _on_emergency_stop(self):
        self._recording = False
        self._gravity_running = False
        self._trajectory_running = False
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
        if not self._ik_solved:
            dpg.set_value("run_status", "Solve IK first so there's an actual target to run to.")
            return

        mode = dpg.get_value("mode_radio")
        kp, kd = self._get_gains()
        smoothing_time = dpg.get_value("smoothing_time")
        self._running = True

        if mode == "Simulate":
            dpg.set_value(
                "run_status",
                f"Running to XYZ={self._target_xyz}, smoothing={smoothing_time}s "
                f"(Kp={_fmt_list(kp)}, Kd={_fmt_list(kd)})...",
            )
            threading.Thread(target=self._simulate, args=(kp, kd, smoothing_time), daemon=True).start()
        else:
            if not self.torque_link or not self.torque_link.connected:
                dpg.set_value("run_status", "Connect to hardware first.")
                self._running = False
                return
            if not self.torque_link.ready:
                dpg.set_value("run_status", "Firmware never confirmed ready -- reconnect before running.")
                self._running = False
                return
            dpg.set_value("run_status", f"Running to XYZ={self._target_xyz} on real hardware...")
            threading.Thread(target=self._run_hardware, args=(kp, kd, smoothing_time), daemon=True).start()

    # ---------------- simulate ----------------
    def _simulate(self, kp, kd, smoothing_time):
        target = self._target_angles
        self.dynamics.reset(HOME_POSE)
        self.trajectory.reset(HOME_POSE)

        n_steps = int(DURATION / DT)
        run = {
            "t": [], "angle": [], "position": [], "velocity": [],
            "angular_velocity": [], "torque": [], "jerk": [], "end_effector": [],
            "target_angle": [],
        }
        prev_positions = kinematics.joint_positions(HOME_POSE)
        prev_ddot = [0.0, 0.0, 0.0, 0.0]

        for step in range(n_steps):
            t = step * DT
            q_ref, qdot_ref, qddot_ref = self.trajectory.step(DT, target, smoothing_time)
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

            with self.state.lock:
                self.state.receive_joint_angle[:] = theta
                self.state.ctrl_joint_angle[:] = list(q_ref)

            time.sleep(DT)

        self._finish_run(run, kp, kd, smoothing_time, source="simulated")

    # ---------------- real hardware ----------------
    def _gravity_updater_loop(self):
        while self._gravity_running:
            if self.torque_link and self.torque_link.last_state:
                _, angles, _velocities, _currents = self.torque_link.last_state
                G = rigid_body_dynamics.gravity_vector(angles)
                scale = dpg.get_value("gravity_scale")
                self.torque_link.send_gravity([g * scale for g in G])
            time.sleep(GRAVITY_UPDATE_PERIOD)

    def _trajectory_updater_loop(self, smoothing_time):
        last_time = time.time()
        while self._trajectory_running:
            now = time.time()
            dt = now - last_time
            last_time = now
            q_ref, _qdot_ref, _qddot_ref = self.trajectory.step(dt, self._target_angles, smoothing_time)
            self._last_q_ref = list(q_ref)
            if self.torque_link:
                self.torque_link.send_target(q_ref)
            with self.state.lock:
                self.state.ctrl_joint_angle[:] = list(q_ref)
            time.sleep(TARGET_UPDATE_PERIOD)

    def _run_hardware(self, kp, kd, smoothing_time):
        self._hw_buffer = []
        self._recording = True

        start = HOME_POSE
        if self.torque_link.last_state:
            start = self.torque_link.last_state[1]
        self.trajectory.reset(start)
        self._last_q_ref = list(start)

        self.torque_link.send_gains(kp, kd)

        self._gravity_running = True
        threading.Thread(target=self._gravity_updater_loop, daemon=True).start()
        self._trajectory_running = True
        threading.Thread(target=self._trajectory_updater_loop, args=(smoothing_time,), daemon=True).start()

        self.torque_link.torque_on()

        time.sleep(DURATION)

        self.torque_link.torque_off()
        self._recording = False
        self._gravity_running = False
        self._trajectory_running = False

        run = self._process_hw_buffer(self._hw_buffer)
        self._finish_run(run, kp, kd, smoothing_time, source="hardware")

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
    def _finish_run(self, run, kp, kd, smoothing_time, source):
        xyz_rounded = tuple(round(v, 1) for v in self._target_xyz)
        png_path, csv_path = plotting.save_run(
            run, kp, kd, source=source,
            title=f"Smooth PD run to XYZ={xyz_rounded}, smoothing={smoothing_time:g}s",
            target=run.get("target_angle") or None,
        )
        self._history = gain_history.append_history(
            kp, kd, png_path, mode=source, path=HISTORY_FILE,
            target_xyz=list(self._target_xyz), smoothing_time=smoothing_time,
        )

        dpg.set_value("run_status", f"Done ({source}). Saved {png_path.name} / {csv_path.name}")
        self._refresh_history_widget()
        try:
            os.startfile(str(png_path))
        except OSError:
            pass

        self._running = False
