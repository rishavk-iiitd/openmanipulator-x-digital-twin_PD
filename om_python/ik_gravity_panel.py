"""Control panel for XYZ-target, gravity-compensated computed-torque PD
control.

Stage 1: solve IK for a target end-effector XYZ (mm) -> target joint
angles, previewed as the ghost arm in the 3D view.
Stage 2: run a full computed-torque controller (mass matrix + Coriolis +
gravity compensation, rigid_body_dynamics.py) to drive the arm smoothly to
that joint configuration -- either the simulated "virtual robot", or the
real arm.

Real Hardware mode reuses firmware/open_manipulator_torque_pd.ino and
torque_link.py from pd_lab.py. That firmware's fast (100 Hz) position/
velocity PD loop runs locally on the OpenCR -- serial round-trip latency is
too slow for a PC-side loop to do that part. Gravity compensation is
different: G(q) only depends on the arm's current pose, which changes
slowly, so it's computed here on the PC (using the same M(q)/C(q,qdot)/G(q)
dynamics -- porting that to the microcontroller isn't practical) and pushed
over periodically as a "gravity" feedforward the firmware just adds to its
own PD output.
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

HOME_POSE = (0.0, 0.0, 0.0, 0.0)
DT = 0.01
DURATION = 6.0

SIM_DEFAULT_KP = [0.6, 0.6, 0.6, 0.6]
SIM_DEFAULT_KD = [0.15, 0.15, 0.15, 0.15]

# mA per N*m -- converts the SI-unit gravity torque G(q) into the current
# space the hardware Kp/Kd already operate in (see pd_lab.py/torque_link.py).
# Exact, not a guess: the inverse of the XM430-W350's torque constant Kt,
# derived from ROBOTIS's own published stall spec (emanual.robotis.com/
# docs/en/dxl/x/xm430-w350) at their recommended 12.0V supply --
# Stall Torque 4.1 N*m at Stall Current 2.3 A, so 2300 mA / 4.1 N*m =
# 560.98 mA/N*m. If your arm is actually powered at 11.1V (common 3S LiPo)
# rather than 12.0V, the datasheet's 11.1V row gives 2100/3.8 = 552.63
# instead -- close enough that it rarely matters, but this field is still
# a live UI input if you want to dial it in per your actual supply.
GRAVITY_SCALE_DEFAULT = 560.98
GRAVITY_UPDATE_PERIOD = 0.05  # 20 Hz -- G(q) changes slowly, no need for 100 Hz

HISTORY_FILE = Path(__file__).resolve().parent.parent / "ik_gain_history.json"


def _fmt_list(values):
    return "[" + ", ".join(f"{v:g}" for v in values) + "]"


class IKGravityPanel:
    def __init__(self, state):
        self.state = state
        self.dynamics = RigidBodyDynamics()
        self.torque_link = None
        self._running = False
        self._connecting = False
        self._recording = False
        self._gravity_running = False
        self._hw_buffer = []
        self._ik_solved = False
        self._target_angles = list(HOME_POSE)
        self._target_xyz = (0.0, 0.0, 0.0)
        self._history = gain_history.load_history(HISTORY_FILE)

    def run(self):
        dpg.create_context()
        dpg.create_viewport(title="Gravity-Compensated PD (XYZ target)", width=480, height=980)
        dpg.setup_dearpygui()

        with dpg.window(tag="ik_main", no_close=True, no_collapse=True):
            dpg.add_text("Gravity/Coriolis/Inertia-compensated PD control", color=(255, 204, 102))
            dpg.add_text(
                "Stage 1: solve IK for a target XYZ. Stage 2: computed-torque\n"
                "PD (M(q), C(q,qdot), G(q)) drives the arm there.",
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
        dpg.set_primary_window("ik_main", True)
        dpg.show_viewport()
        dpg.start_dearpygui()

        self._gravity_running = False
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
                "Units: Kp in N*m/rad, Kd in N*m/(rad/s). This model's inertia is much\n"
                "smaller than the simplified PD lab's -- gains an order of magnitude\n"
                "smaller than there are already enough; too-high gains saturate at a\n"
                "safety-clamped angular acceleration rather than diverge.",
            )
        else:
            dpg.set_value(
                "gain_units",
                "Units: Kp in mA/rad, Kd in mA/(rad/s) -- same scale as pd_lab.py's\n"
                "hardware mode, since it's the same firmware PD loop. Gravity\n"
                "compensation is added on top as a separate feedforward term.",
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
        return (
            f"[{h.get('mode', 'simulated')}] XYZ=({xyz[0]:g},{xyz[1]:g},{xyz[2]:g}) "
            f"Kp={_fmt_list(h['kp'])} Kd={_fmt_list(h['kd'])}  ({h['timestamp']})"
        )

    def _on_history_select(self, sender, value):
        for entry in self._history:
            if self._format_entry(entry) == value:
                self._set_gains(entry["kp"], entry["kd"])
                xyz = entry.get("target_xyz", [0, 0, 0])
                dpg.set_value("target_x", xyz[0])
                dpg.set_value("target_y", xyz[1])
                dpg.set_value("target_z", xyz[2])
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
            "Ghost arm in 3D view shows this pose -- click Run to move there.",
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
            self._hw_buffer.append((t, list(angles), list(velocities), list(currents)))

    def _on_emergency_stop(self):
        self._recording = False
        self._gravity_running = False
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
        self._running = True

        if mode == "Simulate":
            dpg.set_value("run_status", f"Running to XYZ={self._target_xyz} (Kp={_fmt_list(kp)}, Kd={_fmt_list(kd)})...")
            threading.Thread(target=self._simulate, args=(kp, kd), daemon=True).start()
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
            threading.Thread(target=self._run_hardware, args=(kp, kd), daemon=True).start()

    # ---------------- simulate ----------------
    def _simulate(self, kp, kd):
        target = self._target_angles
        self.dynamics.reset(HOME_POSE)

        n_steps = int(DURATION / DT)
        run = {
            "t": [], "angle": [], "position": [], "velocity": [],
            "angular_velocity": [], "torque": [], "jerk": [], "end_effector": [],
        }
        prev_positions = kinematics.joint_positions(HOME_POSE)
        prev_ddot = [0.0, 0.0, 0.0, 0.0]

        for step in range(n_steps):
            t = step * DT
            tau, _G, _Cqd = self.dynamics.step(DT, kp, kd, target)

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

            with self.state.lock:
                self.state.receive_joint_angle[:] = theta

            time.sleep(DT)

        self._finish_run(run, kp, kd, source="simulated")

    # ---------------- real hardware ----------------
    def _gravity_updater_loop(self):
        while self._gravity_running:
            if self.torque_link and self.torque_link.last_state:
                _, angles, _velocities, _currents = self.torque_link.last_state
                G = rigid_body_dynamics.gravity_vector(angles)
                scale = dpg.get_value("gravity_scale")
                self.torque_link.send_gravity([g * scale for g in G])
            time.sleep(GRAVITY_UPDATE_PERIOD)

    def _run_hardware(self, kp, kd):
        target = self._target_angles
        self._hw_buffer = []
        self._recording = True
        self._gravity_running = True
        threading.Thread(target=self._gravity_updater_loop, daemon=True).start()

        with self.state.lock:
            self.state.ctrl_joint_angle[:] = target

        self.torque_link.send_gains(kp, kd)
        self.torque_link.send_target(target)
        self.torque_link.torque_on()

        time.sleep(DURATION)

        self.torque_link.torque_off()
        self._recording = False
        self._gravity_running = False

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
        xyz_rounded = tuple(round(v, 1) for v in self._target_xyz)
        title = f"PD run to XYZ={xyz_rounded} (gravity-compensated)"
        png_path, csv_path = plotting.save_run(
            run, kp, kd, source=source, title=title, target=self._target_angles,
        )

        self._history = gain_history.append_history(
            kp, kd, png_path, mode=source, path=HISTORY_FILE,
            target_xyz=list(self._target_xyz),
        )

        dpg.set_value("run_status", f"Done ({source}). Saved {png_path.name} / {csv_path.name}")
        self._refresh_history_widget()
        try:
            os.startfile(str(png_path))
        except OSError:
            pass

        self._running = False
