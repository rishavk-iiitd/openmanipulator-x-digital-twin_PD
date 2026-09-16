"""Control panel for XYZ-target, gravity-compensated computed-torque PD
control.

Stage 1: solve IK for a target end-effector XYZ (mm) -> target joint
angles, previewed as the ghost arm in the 3D view.
Stage 2: run a full computed-torque controller (mass matrix + Coriolis +
gravity compensation, rigid_body_dynamics.py) to drive the arm smoothly to
that joint configuration -- either the simulated "virtual robot", or the
real arm.

Real Hardware mode reuses firmware/open_manipulator_torque_pd.ino and
torque_link.py from pd_lab.py. That firmware's fast control loop runs
locally on the OpenCR -- serial round-trip latency is too slow for a PC-side
loop to do that part. Gravity compensation is different: G(q) only depends
on the arm's current pose, which changes slowly, so it's computed here on
the PC (using the same M(q)/C(q,qdot)/G(q) dynamics -- porting that to the
microcontroller isn't practical) and pushed over periodically as a
"gravity" feedforward the firmware adds to its own output.

THE CONTROLLER
--------------
Plain PD, per joint, running on the OpenCR at 250 Hz:

    i_cmd = Kp*(q_ref - q) + Kd*(qdot_ref - qdot) + G(q)

Kp and Kd are fixed numbers, sent once before a run and not touched while it
moves. They are CALCULATED rather than hand-tuned -- see pd_panel.HW_GC_KP /
HW_GC_KD, and `python -m om_python.gain_schedule` to recompute them. G(q) is
the one feedforward: a current-mode arm with no gravity term simply falls,
and it is streamed from here at 20 Hz because the PC owns the dynamics model.

There is no integral term, no friction compensation, no acceleration
feedforward and no pose-scheduled gains. Earlier revisions had all four; each
one bought a little steady-state accuracy and cost more than it was worth in
ways that were hard to diagnose from the arm's behaviour. The honest cost of
leaving them out is a couple of degrees of steady-state error, because the
gearbox absorbs 40-200 mA of stiction before the output shaft moves.

WHY THE HARDWARE PATH IS INSTRUMENTED
-------------------------------------
Replaying every hardware run this panel has ever logged (plots/*.csv, indexed
by ik_gain_history.json) shows the measured motor current equals Kp*error to
within 3.3 mA -- one 2.69 mA current LSB -- on all four joints across all 28
runs that parked stationary, and differs from Kp*error + G(q) by 71 mA. The
gravity feedforward was never reaching the motors: every run labelled
"gravity-compensated" was in fact plain PD, which is exactly why the arm
settled short of its target and needed a hand.

The old code could not have revealed that -- "gravity" was write-only, with
no ack and no echo. So the hardware path now:
  * reads the gravity scale ONCE, on the GUI thread, before the worker
    starts (no DearPyGui call from inside the streaming loop);
  * checks the firmware's gravity_ack counter after arming and WARNS loudly
    if compensation is not arriving, instead of quietly producing another
    plain-PD dataset;
  * records the gravity feedforward the firmware reports actually applying,
    plus the reference angle and measured loop rate, into the run CSV;
"""
import math
import os
import threading
import time
from pathlib import Path

import dearpygui.dearpygui as dpg
import numpy as np

from . import (
    gain_history, gain_schedule, inverse_kinematics, kinematics, plotting,
    rigid_body_dynamics,
)
from .pd_panel import HW_GC_KD, HW_GC_KP, _finite_diff
from .renderer import ManipulatorView
from .rigid_body_dynamics import RigidBodyDynamics
from .torque_link import TorqueLink

# One window: controls on the left third, the 3D arm on the right two thirds.
# The 3D view is rendered offscreen at a fixed VIEW_TEX square and scaled to
# fit its pane, so window size costs nothing in render time.
VIEWPORT_W, VIEWPORT_H = 1500, 950
CTRL_PANE_W = VIEWPORT_W // 3
VIEW_TEX = 760

HOME_POSE = (0.0, 0.0, 0.0, 0.0)
DT = 0.01
DURATION = 6.0

# How long the ramped point-to-point move takes, and how long the controller
# then holds the target before disarming. The hold is where steady-state
# accuracy is actually demonstrated -- a move that "arrives" but is measured
# only at the instant it arrives proves nothing about whether it stays.
MOVE_DURATION_DEFAULT = 1.5
SETTLE_DURATION = 4.5

SIM_DEFAULT_KP = [0.6, 0.6, 0.6, 0.6]
SIM_DEFAULT_KD = [0.15, 0.15, 0.15, 0.15]

# mA per N*m -- converts the SI-unit gravity torque G(q) into the current
# space the hardware Kp/Kd operate in.
#
# 560.98 is the datasheet figure: the inverse of the XM430-W350's torque
# constant from ROBOTIS's published stall spec at 12.0 V (2300 mA / 4.1 N*m).
# It is NOT what this arm needs. The first nine hardware runs made with
# gravity compensation actually reaching the motors (plots/run_20260916_1635*
# .. 1734*) show the motor holding the parked arm with consistently LESS
# current than G(q)*560.98 predicts:
#
#     i_meas / feedforward at rest:  joint 2 median 0.80, joint 3 median 0.63
#     (ratios range 0.42-0.94 by pose; worst in reaching poses)
#
# so at 560.98 the feedforward over-drives the arm: it pushes PAST the target
# in the anti-gravity direction and the PD term is left fighting it -- run
# 163731 parks 13 degrees beyond its shoulder target with ff = +313 mA and
# Kp*e = -161 mA pulling back. That was read as "does not reach"; it is the
# opposite. The default is therefore the empirical value. Why the model is
# high is not settled from nine runs -- URDF link masses heavier than this
# arm, a real Kt above the stall-derived figure, or friction sharing the
# hold -- and the pose dependence suggests the COM model rather than a pure
# scale error. This is still a live UI field for dialling it in per arm.
GRAVITY_SCALE_DATASHEET = 560.98
GRAVITY_SCALE_DEFAULT = 390.0
GRAVITY_UPDATE_PERIOD = 0.05  # 20 Hz -- G(q) changes slowly, no need for 250 Hz

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
        self._gravity_error = None
        self._hw_buffer = []
        self._ik_solved = False
        self._target_angles = list(HOME_POSE)
        self._target_xyz = (0.0, 0.0, 0.0)
        self._run_warning = None
        self._view = ManipulatorView(state)
        self._view_buf = None
        self._view_stop = threading.Event()
        self._view_thread = None
        self._drag_prev = (0.0, 0.0)
        self._history = gain_history.load_history(HISTORY_FILE)

    def run(self):
        self.build_ui()
        self.start_view()
        dpg.show_viewport()
        dpg.start_dearpygui()

        self._view_stop.set()
        if self._view_thread:
            self._view_thread.join(timeout=2.0)
        self._gravity_running = False
        if self.torque_link:
            self.torque_link.close()
        dpg.destroy_context()

    # ---------------- embedded 3D view ----------------
    def start_view(self):
        """Drive renderer.py offscreen on its own thread, copying each frame
        into the texture the image widget is showing. The GL context is
        created on that thread and never touched from anywhere else."""
        self._view_stop.clear()
        self._view_thread = threading.Thread(target=self._view_loop, daemon=True)
        self._view_thread.start()

    def _view_loop(self):
        try:
            self._view.run_offscreen(
                VIEW_TEX, VIEW_TEX, self._on_view_frame, self._view_stop.is_set,
            )
        except Exception as exc:  # noqa: BLE001 -- a dead 3D view must not kill the panel
            print(f"[3d view] stopped: {exc!r}")

    def _on_view_frame(self, frame):
        # frame is (H, W, 4) float32, already flipped to top-down by the
        # renderer; the texture wants it flat. Copied in place -- allocating a
        # fresh 9 MB buffer 30 times a second would cost more than the render.
        # Guarded because the buffer goes away when DearPyGui tears the
        # context down on close.
        try:
            self._view_buf[:] = frame.reshape(-1)
        except Exception:
            pass

    def _on_viewport_resize(self, sender=None, app_data=None):
        """Keep the split at one third / two thirds, and letterbox the square
        3D texture into whatever space that leaves."""
        try:
            vw, vh = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()
            pane_w = max(260, vw // 3)
            dpg.configure_item("ctrl_pane", width=pane_w)
            side = max(120, min(vw - pane_w - 40, vh - 60))
            dpg.configure_item("view_image", width=side, height=side)
        except Exception:
            pass

    # ---------------- camera input, forwarded to the renderer ----------------
    def _on_view_drag(self, sender, app_data):
        if not dpg.is_item_hovered("view_image") and self._drag_prev == (0.0, 0.0):
            return
        _button, dx, dy = app_data
        # DearPyGui reports the TOTAL delta since the drag began, where the
        # GLFW path got per-move deltas -- difference them so the orbit rate
        # matches the standalone window exactly.
        self._view.orbit(dx - self._drag_prev[0], dy - self._drag_prev[1])
        self._drag_prev = (dx, dy)

    def _on_view_drag_end(self, sender, app_data):
        self._drag_prev = (0.0, 0.0)

    def _on_view_wheel(self, sender, app_data):
        if dpg.is_item_hovered("view_image"):
            self._view.zoom(app_data)

    def _on_view_key(self, sender, app_data):
        if dpg.is_item_focused("target_x") or dpg.is_item_focused("target_y")            or dpg.is_item_focused("target_z"):
            return                      # typing a coordinate, not driving the camera
        step = {dpg.mvKey_Q: (0, -50), dpg.mvKey_A: (0, 50),
                dpg.mvKey_W: (1, 50), dpg.mvKey_S: (1, -50),
                dpg.mvKey_E: (2, -50), dpg.mvKey_D: (2, 50)}.get(app_data)
        if step:
            self._view.nudge(*step)
        elif app_data == dpg.mvKey_I:
            self._view.reset_view()

    def build_ui(self):
        """Everything except the blocking render loop, so a smoke test can
        construct the whole panel and catch a missing widget tag without
        opening a window."""
        dpg.create_context()
        dpg.create_viewport(title="OpenManipulator-X -- Gravity-Compensated PD",
                            width=VIEWPORT_W, height=VIEWPORT_H)
        dpg.setup_dearpygui()

        # The 3D view is drawn offscreen by renderer.py and blitted into this
        # texture, so it lives inside this window instead of opening a second
        # one. Fixed size regardless of how the window is resized -- the image
        # widget scales it -- which keeps the per-frame readback cost constant.
        self._view_buf = np.zeros(VIEW_TEX * VIEW_TEX * 4, dtype=np.float32)
        with dpg.texture_registry():
            dpg.add_raw_texture(VIEW_TEX, VIEW_TEX, self._view_buf,
                                format=dpg.mvFormat_Float_rgba, tag="view_tex")

        with dpg.window(tag="ik_main", no_close=True, no_collapse=True):
          with dpg.group(horizontal=True):
            with dpg.child_window(tag="ctrl_pane", width=CTRL_PANE_W, autosize_y=True):
                dpg.add_text("Gravity/Coriolis/Inertia-compensated PD control", color=(255, 204, 102))
                dpg.add_text(
                    "Stage 1: solve IK for a target XYZ. Stage 2: computed-torque\n"
                    "PD (M(q), C(q,qdot), G(q)) drives the arm there.",
                    color=(160, 160, 160), wrap=430,
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
                    dpg.add_text("Not connected.", tag="hw_status", color=(255, 140, 140), wrap=430)
                    with dpg.group(horizontal=True):
                        dpg.add_input_float(
                            label="Gravity scale (mA/N*m)", tag="gravity_scale",
                            default_value=GRAVITY_SCALE_DEFAULT, step=25.0, width=130,
                        )
                        dpg.add_input_float(
                            label="Move time (s)", tag="move_duration",
                            default_value=MOVE_DURATION_DEFAULT, step=0.25, width=110,
                        )
                dpg.add_separator()

                dpg.add_text("Target end-effector position (mm):")
                with dpg.group(horizontal=True):
                    dpg.add_input_float(label="X", tag="target_x", default_value=150.0, width=100)
                    dpg.add_input_float(label="Y", tag="target_y", default_value=0.0, width=100)
                    dpg.add_input_float(label="Z", tag="target_z", default_value=150.0, width=100)
                dpg.add_button(label="Solve IK (preview target pose)", width=-1, callback=self._on_solve_ik)
                dpg.add_text("", tag="ik_status", color=(140, 220, 140), wrap=430)

                dpg.add_separator()
                dpg.add_text("Per-joint gains (mA/rad, mA/(rad/s)):")
                with dpg.table(header_row=True):
                    dpg.add_table_column(label="Joint")
                    dpg.add_table_column(label="Kp")
                    dpg.add_table_column(label="Kd")
                    for j in range(4):
                        with dpg.table_row():
                            dpg.add_text(f"Joint {j + 1}")
                            dpg.add_input_float(tag=f"kp_{j}", default_value=SIM_DEFAULT_KP[j], step=0.05, width=-1)
                            dpg.add_input_float(tag=f"kd_{j}", default_value=SIM_DEFAULT_KD[j], step=0.02, width=-1)
                dpg.add_text("", tag="gain_units", color=(160, 160, 160), wrap=430)

                with dpg.group(horizontal=True):
                    dpg.add_button(label="Run to Target", width=250, height=40, callback=self._on_run)
                    dpg.add_button(label="Reset / Torque Off", width=230, height=40, callback=self._on_reset)

                dpg.add_text("", tag="run_status", color=(140, 220, 140), wrap=430)
                dpg.add_text("", tag="ctl_status", color=(150, 190, 255), wrap=430)
                dpg.add_separator()

                dpg.add_text("History (double-click to reload gains + target):")
                dpg.add_listbox([], tag="history_list", num_items=10, width=-1, callback=self._on_history_select)

            with dpg.child_window(tag="view_pane", autosize_x=True, autosize_y=True,
                                  no_scrollbar=True):
                dpg.add_image("view_tex", tag="view_image",
                              width=VIEW_TEX, height=VIEW_TEX)
                dpg.add_text("drag to orbit | wheel to zoom | Q/A W/S E/D nudge | I reset",
                             tag="view_hint", color=(120, 120, 120))

        # Camera input. The standalone window gets this from GLFW callbacks;
        # embedded, DearPyGui owns the mouse, so forward it to the same
        # methods so both modes behave identically.
        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left,
                                       callback=self._on_view_drag)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left,
                                          callback=self._on_view_drag_end)
            dpg.add_mouse_wheel_handler(callback=self._on_view_wheel)
            dpg.add_key_press_handler(callback=self._on_view_key)

        dpg.set_viewport_resize_callback(self._on_viewport_resize)
        self._update_gain_units_label()
        self._refresh_history_widget()
        dpg.set_primary_window("ik_main", True)
        self._on_viewport_resize()

    # ---------------- mode / gains ----------------
    def _on_mode_change(self, sender, value):
        hardware = value == "Real Hardware"
        if hardware:
            self._set_gains(HW_GC_KP, HW_GC_KD)
        else:
            self._set_gains(SIM_DEFAULT_KP, SIM_DEFAULT_KD)
        dpg.configure_item("hw_group", show=hardware)
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
                "Units: Kp in mA/rad, Kd in mA/(rad/s) -- the same current space the\n"
                "firmware control loop works in. PD only, no integral term: gravity\n"
                "and friction are added on top as feedforward, computed from the\n"
                "reference and the arm's physics, never from accumulated error, so\n"
                "the loop closed around the arm is still exactly Kp/Kd.",
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
            link = TorqueLink(on_state=self._on_hw_state, on_ctl=self._on_hw_ctl)
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
            # last_ctl is this same tick's data: the firmware prints "ctl"
            # immediately before "state" precisely so these line up.
            ctl = self.torque_link.last_ctl if self.torque_link else None
            q_ref = list(ctl[1]) if ctl else list(angles)
            applied_ff = list(ctl[2]) if ctl else [0.0] * 4
            self._hw_buffer.append(
                (t, list(angles), list(velocities), list(currents), q_ref, applied_ff)
            )

    def _on_hw_ctl(self, t, q_ref, applied_ff, hz, armed):
        if self._running:
            with self.state.lock:
                self.state.ctrl_joint_angle[:] = list(q_ref)

    def _on_emergency_stop(self):
        self._recording = False
        self._gravity_running = False
        self._running = False
        if self.torque_link:
            self.torque_link.torque_off()  # also abandons any running probe
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

    def _hw_settings(self):
        """Every hardware setting, read here on the GUI thread so the worker
        threads never touch DearPyGui."""
        return {
            "gravity_scale": dpg.get_value("gravity_scale"),
            "move_duration": max(0.0, dpg.get_value("move_duration")),
        }

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

            # Everything the worker needs is read here, on the GUI thread, so
            # the streaming loops never touch DearPyGui (see module docstring).
            settings = self._hw_settings()
            dpg.set_value("run_status", f"Running to XYZ={self._target_xyz} on real hardware...")
            threading.Thread(
                target=self._run_hardware, args=(kp, kd, settings), daemon=True,
            ).start()

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
    def _gravity_updater_loop(self, settings):
        """Streams G(q) as a current feedforward, 20x a second. This is the
        ONLY thing the PC computes for the running loop -- the PD itself lives
        entirely on the firmware, because a serial round trip per control tick
        would be far slower than the 250 Hz the OpenCR closes at.

        settings is a plain dict captured on the GUI thread; no DearPyGui call
        happens in here, which is the one plausible PC-side way the old loop
        could have died silently. Any exception is captured so the run can
        report it instead of producing another uncompensated dataset.
        """
        scale = settings["gravity_scale"]
        try:
            while self._gravity_running:
                if self.torque_link and self.torque_link.last_state:
                    q = self.torque_link.last_state[1]
                    G = rigid_body_dynamics.gravity_vector(q)
                    self.torque_link.send_gravity([float(g) * scale for g in G])
                time.sleep(GRAVITY_UPDATE_PERIOD)
        except Exception as exc:  # noqa: BLE001 -- surfaced in the UI below
            self._gravity_error = repr(exc)
            self._gravity_running = False

    def _run_hardware(self, kp, kd, settings):
        """Wrapped so that ANY failure still disarms. This runs on a worker
        thread, where an uncaught exception would otherwise leave the arm
        energised in current mode with the panel's _running flag stuck True --
        no further runs accepted, and no torque_off ever sent."""
        try:
            self._run_hardware_inner(kp, kd, settings)
        except Exception as exc:  # noqa: BLE001 -- reported in the UI
            try:
                if self.torque_link:
                    self.torque_link.torque_off()
            except Exception:
                pass
            dpg.set_value("run_status", f"Run failed, torque off sent: {exc!r}")
        finally:
            self._recording = False
            self._gravity_running = False
            self._running = False

    def _run_hardware_inner(self, kp, kd, settings):
        target = self._target_angles
        link = self.torque_link
        self._hw_buffer = []
        self._gravity_error = None
        self._run_warning = None

        # Sent once. These are the gains for the whole run -- nothing
        # recomputes or overwrites them while it is moving.
        link.send_gains(kp, kd)

        # Gravity first, so the arm is already being held up at the instant it
        # is armed rather than sagging and then being caught.
        self._gravity_running = True
        threading.Thread(
            target=self._gravity_updater_loop, args=(settings,), daemon=True,
        ).start()
        time.sleep(0.2)

        acks_before = link.gravity_acks
        link.torque_on()
        time.sleep(0.4)

        warning = self._compensation_warning(link, acks_before)
        rate_note = self._enforce_rate_safe_gains(link, kp, kd)
        if rate_note:
            warning = rate_note if not warning else f"{warning}  ALSO: {rate_note}" 

        # Arming latches the setpoint to the measured angle, so this cannot
        # kick; the ramp starts from where the arm actually is.
        self._recording = True
        goto_acks_before = link.acks.get("goto_ack", 0)
        link.send_goto(target, settings["move_duration"])
        time.sleep(0.3)

        if link.acks.get("goto_ack", 0) <= goto_acks_before:
            # The firmware never acknowledged the move command at all. That is
            # not a tuning problem -- it is an OpenCR still running the old
            # sketch, which has no "goto" and will therefore never move.
            link.torque_off()
            self._recording = False
            self._gravity_running = False
            self._running = False
            dpg.set_value(
                "run_status",
                "The firmware did not acknowledge 'goto', so the arm was never "
                "commanded to move. The OpenCR is still running the previous "
                "sketch -- reflash firmware/open_manipulator_torque_pd before "
                "running again.",
            )
            return

        self._run_warning = warning
        if warning:
            dpg.set_value("run_status", f"RUNNING, but: {warning}")

        time.sleep(settings["move_duration"] + SETTLE_DURATION)

        link.torque_off()
        self._recording = False
        self._gravity_running = False

        self._report_ctl_status(link)
        run = self._process_hw_buffer(self._hw_buffer)
        self._finish_run(run, kp, kd, source="hardware", settings=settings)

    # Fraction of the discrete stability limit the damping term is allowed to
    # use. Kd*T/J = 2 is the hard limit for an explicitly-evaluated derivative
    # on a double integrator; 0.4 leaves a wide margin.
    KD_RATE_MARGIN = 0.4

    def _enforce_rate_safe_gains(self, link, kp, kd):
        """Kd and the control period are not independent. A derivative term
        evaluated every T seconds behaves like real damping only while
        Kd*T/J stays well under 2 -- past that it overshoots each sample and
        the joint shakes itself apart. Kd here is sized for the 250 Hz the
        firmware targets, so if the loop is actually running slower (a failed
        sync-read drops it to ~20 Hz on the old per-joint path, a 12x hit)
        the SAME gains become violently unstable, and it shows up worst once
        the arm arrives and velocity feedback is all that is left acting.

        So rather than trust the rate, read what the firmware reports it is
        achieving and scale Kd to what that rate can actually support.
        """
        ctl = link.last_ctl
        if not ctl:
            return None
        hz = ctl[3]
        if hz <= 1.0:
            return None
        J = gain_schedule.effective_inertia(self._target_angles)
        kd_max = [self.KD_RATE_MARGIN * J[i] * gain_schedule.TORQUE_TO_CURRENT * hz
                  for i in range(4)]
        over = [i for i in range(4) if kd[i] > kd_max[i]]
        if not over:
            return None
        safe_kd = [min(kd[i], kd_max[i]) for i in range(4)]
        link.send_gains(kp, safe_kd)
        self._set_gains(kp, safe_kd)
        return (
            f"the control loop is only achieving {hz:.0f} Hz, not the 250 Hz these "
            f"gains assume, so Kd on joint(s) {[i + 1 for i in over]} would be "
            f"unstable and was reduced to {_fmt_list([round(v) for v in safe_kd])}. "
            "Expect sluggish damping until the loop rate is fixed -- check the "
            "console for 'warn,read_failures', which means the sync-read is "
            "falling back to the slow per-joint path."
        )

    def _compensation_warning(self, link, acks_before):
        """Returns a description of why gravity compensation is not live, or
        None if it is. The run proceeds either way -- an uncompensated run is
        worse than no run only if you cannot tell which one you got, and this
        plus the feedforward column in the CSV make that unmistakable."""
        if self._gravity_error:
            return f"the gravity loop raised {self._gravity_error}"
        if link.errors:
            return f"firmware rejected commands: {link.errors[-3:]}"
        if link.gravity_acks <= acks_before:
            return (
                "the firmware acknowledged no gravity updates, so compensation is "
                "not reaching the motors -- the exact failure that made every "
                "previous 'compensated' run actually plain PD. Expect the arm to "
                "sag and stop short."
            )
        ctl = link.last_ctl
        if ctl and not any(abs(f) > 0.0 for f in ctl[2]):
            return "the firmware reports a zero feedforward even though updates are arriving"
        return None

    def _report_ctl_status(self, link):
        ctl = link.last_ctl
        if not ctl:
            dpg.set_value("ctl_status", "No control diagnostics received (old firmware?).")
            return
        _t, _q_ref, applied_ff, hz, _armed = ctl
        dpg.set_value(
            "ctl_status",
            f"Control loop {hz:.0f} Hz | gravity acks {link.gravity_acks} | "
            f"gravity feedforward {_fmt_list([round(f) for f in applied_ff])} mA"
            + (f" | firmware rejections: {link.errors[-2:]}" if link.errors else ""),
        )

    def _process_hw_buffer(self, buffer):
        run = {
            "t": [], "angle": [], "position": [], "angular_velocity": [],
            "torque": [], "end_effector": [], "target_angle": [], "feedforward": [],
        }
        if not buffer:
            run["velocity"] = []
            run["jerk"] = []
            return run

        t0 = buffer[0][0]
        for t, angles, velocities, currents, q_ref, applied_ff in buffer:
            run["t"].append(t - t0)
            run["angle"].append(angles)
            run["angular_velocity"].append(velocities)
            run["torque"].append(currents)
            run["target_angle"].append(q_ref)
            run["feedforward"].append(applied_ff)
            run["position"].append([math.dist((0, 0, 0), p) for p in kinematics.joint_positions(angles)])
            run["end_effector"].append(kinematics.gripper_center(angles, (0, 0, 0), 0.0))

        run["velocity"] = _finite_diff(run["t"], run["position"])
        angular_accel = _finite_diff(run["t"], run["angular_velocity"])
        run["jerk"] = _finite_diff(run["t"], angular_accel)
        return run

    # ---------------- shared completion path ----------------
    def _finish_run(self, run, kp, kd, source, settings=None):
        xyz_rounded = tuple(round(v, 1) for v in self._target_xyz)
        title = f"PD run to XYZ={xyz_rounded} (gravity-compensated)"

        # On hardware the reference is the ramp the firmware actually followed,
        # so tracking error is measured against the trajectory that was really
        # commanded rather than against a step the arm was never asked to take.
        target = run.get("target_angle") or self._target_angles
        png_path, csv_path = plotting.save_run(
            run, kp, kd, source=source, title=title, target=target,
        )

        extra = {"target_xyz": list(self._target_xyz)}
        if settings is not None:
            extra["move_duration"] = settings["move_duration"]
        self._history = gain_history.append_history(
            kp, kd, png_path, mode=source, path=HISTORY_FILE, **extra
        )

        done = f"Done ({source}). Saved {png_path.name} / {csv_path.name}"
        # A warning raised at the start of the run must survive to the end --
        # otherwise "Done" quietly replaces "gravity never arrived", which is
        # exactly how an uncompensated run gets mistaken for a valid one.
        if self._run_warning:
            done += f"  --  BUT: {self._run_warning}"
        dpg.set_value("run_status", done)
        self._refresh_history_widget()
        try:
            os.startfile(str(png_path))
        except OSError:
            pass

        self._running = False
