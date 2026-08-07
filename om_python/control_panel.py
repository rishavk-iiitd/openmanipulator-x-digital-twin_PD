"""Control window: a functional port of the Processing sketch's ChildApplet
(controlP5 UI) to DearPyGui. Tabs mirror the original: Joint Space Control
(default), Task Space Control, Hand Guiding, Motion.
"""
import math

import dearpygui.dearpygui as dpg

from . import protocol as proto
from .utils import map_range

JOINT_TAGS = ("joint1", "joint2", "joint3", "joint4")
BASIC_POSE = (0.0, math.radians(-60.0), math.radians(20.0), math.radians(40.0))


class ControlPanel:
    def __init__(self, state, link):
        self.state = state
        self.link = link
        self.grip_angle = 0.0
        self.motion_num = 0

    def run(self):
        dpg.create_context()
        dpg.create_viewport(title="Control Interface", width=420, height=700)
        dpg.setup_dearpygui()

        with dpg.window(tag="main_window", no_close=True, no_collapse=True):
            dpg.add_text("Controller for OpenManipulator", color=(255, 204, 102))
            dpg.add_checkbox(label="Controller On/Off", callback=self._on_controller_toggle)
            dpg.add_text("", tag="status_text", color=(255, 140, 140))
            dpg.add_separator()

            with dpg.tab_bar():
                with dpg.tab(label="Joint Space Control"):
                    self._build_joint_tab()
                with dpg.tab(label="Task Space Control"):
                    self._build_task_tab()
                with dpg.tab(label="Hand Guiding"):
                    self._build_hand_tab()
                with dpg.tab(label="Motion"):
                    self._build_motion_tab()

        dpg.set_primary_window("main_window", True)
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

    # ---------------- shared helpers ----------------
    def _send(self, text):
        if not self.state.controller_on:
            self._status("Please, Set On Controller")
            print("Please, Set On Controller")
            return
        self.link.write_line(text)

    def _status(self, text):
        if dpg.does_item_exist("status_text"):
            dpg.set_value("status_text", text)

    def _sync_joint_widgets(self):
        with self.state.lock:
            vals = list(self.state.ctrl_joint_angle)
        for tag, v in zip(JOINT_TAGS, vals):
            dpg.set_value(tag, v)

    def _set_ctrl_gripper_from_angle(self, angle):
        pos = map_range(angle, 0.907, -1.13, 10.0, 35.0)
        with self.state.lock:
            self.state.ctrl_gripper_pos[0] = pos
            self.state.ctrl_gripper_pos[1] = pos * -2

    def _on_controller_toggle(self, sender, value):
        self.state.controller_on = value
        if value:
            self._sync_joint_widgets()
            self.link.write_line(proto.opm_ready())
            self._status("OpenManipulator Ready!!!")
        else:
            self.link.write_line(proto.opm_end())
            self._status("OpenManipulator End...")

    # ---------------- Joint Space Control ----------------
    def _build_joint_tab(self):
        dpg.add_slider_float(label="Joint 1", tag="joint1", min_value=-3.14, max_value=3.14,
                              default_value=0.0, callback=self._on_joint, user_data=0)
        dpg.add_slider_float(label="Joint 2", tag="joint2", min_value=-2.05, max_value=1.57,
                              default_value=0.0, callback=self._on_joint, user_data=1)
        dpg.add_slider_float(label="Joint 3", tag="joint3", min_value=-1.53, max_value=1.57,
                              default_value=0.0, callback=self._on_joint, user_data=2)
        dpg.add_slider_float(label="Joint 4", tag="joint4", min_value=-1.8, max_value=2.0,
                              default_value=0.0, callback=self._on_joint, user_data=3)
        dpg.add_slider_float(label="Gripper knob", tag="gripper_knob", min_value=-10.0, max_value=10.0,
                              default_value=0.0, callback=self._on_gripper_knob)
        dpg.add_spacer(height=10)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Origin", width=195, callback=self._on_origin)
            dpg.add_button(label="Basic", width=195, callback=self._on_basic)
        dpg.add_button(label="Send Joint Angle", width=-1, height=40, callback=self._on_send_joint_angle)
        dpg.add_button(label="Set Gripper", width=-1, height=40, callback=self._on_set_gripper)
        dpg.add_checkbox(label="Gripper Open/Close", callback=self._on_gripper_onoff)

    def _on_joint(self, sender, value, user_data):
        with self.state.lock:
            self.state.ctrl_joint_angle[user_data] = value

    def _on_gripper_knob(self, sender, value):
        self.grip_angle = value
        self._set_ctrl_gripper_from_angle(value)

    def _apply_pose(self, angles):
        with self.state.lock:
            self.state.ctrl_joint_angle[:] = angles
        self._sync_joint_widgets()
        self._send(proto.joint(angles))

    def _on_origin(self):
        self._apply_pose([0.0, 0.0, 0.0, 0.0])

    def _on_basic(self):
        self._apply_pose(list(BASIC_POSE))

    def _on_send_joint_angle(self):
        with self.state.lock:
            angles = list(self.state.ctrl_joint_angle)
        self._send(proto.joint(angles))

    def _on_set_gripper(self):
        self._send(proto.gripper_cmd(self.grip_angle * 0.001))

    def _on_gripper_onoff(self, sender, value):
        self._send(proto.grip(value))

    # ---------------- Task Space Control ----------------
    def _build_task_tab(self):
        with dpg.group(horizontal=True):
            dpg.add_spacer(width=100)
            dpg.add_button(label="Forward", width=100, height=100,
                            callback=lambda: self._send(proto.task("forward")))
        with dpg.group(horizontal=True):
            dpg.add_button(label="Left", width=100, height=100,
                            callback=lambda: self._send(proto.task("left")))
            dpg.add_button(label="Basic", width=100, height=100, callback=self._on_basic)
            dpg.add_button(label="Right", width=100, height=100,
                            callback=lambda: self._send(proto.task("right")))
        with dpg.group(horizontal=True):
            dpg.add_spacer(width=100)
            dpg.add_button(label="Back", width=100, height=100,
                            callback=lambda: self._send(proto.task("back")))
        dpg.add_spacer(height=10)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Up", width=100, height=100,
                            callback=lambda: self._send(proto.task("up")))
            dpg.add_button(label="Down", width=100, height=100,
                            callback=lambda: self._send(proto.task("down")))
        dpg.add_checkbox(label="Gripper", callback=lambda s, v: self._send(proto.grip(v)))

    # ---------------- Hand Guiding ----------------
    def _build_hand_tab(self):
        dpg.add_checkbox(label="Torque On/Off", default_value=True,
                          callback=lambda s, v: self._send(proto.torque(v)))
        with dpg.group(horizontal=True):
            dpg.add_button(label="Motion Clear", width=195, height=80,
                            callback=lambda: self._send(proto.get_clear()))
            dpg.add_button(label="Save Joint Pose", width=195, height=80,
                            callback=self._on_make_joint_pose)
        dpg.add_checkbox(label="Gripper Open/Close (save pose)", callback=self._on_make_gripper_pose)
        dpg.add_button(label="Motion Start", width=-1, height=60, callback=self._on_motion_start)
        dpg.add_checkbox(label="Motion Repeat", callback=self._on_motion_repeat)

    def _on_make_joint_pose(self):
        self._send(proto.get_pose(self.motion_num))
        self.motion_num += 1

    def _on_make_gripper_pose(self, sender, value):
        self._send(proto.get_gripper(value))
        target = 1.3 if value else 0.0
        dpg.set_value("gripper_knob", target)
        self.grip_angle = target
        self._set_ctrl_gripper_from_angle(target)
        self.motion_num += 1

    def _on_motion_start(self):
        self._send(proto.hand("once"))
        self.motion_num = 0

    def _on_motion_repeat(self, sender, value):
        self._send(proto.hand("repeat" if value else "stop"))

    # ---------------- Motion ----------------
    def _build_motion_tab(self):
        dpg.add_button(label="Motion 1", width=-1, height=100, callback=lambda: self._send(proto.motion(1)))
        dpg.add_button(label="Motion 2", width=-1, height=100, callback=lambda: self._send(proto.motion(2)))
