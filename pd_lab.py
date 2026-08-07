"""PD Control Lab -- a new program built on the OpenManipulator-X viewer.

Simulates the simplest torque-based PD control driving the 4 arm joints from
home to the "Basic" pose, visualizes it live in the same 3D viewer used by
main.py, keeps a history of the Kp/Kd gains used, and plots angle, position,
velocity, angular velocity, torque, and jerk vs. time for every joint.

This does NOT talk to the real OpenCR/hardware -- the firmware only accepts
position commands over serial, so a from-scratch torque-based PD controller
is implemented here as a simulation layered on the existing visualization.
"""
import threading

from om_python.pd_panel import PDLabPanel
from om_python.renderer import ManipulatorView
from om_python.state import SharedState


def main():
    state = SharedState()

    view = ManipulatorView(state)
    view_thread = threading.Thread(target=view.run, daemon=True)
    view_thread.start()

    panel = PDLabPanel(state)
    panel.run()  # blocks on the main thread until the panel window closes


if __name__ == "__main__":
    main()
