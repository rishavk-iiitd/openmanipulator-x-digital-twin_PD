"""Circular Trajectory Tracking Lab -- a trimmed-down path_follow_lab.py
with only the circle path, no shape picker.

Same computed-torque (feedback-linearization) controller as the other labs
-- mass matrix M(q), Coriolis C(q,qdot), gravity G(q), all derived from the
same forward-kinematics chain (om_python/rigid_body_dynamics.py) -- but the
goal continuously traces a circle (om_python/paths.py's circle_path()) in a
chosen plane instead of sitting at one fixed point. IK is re-solved every
tick for the circle's current point, then run through the same critically-
damped reference filter as smooth_trajectory_lab.py (om_python/trajectory.py)
before reaching the controller -- driving either the simulated "virtual
robot" or the real arm, visualized live in the same 3D viewer as the other
programs.
"""
import threading

from om_python.circular_tracking_panel import CircularTrackingPanel
from om_python.renderer import ManipulatorView
from om_python.state import SharedState


def main():
    state = SharedState()

    view = ManipulatorView(state)
    view_thread = threading.Thread(target=view.run, daemon=True)
    view_thread.start()

    panel = CircularTrackingPanel(state)
    panel.run()  # blocks on the main thread until the panel window closes


if __name__ == "__main__":
    main()
