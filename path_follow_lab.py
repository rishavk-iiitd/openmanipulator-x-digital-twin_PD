"""Feedback-Linearization Path-Follow Lab -- a fourth program on the same
package.

Same computed-torque (feedback-linearization) controller as
ik_gravity_lab.py/smooth_trajectory_lab.py -- mass matrix M(q), Coriolis
C(q,qdot), gravity G(q), all derived from the same forward-kinematics chain
(om_python/rigid_body_dynamics.py) -- but instead of a single fixed XYZ
target, the goal continuously moves along a parametric circle or square
(om_python/paths.py) traced in a chosen plane. IK is re-solved every tick
for the path's current point, then run through the same critically-damped
reference filter as smooth_trajectory_lab.py (om_python/trajectory.py)
before reaching the controller -- driving either the simulated "virtual
robot" or the real arm, visualized live in the same 3D viewer as the other
programs.
"""
import threading

from om_python.path_follow_panel import PathFollowPanel
from om_python.renderer import ManipulatorView
from om_python.state import SharedState


def main():
    state = SharedState()

    view = ManipulatorView(state)
    view_thread = threading.Thread(target=view.run, daemon=True)
    view_thread.start()

    panel = PathFollowPanel(state)
    panel.run()  # blocks on the main thread until the panel window closes


if __name__ == "__main__":
    main()
