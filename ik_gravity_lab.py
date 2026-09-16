"""IK + Gravity-Compensated PD Lab.

Stage 1: type a target end-effector XYZ (mm); numerical inverse kinematics
(om_python/inverse_kinematics.py) finds joint angles that reach it.
Stage 2: a PD controller with gravity compensation -- G(q) derived from the
same forward-kinematics chain (om_python/rigid_body_dynamics.py) -- drives
the arm to that pose, either simulated or on the real hardware.

One window: controls on the left third, the 3D arm on the right two thirds.
The 3D view is the same renderer.py used by the other labs, but rendered
offscreen and blitted into the panel rather than opening a second window.
"""
from om_python.ik_gravity_panel import IKGravityPanel
from om_python.state import SharedState


def main():
    IKGravityPanel(SharedState()).run()


if __name__ == "__main__":
    main()
