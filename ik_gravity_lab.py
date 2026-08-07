"""IK + Gravity-Compensated PD Lab -- a third program built on the same
package.

Stage 1: type a target end-effector XYZ (mm); numerical inverse kinematics
(om_python/inverse_kinematics.py) finds joint angles that reach it.
Stage 2: a full computed-torque controller -- mass matrix M(q), Coriolis
C(q,qdot), and gravity G(q), all derived from the same forward-kinematics
chain (om_python/rigid_body_dynamics.py) -- drives the simulated "virtual
robot" smoothly to that pose, visualized live in the same 3D viewer as
main.py/pd_lab.py.

Simulation only -- this does not talk to the real arm.
"""
import threading

from om_python.ik_gravity_panel import IKGravityPanel
from om_python.renderer import ManipulatorView
from om_python.state import SharedState


def main():
    state = SharedState()

    view = ManipulatorView(state)
    view_thread = threading.Thread(target=view.run, daemon=True)
    view_thread.start()

    panel = IKGravityPanel(state)
    panel.run()  # blocks on the main thread until the panel window closes


if __name__ == "__main__":
    main()
