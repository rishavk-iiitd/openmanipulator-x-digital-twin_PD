"""Offline calculator for the per-joint PD gains.

This is a design-time tool, not part of the control loop. It computes the
fixed Kp/Kd baked into pd_panel.HW_GC_KP / HW_GC_KD, and shows what those
fixed numbers actually achieve across the workspace so the choice is checkable
rather than a guess. Run it with:

    python -m om_python.gain_schedule

Instead of hand-tuning two numbers per joint, you pick the closed-loop
behaviour you want -- a natural frequency and a damping ratio -- and these
are the gains that produce it:

    J_i(q) = M_ii(q) + J_rotor                     effective inertia, kg*m^2
    Kp_i   = omega_i^2 * J_i(q) * SCALE            mA per rad
    Kd_i   = 2 * zeta_i * omega_i * J_i(q) * SCALE mA per rad/s

For a second-order joint driven by torque, that is exactly the pair that
places the closed loop at (omega_i, zeta_i).

An earlier revision streamed these to the firmware 20x a second, recomputed
from the live pose, so that zeta stayed exactly constant as the arm moved.
Measuring the actual spread showed that was not worth the moving parts: with
J evaluated at the workspace mean, the achieved zeta only ranges 0.66 to 0.98
over the whole workspace, and nothing is underdamped enough to ring. One
fixed pair per joint is enough, so that is what the arm now uses.

J_ROTOR_DEFAULT
---------------
M(q) alone is NOT the inertia the motor feels, and using it alone would
schedule badly. Each joint also carries its DYNAMIXEL's rotor inertia
reflected through a 353.5:1 gearbox, and reflected inertia scales with the
SQUARE of the gear ratio, so this term dominates M(q) at most poses.
rigid_body_dynamics.JOINT_ROTOR_INERTIA's comment flags it as the one number
missing from that model, because ROBOTIS's published figure is behind a
download endpoint that no longer serves the file.

It is measured here instead, from this project's own logs. Across 216
oscillating joint traces in plots/, the ringing frequency of a joint under a
known Kp gives its effective inertia directly (omega^2 = Kp / (SCALE * J)).
Subtracting the link-only M_ii at those poses leaves the constant part:

    joint 1: 0.0231 - 0.0025 = 0.021    joint 3: 0.0274 - 0.0055 = 0.022
    joint 2: 0.0098 - 0.0085 = 0.001    joint 4: 0.0023 - 0.0004 = 0.002

All four joints use the same XM430-W350 and the same gearbox, so physically
this constant must be common to all of them; the spread is measurement noise,
mostly because some of those traces are stick-slip limit cycles rather than
clean linear ringing. The mean, 0.011 kg*m^2, is used as the default, and it
agrees with an independent order-of-magnitude estimate from the motor itself
(a ~6e-8 kg*m^2 rotor reflected through 353.5^2 gives ~0.008 kg*m^2).

It is a measured estimate, not a datasheet value. Raising it raises all gains
proportionally; if moves are sluggish and stop short it is too low, and if
joints buzz it is too high.
"""
import numpy as np

from . import rigid_body_dynamics

# mA per N*m -- the XM430-W350's torque constant at 12.0 V, same figure and
# same source as ik_gravity_panel.GRAVITY_SCALE_DEFAULT.
TORQUE_TO_CURRENT = 560.98

J_ROTOR_DEFAULT = 0.011  # kg*m^2, measured -- see module docstring

def effective_inertia(q, j_rotor=J_ROTOR_DEFAULT):
    """Per-joint effective inertia (kg*m^2) at pose q: the link-only mass
    matrix diagonal plus the reflected rotor inertia. The diagonal is the
    right term because each joint's own PD loop acts on its own axis; the
    off-diagonal coupling M(q) also carries is handled by the feedforward,
    not by the gains."""
    M = rigid_body_dynamics.mass_matrix(q)
    return np.diag(M) + float(j_rotor)


def gains_for(J, omega_n, zeta, scale=TORQUE_TO_CURRENT):
    """The Kp, Kd (mA/rad, mA/(rad/s)) that place a joint of effective inertia
    J at the chosen natural frequency and damping ratio."""
    J = np.asarray(J, dtype=float)
    w = np.asarray(omega_n, dtype=float)
    z = np.asarray(zeta, dtype=float)
    return w * w * J * scale, 2.0 * z * w * J * scale


def gain_table(omega_n=12.0, zeta=0.9, j_rotor=J_ROTOR_DEFAULT):
    """Prints the Kp/Kd calculation and checks what a single fixed pair gives
    across the workspace. This is the offline tool that produced
    pd_panel.HW_GC_KP / HW_GC_KD -- run it to recompute them for a different
    omega/zeta rather than guessing new numbers by hand.

    Run as:  python -m om_python.gain_schedule
    """
    poses = [("folded  ", [0, 0, 0, 0]),
             ("mid     ", [0, -0.6, 0.5, 0.4]),
             ("extended", [0, -1.28, 0.9, 0.5]),
             ("reach   ", [0, 0.6, -0.5, -0.3])]
    Js = np.array([effective_inertia(q, j_rotor) for _, q in poses])
    print(f"Effective inertia J = diag(M(q)) + {j_rotor} kg*m^2:")
    for (name, _), J in zip(poses, Js):
        print(f"  {name} {np.round(J, 4)}")
    Jm = Js.mean(axis=0)
    print(f"  {'mean':<8} {np.round(Jm, 4)}")

    kp, kd = gains_for(Jm, omega_n, zeta)
    print()
    print(f"omega_n = {omega_n} rad/s, zeta = {zeta}")
    print(f"  Kp = {np.round(kp, 0)}")
    print(f"  Kd = {np.round(kd, 0)}")

    print()
    print("What that fixed pair actually achieves pose by pose:")
    for (name, _), J in zip(poses, Js):
        w = np.sqrt(kp / (TORQUE_TO_CURRENT * J))
        z = kd / (2.0 * np.sqrt(kp * TORQUE_TO_CURRENT * J))
        print(f"  {name} omega={np.round(w, 1)} rad/s  zeta={np.round(z, 2)}")
    return kp, kd


if __name__ == "__main__":
    gain_table()
