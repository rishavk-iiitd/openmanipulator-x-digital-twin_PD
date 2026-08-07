"""Point-mass Lagrangian rigid-body dynamics for the 4-DOF chain: mass
matrix M(q), Coriolis matrix C(q, qdot), and gravity vector G(q), all
derived numerically from the same forward-kinematics chain used elsewhere
(kinematics.py) rather than a hand-derived closed form.

LINK_MASS below are approximate, not manufacturer-verified (no local URDF/
datasheet with real mass properties was available) -- same "documented
placeholder" spirit as dynamics.py's INERTIA/DAMPING. Each link's mass is
treated as a point at its COM (kinematics.link_com_positions), which
ignores each link's own rotational inertia about its own COM but captures
the dominant coupling/gravity effects -- a standard simplification for
light, slender links like these.

Method (all in SI units -- kg, meters, seconds -- converting from
kinematics.py's millimetres):
    T(q, qdot) = 0.5 * qdot^T M(q) qdot,  M(q) = sum_link  m_link * J_link^T J_link
    U(q)       = sum_link  m_link * g * z_link(q)             (z is "up")
    G_i(q)     = dU/dq_i                                       (numerical gradient)
    C_ij(q,qd) = sum_k 0.5*(dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i) * qd_k
                                                                (Christoffel symbols,
                                                                 dM/dq via numerical diff)
"""
import numpy as np

from . import kinematics

MM_TO_M = 0.001
GRAVITY = 9.81  # m/s^2

# kg, one lumped point mass per moving link (waist bracket+shoulder motor,
# upper arm, forearm, wrist+gripper) -- approximate, see module docstring.
LINK_MASS = (0.15, 0.15, 0.15, 0.15)

# kg*m^2, added to M(q)'s diagonal per joint: each servo's own rotor
# (reflected through its gearbox) and the local link structure both have
# real rotational inertia about the joint axis that a single point mass at
# the COM doesn't capture -- without this, M(q) is nearly singular whenever
# a link's COM sits close to its own rotation axis (e.g. the waist joint at
# a near-zero pose), and simulation blows up. Approximate, not datasheet.
JOINT_ROTOR_INERTIA = (0.004, 0.004, 0.004, 0.004)

# rad/s^2, safety clamp -- see the comment in RigidBodyDynamics.step().
MAX_ANGULAR_ACCEL = 500.0

EPS = 1e-4


def _link_coms_m(q):
    """Link COM positions in meters, as an (4,3) array."""
    coms_mm = kinematics.link_com_positions(list(q))
    return np.array(coms_mm) * MM_TO_M


def _link_jacobians_m(q):
    """List of 4 (3x4) Jacobians (d COM_link / d q), meters/rad."""
    p0 = _link_coms_m(q)
    Js = [np.zeros((3, 4)) for _ in range(4)]
    for i in range(4):
        perturbed = list(q)
        perturbed[i] += EPS
        p1 = _link_coms_m(perturbed)
        for link in range(4):
            Js[link][:, i] = (p1[link] - p0[link]) / EPS
    return Js


def mass_matrix(q):
    """4x4 inertia matrix M(q), kg*m^2."""
    M = np.zeros((4, 4))
    Js = _link_jacobians_m(q)
    for link, mass in enumerate(LINK_MASS):
        J = Js[link]
        M += mass * (J.T @ J)
    M += np.diag(JOINT_ROTOR_INERTIA)
    return M


def potential_energy(q):
    coms_m = _link_coms_m(q)
    return sum(mass * GRAVITY * coms_m[link][2] for link, mass in enumerate(LINK_MASS))


def gravity_vector(q):
    """4-vector G(q) = dU/dq, N*m."""
    G = np.zeros(4)
    for i in range(4):
        q_plus = list(q)
        q_plus[i] += EPS
        q_minus = list(q)
        q_minus[i] -= EPS
        G[i] = (potential_energy(q_plus) - potential_energy(q_minus)) / (2 * EPS)
    return G


def coriolis_matrix(q, qdot):
    """4x4 Coriolis/centripetal matrix C(q, qdot), via Christoffel symbols."""
    n = 4
    dM = []
    for k in range(n):
        q_plus = list(q)
        q_plus[k] += EPS
        q_minus = list(q)
        q_minus[k] -= EPS
        dM.append((mass_matrix(q_plus) - mass_matrix(q_minus)) / (2 * EPS))

    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            c_ij = 0.0
            for k in range(n):
                christoffel = 0.5 * (dM[k][i, j] + dM[j][i, k] - dM[i][j, k])
                c_ij += christoffel * qdot[k]
            C[i, j] = c_ij
    return C


class RigidBodyDynamics:
    """Simulated 'virtual robot' plant driven by full computed-torque
    control: the caller supplies Kp/Kd, this handles M/C/G compensation."""

    def __init__(self):
        self.theta = np.zeros(4)
        self.theta_dot = np.zeros(4)
        self.theta_ddot = np.zeros(4)

    def reset(self, start_angles):
        self.theta = np.array(start_angles, dtype=float)
        self.theta_dot = np.zeros(4)
        self.theta_ddot = np.zeros(4)

    def step(self, dt, kp, kd, target_angles, target_velocity=None, target_accel=None):
        """One control + physics step. kp/kd: per-joint (length-4).
        target_velocity/target_accel default to zero (plain point
        regulation, the original behavior) -- pass non-zero values to track
        a moving reference trajectory (see trajectory.py) instead of
        reacting to a step input.
        Returns (torque, gravity_term, coriolis_term) for telemetry."""
        q = self.theta
        qdot = self.theta_dot
        target = np.array(target_angles)
        target_vel = np.zeros(4) if target_velocity is None else np.array(target_velocity)
        target_acc = np.zeros(4) if target_accel is None else np.array(target_accel)

        M = mass_matrix(q)
        C = coriolis_matrix(q, qdot)
        G = gravity_vector(q)

        error = target - q
        error_dot = target_vel - qdot
        pd_term = np.array(kp) * error + np.array(kd) * error_dot

        # Computed-torque trajectory-tracking law: cancel the arm's own
        # C*qdot + G, feed forward the reference acceleration, then let the
        # PD term correct only the (now-linear, decoupled) tracking error.
        tau = M @ target_acc + pd_term + C @ qdot + G

        qddot = np.linalg.solve(M, tau - C @ qdot - G)
        # Explicit-Euler integration of this stiff a coupled system goes
        # numerically unstable for large gains well before torque limits
        # would matter (a "reasonable-looking" Kp=5 here is already enough
        # to diverge to NaN within ~30 steps -- this model's inertia is much
        # smaller than dynamics.py's placeholder, so gains that felt fine
        # there are wildly too aggressive here). Clamp as a safety net so a
        # bad gain produces a visibly-saturated response, not garbage.
        qddot = np.clip(qddot, -MAX_ANGULAR_ACCEL, MAX_ANGULAR_ACCEL)

        self.theta_dot = qdot + qddot * dt
        self.theta = q + self.theta_dot * dt
        self.theta_ddot = qddot

        return tau, G, C @ qdot
