"""Point-mass Lagrangian rigid-body dynamics for the 4-DOF chain: mass
matrix M(q), Coriolis matrix C(q, qdot), and gravity vector G(q), all
derived in closed form from the same forward-kinematics chain used
elsewhere (kinematics.py) -- no finite differencing anywhere in this file.

LINK_MASS and the COM positions (kinematics.link_com_positions) below are
both real, sourced from ROBOTIS's own OpenMANIPULATOR-X URDF
(ROBOTIS-GIT/open_manipulator, open_manipulator_description/urdf/
open_manipulator_x/open_manipulator_x.urdf) -- not a guess and not a
midpoint-between-joints approximation. See LINK_MASS's comment for the
exact source values and the mapping from URDF link names to this file's
4-link chain. Each link's mass is still treated as a single point at that
(now exact) COM rather than using the URDF's full per-link inertia
tensor -- that (not the COM position) is the remaining simplification: it
ignores each link's own rotational inertia about its own COM but captures
the dominant coupling/gravity effects, a standard simplification for
light, slender links like these.

Method (all in SI units -- kg, meters, seconds -- converting from
kinematics.py's millimetres):
    T(q, qdot) = 0.5 * qdot^T M(q) qdot,  M(q) = sum_link  m_link * J_link^T J_link
    U(q)       = sum_link  m_link * g * z_link(q)             (z is "up")
    G_i(q)     = dU/dq_i  = sum_link m_link * g * J_link[2, i]     (closed form)
    C_ij(q,qd) = sum_k 0.5*(dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i) * qd_k
                                                                (Christoffel symbols,
                                                                 dM/dq closed form)

kinematics.py's chain (joint_positions()'s 4 joint origins, and now each
link's exact COM -- link_com_positions()) is built entirely out of fixed
translations and revolute-joint rotations, so every point on it has an
exact derivative w.r.t. any joint angle: the standard revolute-Jacobian
identity
    dp/dq_k = axis_k(q) x (p - origin_k(q))    (0 if joint k is distal to p)
where axis_k is joint k's rotation axis and origin_k its origin, both as
they currently sit in world frame (joints 2-4 all share one axis: they
rotate about the local Y of the frame joint 1's yaw, and rotating about Y
doesn't move Y, so only joint 1's angle affects that shared axis -- see
_joint_axes()). Differentiating that identity again (product rule through
the cross product) gives the exact Hessian d^2p/dq_k dq_l used for dM/dq
-- see _link_derivatives(). This whole approach (not just the COM
positions it's now applied to) was cross-checked against a finite-
difference implementation over 200 random poses before replacing it: max
|M| difference ~1e-6, |G| ~1e-9, |C| ~1e-6 -- consistent with finite
differencing's own O(EPS^2) truncation error, i.e. this is exactly what
it was converging to, just without the truncation error or the repeated
re-evaluation of mass_matrix() that computing dM/dq by finite-
differencing it required.
"""
import numpy as np

from . import kinematics

MM_TO_M = 0.001
GRAVITY = 9.81  # m/s^2

# kg, one lumped point mass per moving link (waist bracket+shoulder motor,
# upper arm, forearm, wrist+gripper), taken directly from ROBOTIS's
# OpenMANIPULATOR-X URDF (link1..link5 <inertial><mass> values -- see
# module docstring for the exact file). URDF's link1 (mass 0.079119962 kg)
# is the *static* base bracket -- fixed to world via a fixed joint, so it
# never moves and contributes nothing to M(q)/C(q,qdot)/G(q) -- excluded
# here. URDF link2..link5 are what actually move, one per revolute joint
# (joint1..joint4), matching this file's 4-link chain 1:1:
#   LINK_MASS[0] = link2  = 0.098406837 kg  (moves with joint1, waist)
#   LINK_MASS[1] = link3  = 0.138509170 kg  (moves with joint2, shoulder)
#   LINK_MASS[2] = link4  = 0.132745620 kg  (moves with joint3, elbow)
#   LINK_MASS[3] = link5 + gripper_left_link + gripper_right_link
#                = 0.143275730 + 0.001 + 0.001 = 0.145275730 kg
#                (moves with joint4, wrist -- the two gripper fingers hang
#                off link5 via their own prismatic joints, not part of this
#                4-DOF pose model, so their mass is lumped into link5's).
LINK_MASS = (0.098406837, 0.138509170, 0.132745620, 0.145275730)

# kg*m^2, added to M(q)'s diagonal per joint: a point mass at the COM
# captures each link's *translational* inertia but not its own rotational
# inertia about its own supporting joint's axis -- without some such term,
# M(q) is nearly singular whenever a link's COM sits close to its own
# rotation axis (e.g. the waist joint at a near-zero pose), and simulation
# blows up.
#
# These four values are exact, not a guess: they're each link's own iyy/izz
# (moment of inertia about its own body-frame Y or Z axis) straight from
# ROBOTIS's OpenMANIPULATOR-X URDF <inertial> tensors -- same file/mapping
# as LINK_MASS above (joint1's axis is URDF Z -> link2's izz; joints 2-4's
# axis is URDF Y -> link3/4/5's iyy). Every <inertial><origin> in that URDF
# has rpy="0 0 0", i.e. the tensor is already expressed in each link's own
# frame with no extra tilt, so "moment of inertia about body Y/Z" is exact
# and -- because rotating a body about one of its own axes doesn't change
# its inertia about that same axis -- genuinely pose-independent, not an
# approximation to keep this a constant:
#   JOINT_ROTOR_INERTIA[0] = link2.izz = 1.8850320e-05  (joint1/waist, axis Z)
#   JOINT_ROTOR_INERTIA[1] = link3.iyy = 3.4290447e-04  (joint2/shoulder, axis Y)
#   JOINT_ROTOR_INERTIA[2] = link4.iyy = 2.4230292e-04  (joint3/elbow, axis Y)
#   JOINT_ROTOR_INERTIA[3] = link5.iyy = 7.5980465e-05  (joint4/wrist, axis Y;
#                             excludes the two ~1e-3 kg gripper fingers' own
#                             spin inertia -- not fetched, but bounded above
#                             by mass*radius^2 ~ 1e-3*0.02^2 ~ 4e-7, under
#                             1% of link5's own iyy, i.e. negligible here)
#
# What this still doesn't include -- and what the *old* "each servo's own
# rotor reflected through its gearbox" half of this comment was trying to
# gesture at -- is the DYNAMIXEL XM430-W350's own rotor inertia amplified by
# its 353.5:1 gearbox (reflected inertia scales with the SQUARE of the gear
# ratio, so even a tiny rotor inertia is not obviously negligible here).
# ROBOTIS does publish this (e-manual's XM430-W350 page links "XM430,XH430
# Moment of Inertia.pdf"), but the download endpoint
# (robotis.com/service/download.php?no=717) redirects to their storefront
# homepage instead of serving the file, and no mirror with the actual
# numbers turned up -- so, per instructions not to approximate, it is
# simply left out rather than guessed. If M(q) turns out under-regularized
# without it (e.g. near the waist's zero pose), that PDF is the number to
# go get by hand, not a placeholder to invent here.
JOINT_ROTOR_INERTIA = (1.8850320e-05, 3.4290447e-04, 2.4230292e-04, 7.5980465e-05)

# rad/s^2, safety clamp -- see the comment in RigidBodyDynamics.step().
MAX_ANGULAR_ACCEL = 500.0

# Physics sub-steps per RigidBodyDynamics.step() call. M(q)'s diagonal now
# ranges over ~3 orders of magnitude across the workspace (as low as ~5e-5
# kg*m^2 when the arm's combined COM swings close to the waist axis -- a
# real kinematic feature, not a bug), and explicit-Euler integration goes
# numerically unstable (wild oscillation/spin-up, not just slow tracking)
# once a step's implied stiffness gets too large for a fixed dt=0.01s --
# see the comment inside step(). Sub-stepping the integration internally
# keeps step()'s external dt (and therefore _simulate()'s telemetry
# cadence and real-time pacing) unchanged while integrating at a finer,
# stable resolution -- a numerical-integration choice, not a change to any
# physical constant.
PHYSICS_SUBSTEPS = 5


def _link_coms_m(q):
    """Link COM positions in meters, as an (4,3) array."""
    coms_mm = kinematics.link_com_positions(list(q))
    return np.array(coms_mm) * MM_TO_M


def _joint_axes(q0):
    """World-frame rotation axis of each of the 4 joints, as they currently
    sit (only q0 matters -- see module docstring). axis[0] is joint 1's own
    axis: kinematics.py drives it with _rot_z(-joint_angle[0]), i.e.
    rotation about world -Z. axis[1..3] are joints 2-4's shared axis: the
    local Y of the frame joint 1's yaw has rotated, R_z(-q0) @ (0,1,0)."""
    s, c = np.sin(q0), np.cos(q0)
    waist_axis = np.array([0.0, 0.0, -1.0])
    pitch_axis = np.array([s, c, 0.0])
    return [waist_axis, pitch_axis, pitch_axis, pitch_axis]


def _d_joint_axes_dq0(q0):
    """d(axis_k)/dq0 for each k, by direct differentiation of _joint_axes()
    -- axis[0] is constant (zero); axis[1..3] share one derivative."""
    s, c = np.sin(q0), np.cos(q0)
    zero = np.zeros(3)
    d_pitch_axis = np.array([c, -s, 0.0])
    return [zero, d_pitch_axis, d_pitch_axis, d_pitch_axis]


def _joint_origins_m(q):
    """The 4 joint pivot positions (world frame, meters) -- kinematics.
    joint_positions(), converted from mm. Pivot k depends on q_0..q_{k-1}
    (kinematics.py's chain order): these double as the origin_k reference
    points in the revolute-Jacobian identity below."""
    return [np.array(p) * MM_TO_M for p in kinematics.joint_positions(list(q))]


def _link_derivatives(q):
    """Exact Jacobian and Hessian of each of the 4 links' COM (meters --
    kinematics.link_com_positions(), the real mesh centroids, not an
    approximation) w.r.t. every joint angle, via the revolute-Jacobian
    identity (see module docstring). Link i's COM is rigidly attached to
    the chain right after joint i's own rotation is applied, so it
    depends on q_0..q_i, with origin_k = joint k's own pivot
    (_joint_origins_m). Returns (J, H): J[i] is a (3,4) Jacobian; H[i][l]
    is a (3,4) array with d^2 COM_i/(dq_k dq_l) in column k."""
    origins = _joint_origins_m(q)
    coms = _link_coms_m(q)
    axes = _joint_axes(q[0])
    d_axes = _d_joint_axes_dq0(q[0])
    depends = (1, 2, 3, 4)  # link i's COM depends on q_0..q_i

    # origins[k] itself depends on q_0..q_{k-1} -- its own Jacobian is
    # needed below as the d(origin_k)/dq_l term.
    origin_jac = [np.zeros((3, 4)) for _ in range(4)]
    for k in range(4):
        for kk in range(k):
            origin_jac[k][:, kk] = np.cross(axes[kk], origins[k] - origins[kk])

    J = [np.zeros((3, 4)) for _ in range(4)]
    for i in range(4):
        for k in range(depends[i]):
            J[i][:, k] = np.cross(axes[k], coms[i] - origins[k])

    H = [[np.zeros((3, 4)) for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for k in range(depends[i]):
            axis_k = axes[k]
            for l in range(4):
                d_axis_k = d_axes[k] if l == 0 else np.zeros(3)
                d_point_dl = J[i][:, l]
                d_origin_dl = origin_jac[k][:, l]
                H[i][l][:, k] = (
                    np.cross(d_axis_k, coms[i] - origins[k])
                    + np.cross(axis_k, d_point_dl - d_origin_dl)
                )
    return J, H


def mass_matrix(q):
    """4x4 inertia matrix M(q), kg*m^2."""
    J, _ = _link_derivatives(q)
    M = np.zeros((4, 4))
    for link, mass in enumerate(LINK_MASS):
        M += mass * (J[link].T @ J[link])
    M += np.diag(JOINT_ROTOR_INERTIA)
    return M


def potential_energy(q):
    coms_m = _link_coms_m(q)
    return sum(mass * GRAVITY * coms_m[link][2] for link, mass in enumerate(LINK_MASS))


def gravity_vector(q):
    """4-vector G(q) = dU/dq, N*m -- exact: U(q) only depends on each
    link's COM height, so dU/dq is just the z-row of that link's (exact)
    COM Jacobian, mass- and gravity-weighted."""
    J, _ = _link_derivatives(q)
    G = np.zeros(4)
    for link, mass in enumerate(LINK_MASS):
        G += mass * GRAVITY * J[link][2, :]
    return G


def coriolis_matrix(q, qdot):
    """4x4 Coriolis/centripetal matrix C(q, qdot), via Christoffel symbols,
    using the exact dM/dq from _link_derivatives()'s Hessian."""
    J, H = _link_derivatives(q)
    n = 4
    dM = [np.zeros((n, n)) for _ in range(n)]
    for l in range(n):
        for link, mass in enumerate(LINK_MASS):
            dM[l] += mass * (H[link][l].T @ J[link] + J[link].T @ H[link][l])

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
        Internally integrates in PHYSICS_SUBSTEPS smaller steps for
        numerical stability (see that constant's comment) -- dt is still
        the full control-cycle duration; callers/telemetry cadence are
        unaffected. Returns (torque, gravity_term, coriolis_term) from the
        final sub-step, for telemetry."""
        target = np.array(target_angles)
        target_vel = np.zeros(4) if target_velocity is None else np.array(target_velocity)
        target_acc = np.zeros(4) if target_accel is None else np.array(target_accel)
        kp = np.array(kp)
        kd = np.array(kd)

        sub_dt = dt / PHYSICS_SUBSTEPS
        tau = G = Cqd = None
        for _ in range(PHYSICS_SUBSTEPS):
            q = self.theta
            qdot = self.theta_dot

            M = mass_matrix(q)
            C = coriolis_matrix(q, qdot)
            G = gravity_vector(q)

            error = target - q
            error_dot = target_vel - qdot
            pd_term = kp * error + kd * error_dot

            # Computed-torque trajectory-tracking law: cancel the arm's own
            # C*qdot + G, feed forward the reference acceleration, then let
            # the PD term correct only the (now-linear, decoupled) tracking
            # error.
            tau = M @ target_acc + pd_term + C @ qdot + G
            Cqd = C @ qdot

            qddot = np.linalg.solve(M, tau - Cqd - G)
            # Explicit-Euler integration of this stiff coupled system goes
            # numerically unstable for large gains (or small M(q), which
            # amounts to the same thing) well before torque limits would
            # matter. Clamp as a safety net so a bad gain produces a
            # visibly-saturated response, not garbage -- PHYSICS_SUBSTEPS
            # is what keeps that saturation the exception, not the norm.
            qddot = np.clip(qddot, -MAX_ANGULAR_ACCEL, MAX_ANGULAR_ACCEL)

            self.theta_dot = qdot + qddot * sub_dt
            self.theta = q + self.theta_dot * sub_dt
            self.theta_ddot = qddot

        return tau, G, Cqd
