"""Numerical inverse kinematics for the 4-DOF chain: given a target
end-effector XYZ (mm), find joint angles that reach it.

A closed-form solution exists in principle, but link4's offset (24, 0, 128)
isn't axis-aligned, which makes a hand-derived geometric solution easy to
get subtly wrong. Damped least squares (Levenberg-Marquardt-style) on a
numerically-differentiated Jacobian is slower per-call but avoids that risk
entirely, degrades gracefully for unreachable targets (reports failure
instead of returning nonsense), and is a standard, well-understood approach.
"""
import numpy as np

from . import kinematics


def _jacobian(joint_angle, eps=1e-4):
    """3x4 numerical Jacobian of gripper_center() w.r.t. joint angles."""
    p0 = np.array(kinematics.gripper_center(joint_angle, (0.0, 0.0, 0.0), 0.0))
    J = np.zeros((3, 4))
    for i in range(4):
        perturbed = list(joint_angle)
        perturbed[i] += eps
        p1 = np.array(kinematics.gripper_center(perturbed, (0.0, 0.0, 0.0), 0.0))
        J[:, i] = (p1 - p0) / eps
    return J


def solve(target_xyz, initial_guess=(0.0, 0.0, 0.0, 0.0), max_iters=200,
          tol_mm=0.5, damping=8.0):
    """Damped-least-squares IK. Returns (joint_angle, reached, error_mm)."""
    q = np.array(initial_guess, dtype=float)
    target = np.array(target_xyz, dtype=float)

    for _ in range(max_iters):
        p = np.array(kinematics.gripper_center(list(q), (0.0, 0.0, 0.0), 0.0))
        err = target - p
        err_mm = float(np.linalg.norm(err))
        if err_mm < tol_mm:
            return kinematics.clamp_to_joint_limits(list(q)), True, err_mm

        J = _jacobian(list(q))
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + (damping ** 2) * np.eye(3), err)

        # Joint-limit clamping (Buss): if a joint is already sitting at a
        # limit and this step would push it further past that limit, treat
        # it as locked for this iteration -- zero its Jacobian column and
        # re-solve for the remaining free joints. Without this, a target
        # that grazes a limit (which continuous IK solving along a moving
        # path, e.g. path_follow_lab.py, can do every tick) makes the
        # solver treat the locked joint as if it were still free: it keeps
        # computing a step for it that the next clamp immediately undoes,
        # while the *other* joints' share of that same (wrong) step is not
        # undone -- so they walk away from the target instead of
        # compensating for it, and the residual can grow without bound
        # instead of settling at "closest reachable point."
        at_limit = [
            (q[i] <= kinematics.JOINT_MIN[i] and dq[i] < 0) or
            (q[i] >= kinematics.JOINT_MAX[i] and dq[i] > 0)
            for i in range(4)
        ]
        if any(at_limit):
            J_locked = J.copy()
            for i, locked in enumerate(at_limit):
                if locked:
                    J_locked[:, i] = 0.0
            JJt_locked = J_locked @ J_locked.T
            dq = J_locked.T @ np.linalg.solve(JJt_locked + (damping ** 2) * np.eye(3), err)

        q = q + dq
        q = np.array(kinematics.clamp_to_joint_limits(list(q)))

    p = np.array(kinematics.gripper_center(list(q), (0.0, 0.0, 0.0), 0.0))
    err_mm = float(np.linalg.norm(target - p))
    return kinematics.clamp_to_joint_limits(list(q)), err_mm < tol_mm, err_mm
