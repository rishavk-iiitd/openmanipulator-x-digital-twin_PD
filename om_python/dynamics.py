"""Simplest possible torque-based PD control for the 4 arm joints.

Each joint is modeled as an independent (decoupled) second-order rotational
system driven by a PD torque law, with its own Kp_i/Kd_i:

    tau_i = Kp_i*(theta_target_i - theta_i) + Kd_i*(0 - theta_dot_i)
    theta_ddot_i = (tau_i - DAMPING*theta_dot_i) / INERTIA

INERTIA/DAMPING are placeholder constants, not manufacturer specs -- this is
a demonstration model for tuning PD gains and looking at the resulting
motion, not a physically-accurate simulation of the real servos.
"""

INERTIA = 0.05    # kg*m^2, placeholder per-joint inertia
DAMPING = 0.4     # N*m*s/rad, placeholder viscous damping


class JointDynamics:
    def __init__(self, n_joints=4):
        self.n = n_joints
        self.theta = [0.0] * n_joints
        self.theta_dot = [0.0] * n_joints
        self.theta_ddot = [0.0] * n_joints

    def reset(self, start_angles):
        self.theta = list(start_angles)
        self.theta_dot = [0.0] * self.n
        self.theta_ddot = [0.0] * self.n

    def step(self, dt, kp, kd, target_angles):
        """Advance one timestep; kp/kd are per-joint (length-4) sequences.
        Returns (torques, jerks) for this step."""
        torques = [0.0] * self.n
        jerks = [0.0] * self.n

        for i in range(self.n):
            error = target_angles[i] - self.theta[i]
            error_dot = 0.0 - self.theta_dot[i]
            tau = kp[i] * error + kd[i] * error_dot
            torques[i] = tau

            ddot = (tau - DAMPING * self.theta_dot[i]) / INERTIA
            jerks[i] = (ddot - self.theta_ddot[i]) / dt if dt > 0 else 0.0

            self.theta_dot[i] += ddot * dt
            self.theta[i] += self.theta_dot[i] * dt
            self.theta_ddot[i] = ddot

        return torques, jerks
