"""Forward-kinematics helper used only for the gripper trail effect.

The offsets below are the exact numbers used in the Processing sketch's
drawManipulator() translate() calls (already in millimetres, i.e. pre-
multiplied by 1000 the way the original .pde does).
"""
import numpy as np

LINK2_OFFSET = (0.0, 0.0, 36.0)
LINK3_OFFSET = (0.0, 0.0, 40.0)
LINK4_OFFSET = (24.0, 0.0, 128.0)
LINK5_OFFSET = (124.0, 0.0, 0.0)
WRIST_OFFSET = (130.0, 14.0, 0.0)
SPHERE_OFFSET = (0.0, -7.0, 0.0)

# Real joint limits (radians), matching open_manipulator_libs's addJoint()
# calls and firmware/open_manipulator_torque_pd.ino's JOINT_MIN/MAX.
JOINT_MIN = (-3.14159265, -2.05, -1.57079633, -1.8)
JOINT_MAX = (3.14159265, 1.57079633, 1.53, 2.0)


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def _trans(v):
    m = np.eye(4)
    m[0, 3], m[1, 3], m[2, 3] = v
    return m


def gripper_center(joint_angle, model_trans, model_scale):
    """Position of the gripper's centre sphere, in the same baked model space
    the Processing sketch's modelX/Y/Z(0,0,0) call captured it in (i.e. after
    scale + model_trans, before the camera/world rotation)."""
    t = _trans((-model_trans[0], -model_trans[1], -model_trans[2]))
    t = t @ _trans(LINK2_OFFSET)
    t = t @ _rot_z(-joint_angle[0])
    t = t @ _trans(LINK3_OFFSET)
    t = t @ _rot_y(joint_angle[1])
    t = t @ _trans(LINK4_OFFSET)
    t = t @ _rot_y(joint_angle[2])
    t = t @ _trans(LINK5_OFFSET)
    t = t @ _rot_y(joint_angle[3])
    t = t @ _trans(WRIST_OFFSET)
    t = t @ _trans(SPHERE_OFFSET)

    point = (t @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
    return tuple(point * (1.0 + model_scale))


_ORIGIN = np.array([0.0, 0.0, 0.0, 1.0])


def joint_positions(joint_angle):
    """Position of each of the 4 joints' own rotation axes (i.e. where that
    motor physically sits), in unscaled model space (mm), with no model_trans
    applied. Each entry only depends on the *preceding* joints' angles, since
    rotating a joint doesn't move its own axis -- only what's distal to it."""
    t = np.eye(4)
    positions = []

    t = t @ _trans(LINK2_OFFSET)
    positions.append(tuple((t @ _ORIGIN)[:3]))  # joint1 (waist)
    t = t @ _rot_z(-joint_angle[0])

    t = t @ _trans(LINK3_OFFSET)
    positions.append(tuple((t @ _ORIGIN)[:3]))  # joint2 (shoulder)
    t = t @ _rot_y(joint_angle[1])

    t = t @ _trans(LINK4_OFFSET)
    positions.append(tuple((t @ _ORIGIN)[:3]))  # joint3 (elbow)
    t = t @ _rot_y(joint_angle[2])

    t = t @ _trans(LINK5_OFFSET)
    positions.append(tuple((t @ _ORIGIN)[:3]))  # joint4 (wrist)
    t = t @ _rot_y(joint_angle[3])

    return positions


def clamp_to_joint_limits(joint_angle):
    return [
        min(max(joint_angle[i], JOINT_MIN[i]), JOINT_MAX[i])
        for i in range(4)
    ]


def link_com_positions(joint_angle):
    """Approximate center-of-mass position (mm) of each of the 4 moving
    links, taken as the midpoint between the joint origin it starts at and
    the next point in the chain (the following joint's origin, or the
    gripper center for the last link). Real mesh-based centroids aren't
    used -- this is a documented approximation for the dynamics model in
    rigid_body_dynamics.py, same spirit as dynamics.py's placeholder
    inertia/damping."""
    joints = joint_positions(joint_angle)
    tip = gripper_center(joint_angle, (0.0, 0.0, 0.0), 0.0)
    chain = joints + [tip]  # 5 points: joint1..joint4, then the gripper tip

    coms = []
    for i in range(4):
        a = np.array(chain[i])
        b = np.array(chain[i + 1])
        coms.append(tuple((a + b) / 2.0))
    return coms
