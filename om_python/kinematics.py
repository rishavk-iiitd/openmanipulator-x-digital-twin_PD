"""Forward-kinematics helper, originally written only for the gripper
trail effect, now also the shared FK backbone for inverse_kinematics.py
and rigid_body_dynamics.py.

LINK2_OFFSET/LINK3_OFFSET/LINK4_OFFSET/LINK5_OFFSET are each joint's exact
origin, straight from ROBOTIS's OpenMANIPULATOR-X URDF (ROBOTIS-GIT/
open_manipulator, open_manipulator_description/urdf/open_manipulator_x/
open_manipulator_x.urdf), meters converted to mm:
    world_fixed (world->link1): xyz="0 0 0"        -> link1 IS world frame
    joint1 (link1->link2):      xyz="0.012 0 0"     -> LINK2_OFFSET
    joint2 (link2->link3):      xyz="0 0 0.0595"    -> LINK3_OFFSET
    joint3 (link3->link4):      xyz="0.024 0 0.128" -> LINK4_OFFSET
    joint4 (link4->link5):      xyz="0.124 0 0"     -> LINK5_OFFSET
LINK2_OFFSET and LINK3_OFFSET used to be (0,0,36) and (0,0,40) -- numbers
carried over from a Processing sketch's drawManipulator() translate()
calls, not the real CAD dimensions (a ~20mm-scale error: e.g. joint1's
pivot was modeled as straight up from the origin, when it's actually
offset mostly in X). LINK4_OFFSET/LINK5_OFFSET already matched the URDF
exactly and are unchanged.

WRIST_OFFSET/SPHERE_OFFSET remain an approximation: they place a "gripper
centre sphere" for the trail-effect visualization (deliberately offset to
sit *between* the two gripper fingers), and no single URDF frame
corresponds to that exact point -- the URDF only defines the two finger
frames (which move with the gripper's own prismatic joint, not modeled
here) and a centerline end_effector_link (fixed joint, xyz="0.126 0 0"
from link5 -- close to WRIST_OFFSET's 130mm-X but not the same point).
"""
import numpy as np

LINK2_OFFSET = (12.0, 0.0, 0.0)
LINK3_OFFSET = (0.0, 0.0, 59.5)
LINK4_OFFSET = (24.0, 0.0, 128.0)
LINK5_OFFSET = (124.0, 0.0, 0.0)
WRIST_OFFSET = (130.0, 14.0, 0.0)
SPHERE_OFFSET = (0.0, -7.0, 0.0)

# Exact center-of-mass offset (mm) of each moving link, in that link's own
# frame -- applied at the same point in the chain WRIST_OFFSET is applied
# for the gripper, i.e. right after that link's own joint has rotated.
# Straight from ROBOTIS's OpenMANIPULATOR-X URDF <inertial><origin> per
# link (ROBOTIS-GIT/open_manipulator, open_manipulator_description/urdf/
# open_manipulator_x/open_manipulator_x.urdf), meters converted to mm --
# not a guess, and not the old "midpoint between chain joints"
# approximation link_com_positions() used to use. link1 (the static base
# bracket) has no entry -- it never moves, and rigid_body_dynamics.py's
# point-mass model excludes it (see that file's LINK_MASS comment).
LINK2_COM_OFFSET = (-0.30184870, 0.54043684, 47.433464)   # link2, after joint1 rotates
LINK3_COM_OFFSET = (10.308393, 0.37743363, 101.70197)     # link3, after joint2 rotates
LINK4_COM_OFFSET = (90.909590, 0.38929816, 0.22413279)    # link4, after joint3 rotates
LINK5_COM_OFFSET = (44.206755, 0.00036839985, 8.9142216)  # link5, after joint4 rotates

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
    """Exact center-of-mass position (mm) of each of the 4 moving links,
    from ROBOTIS's real mesh-based URDF centroids (LINK2_COM_OFFSET etc.
    above) -- not the midpoint-between-chain-joints approximation this
    used to be. Each offset is applied in that link's own frame, i.e.
    right after that link's own joint has rotated -- the same point in
    the chain gripper_center() applies WRIST_OFFSET at."""
    t = np.eye(4)
    coms = []

    t = t @ _trans(LINK2_OFFSET)
    t = t @ _rot_z(-joint_angle[0])
    coms.append(tuple((t @ _trans(LINK2_COM_OFFSET) @ _ORIGIN)[:3]))  # link2 COM

    t = t @ _trans(LINK3_OFFSET)
    t = t @ _rot_y(joint_angle[1])
    coms.append(tuple((t @ _trans(LINK3_COM_OFFSET) @ _ORIGIN)[:3]))  # link3 COM

    t = t @ _trans(LINK4_OFFSET)
    t = t @ _rot_y(joint_angle[2])
    coms.append(tuple((t @ _trans(LINK4_COM_OFFSET) @ _ORIGIN)[:3]))  # link4 COM

    t = t @ _trans(LINK5_OFFSET)
    t = t @ _rot_y(joint_angle[3])
    coms.append(tuple((t @ _trans(LINK5_COM_OFFSET) @ _ORIGIN)[:3]))  # link5 COM

    return coms
