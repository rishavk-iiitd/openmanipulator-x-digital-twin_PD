"""Parametric Cartesian paths (circle, square) traced in a plane, for the
feedback-linearization path-following controller (path_follow_lab.py).

Each function maps a time t (seconds, wraps every `period`) to an XYZ point
(mm) in world/model space. These are pure functions of t -- no state, no
integration -- unlike trajectory.py's ReferenceTrajectory, which is a
*filter* that smooths the transition between whatever this module returns.
The two compose: path_follow_panel.py samples a path here, solves IK for
that instantaneous point, then runs it through ReferenceTrajectory before
handing it to the computed-torque controller (rigid_body_dynamics.py) --
exactly the same controller used for a single fixed target in
ik_gravity_lab.py/smooth_trajectory_lab.py, just fed a continuously moving
goal instead of a static one.
"""
import math

_PLANE_AXES = ("xy", "xz", "yz")


def _place(center, plane, u, v):
    """Maps a 2D point (u, v) into the named plane through `center`."""
    cx, cy, cz = center
    if plane == "xy":
        return (cx + u, cy + v, cz)
    if plane == "xz":
        return (cx + u, cy, cz + v)
    if plane == "yz":
        return (cx, cy + u, cz + v)
    raise ValueError(f"plane must be one of {_PLANE_AXES}, got {plane!r}")


def circle_path(t, center, radius, plane="xz", period=6.0):
    """Point on a circle of the given radius (mm), completing one full loop
    every `period` seconds. Smooth (C-infinity) in t -- unlike square_path,
    tracking this exactly needs no corner-rounding."""
    angle = 2.0 * math.pi * ((t % period) / period)
    u = radius * math.cos(angle)
    v = radius * math.sin(angle)
    return _place(center, plane, u, v)


def square_path(t, center, side, plane="xz", period=8.0):
    """Point on a square of the given side length (mm), traversed at
    constant speed around the perimeter, completing one full loop every
    `period` seconds. Velocity direction flips instantaneously at each of
    the 4 corners -- a real, unavoidable motion discontinuity that
    trajectory.py's reference filter rounds off rather than tracking exactly
    (see the report's note on corner-cutting vs. smoothing_time)."""
    half = side / 2.0
    corners = ((half, -half), (half, half), (-half, half), (-half, -half))

    s = 4.0 * ((t % period) / period)  # s in [0, 4)
    leg = int(s) % 4
    frac = s - int(s)

    a = corners[leg]
    b = corners[(leg + 1) % 4]
    u = a[0] + (b[0] - a[0]) * frac
    v = a[1] + (b[1] - a[1]) * frac
    return _place(center, plane, u, v)


PATHS = {"circle": circle_path, "square": square_path}
DEFAULT_PERIOD = {"circle": 6.0, "square": 8.0}


def sample_path(shape, center, size, plane, period, n=120):
    """n evenly-spaced points around one full loop, for preview/plotting."""
    path_fn = PATHS[shape]
    return [path_fn(period * i / n, center, size, plane, period) for i in range(n)]
