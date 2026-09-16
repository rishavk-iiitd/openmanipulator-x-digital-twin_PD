"""3D viewer: a fairly direct port of the Processing sketch's
setWindow()/drawWorldFrame()/drawManipulator() to PyOpenGL + GLFW.

Two ways to use it, both on a background thread, both driven entirely by
the shared SharedState under its lock:

  run()                  -- its own visible GLFW window, as before. Used by
                            main.py and the labs that still open two windows.
  run_offscreen(...)     -- renders into a HIDDEN GLFW context and hands each
                            frame back as an RGBA float array, so a panel can
                            show the 3D view inside its own UI instead of
                            popping a second window. See ik_gravity_panel.

GLFW contexts are thread-affine, so whichever entry point is used must
create the context and draw on the same thread -- both of these do.
"""
import math
import time
from collections import deque
from pathlib import Path

import glfw
from OpenGL.GL import (
    GL_AMBIENT,
    GL_AMBIENT_AND_DIFFUSE,
    GL_COLOR_BUFFER_BIT,
    GL_COLOR_MATERIAL,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DIFFUSE,
    GL_FRONT_AND_BACK,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_LINES,
    GL_POSITION,
    GL_SMOOTH,
    glBegin,
    glClear,
    glClearColor,
    glColor3f,
    glColorMaterial,
    glDisable,
    glEnable,
    glEnd,
    glLightfv,
    glLineWidth,
    glLoadIdentity,
    glMatrixMode,
    glPopMatrix,
    glPushMatrix,
    glRasterPos3f,
    glRotatef,
    glScalef,
    glShadeModel,
    glTranslatef,
    glVertex3f,
    glViewport,
)
from OpenGL.GL import GL_MODELVIEW, GL_PROJECTION, GL_FLOAT, GL_RGBA, glReadPixels
from OpenGL.GLU import gluLookAt, gluNewQuadric, gluPerspective, gluSphere

try:
    from OpenGL.GLUT import GLUT_BITMAP_HELVETICA_10, glutBitmapCharacter, glutInit
    _GLUT_OK = True
except Exception:
    _GLUT_OK = False

from . import kinematics
from .mesh import load_obj

MESH_DIR = Path(__file__).resolve().parent.parent / "meshes"
WIDTH, HEIGHT = 900, 900

GRID_EXTENT = 300.0    # mm, how far the floor grid spans in +/-X and +/-Y
GRID_MINOR = 50.0      # mm between minor gridlines
GRID_MAJOR = 100.0     # mm between major (brighter, labeled) gridlines

LINK_FILES = [
    ("link1", "chain_link1.obj"),
    ("link2", "chain_link2.obj"),
    ("link3", "chain_link3.obj"),
    ("link4", "chain_link4.obj"),
    ("link5", "chain_link5.obj"),
    ("grip_r", "chain_link_grip_r.obj"),
    ("grip_l", "chain_link_grip_l.obj"),
]

GOAL_COLOR = (0.85, 0.85, 0.85)
GOAL_SPHERE_COLOR = (100 / 255, 100 / 255, 100 / 255)
CTRL_COLOR = (200 / 255, 200 / 255, 200 / 255)
CTRL_SPHERE_COLOR = (0.75, 0.75, 0.75)


class ManipulatorView:
    def __init__(self, state):
        self.state = state
        self.shapes = {}
        self.window = None
        self.quadric = None
        self.trail = deque(maxlen=50)
        self._drag_last = None
        self._camera_z = 1.0

    def run(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        self.window = glfw.create_window(WIDTH, HEIGHT, "OpenManipulator", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor_pos)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_key_callback(self.window, self._on_key)

        self._load_shapes()
        self._init_gl()
        self._init_projection()

        while not glfw.window_should_close(self.window):
            self._draw_frame()
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            time.sleep(1 / 60)

        glfw.terminate()

    # ---------------- embedded (offscreen) rendering ----------------
    def run_offscreen(self, width, height, on_frame, should_stop, fps=30.0):
        """Render into a hidden GLFW context and hand each finished frame to
        on_frame(buf) as a flat float32 RGBA array in 0..1, row 0 at the TOP
        (OpenGL reads bottom-up, so it is flipped here).

        Runs until should_stop() returns True. 30 fps rather than 60 because
        every frame costs a readback plus a copy, and the arm is not moving
        fast enough for the difference to be visible."""
        import numpy as np

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)   # context without a window
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        self.window = glfw.create_window(width, height, "OpenManipulator (embedded)",
                                         None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create offscreen GLFW context")

        glfw.make_context_current(self.window)
        self._load_shapes()
        self._init_gl()
        self._init_projection()
        glViewport(0, 0, width, height)

        raw = np.empty(height * width * 4, dtype=np.float32)
        period = 1.0 / fps
        try:
            while not should_stop():
                t0 = time.time()
                self._draw_frame()
                glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, raw)
                on_frame(raw.reshape(height, width, 4)[::-1])
                glfw.poll_events()
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
        finally:
            try:
                glfw.destroy_window(self.window)
                glfw.terminate()
            except Exception:
                pass
            self.window = None

    # ---------------- camera control from an embedding UI ----------------
    # The GLFW callbacks below only fire for the standalone window. When the
    # view is embedded these are called directly by the host panel instead,
    # so drag/zoom/nudge behave identically in both modes.
    def orbit(self, dx, dy):
        with self.state.lock:
            self.state.world_rot[0] -= dx * 2.0
            self.state.world_rot[1] -= dy * 2.0

    def zoom(self, steps):
        with self.state.lock:
            self.state.model_scale_factor += steps * 0.01

    def nudge(self, axis, amount):
        with self.state.lock:
            self.state.model_trans[axis] += amount

    def reset_view(self):
        with self.state.lock:
            self.state.model_trans[0] = 0.0
            self.state.model_trans[1] = 0.0
            self.state.model_trans[2] = 0.0
            self.state.model_scale_factor = 0.0
            self.state.world_rot[0] = 0.0
            self.state.world_rot[1] = 0.0

    def _load_shapes(self):
        for key, filename in LINK_FILES:
            self.shapes[key] = load_obj(str(MESH_DIR / filename))

    def _init_gl(self):
        if _GLUT_OK:
            try:
                glutInit()
            except Exception:
                pass
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLightfv(GL_LIGHT0, GL_POSITION, (0.0, 600.0, 900.0, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.35, 0.35, 0.35, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.9, 0.9, 0.9, 1.0))
        glShadeModel(GL_SMOOTH)
        self.quadric = gluNewQuadric()

    def _init_projection(self):
        camera_y = HEIGHT / 2.0
        fov = (200.0 / WIDTH) * (math.pi / 2.0)
        self._camera_z = camera_y / math.tan(fov / 2.0)
        aspect = WIDTH / float(HEIGHT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(math.degrees(fov), aspect, self._camera_z / 10.0, self._camera_z * 10.0)
        glMatrixMode(GL_MODELVIEW)

    def _set_camera(self):
        glLoadIdentity()
        with self.state.lock:
            wx, wy = self.state.world_rot
        gluLookAt(
            WIDTH / 2.0 + wx, HEIGHT / 2.0 - 500 + wy, HEIGHT / 2.0 * 4,
            WIDTH / 2.0 - 100, HEIGHT / 2.0, 0,
            0, -1, 0,
        )

    def _draw_frame(self):
        glViewport(0, 0, WIDTH, HEIGHT)
        glClearColor(30 / 255, 30 / 255, 30 / 255, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self._set_camera()

        glTranslatef(WIDTH / 2.0, HEIGHT / 2.0, 0.0)
        glRotatef(90, 1, 0, 0)
        glRotatef(140, 0, 0, 1)

        self._draw_axes(200)
        self._draw_floor_grid()

        with self.state.lock:
            recv_angle = list(self.state.receive_joint_angle)
            recv_grip = list(self.state.receive_gripper_pos)
            ctrl_angle = list(self.state.ctrl_joint_angle)
            ctrl_grip = list(self.state.ctrl_gripper_pos)
            model_trans = list(self.state.model_trans)
            model_scale = self.state.model_scale_factor
            tab_flag = self.state.tab_flag

        self._draw_arm(
            recv_angle, recv_grip, model_trans, model_scale,
            GOAL_COLOR, GOAL_SPHERE_COLOR, record_trail=True,
        )
        self._draw_trail()

        if tab_flag == 1:
            self._draw_arm(
                ctrl_angle, ctrl_grip, model_trans, model_scale,
                CTRL_COLOR, CTRL_SPHERE_COLOR, record_trail=False,
            )

    def _draw_arm(self, joint_angle, gripper_pos, model_trans, model_scale, color, sphere_color, record_trail):
        glPushMatrix()
        s = 1.0 + model_scale
        glScalef(s, s, s)
        glTranslatef(-model_trans[0], -model_trans[1], -model_trans[2])
        glColor3f(*color)

        self.shapes["link1"].draw()
        self._draw_axes(60)
        glColor3f(*color)

        glTranslatef(0.0, 0.0, 36.0)
        glRotatef(math.degrees(-joint_angle[0]), 0, 0, 1)
        self.shapes["link2"].draw()
        self._draw_axes(60)
        glColor3f(*color)

        glTranslatef(0.0, 0.0, 40.0)
        glRotatef(math.degrees(joint_angle[1]), 0, 1, 0)
        self.shapes["link3"].draw()
        self._draw_axes(60)
        glColor3f(*color)

        glTranslatef(24.0, 0.0, 128.0)
        glRotatef(math.degrees(joint_angle[2]), 0, 1, 0)
        self.shapes["link4"].draw()
        self._draw_axes(60)
        glColor3f(*color)

        glTranslatef(124.0, 0.0, 0.0)
        glRotatef(math.degrees(joint_angle[3]), 0, 1, 0)
        self.shapes["link5"].draw()
        self._draw_axes(60)
        glColor3f(*color)

        glTranslatef(130.0, 14.0, 0.0)
        glTranslatef(0.0, -7.0, 0.0)
        glColor3f(*sphere_color)
        gluSphere(self.quadric, 10, 16, 16)
        glColor3f(*color)

        if record_trail:
            self.trail.append(kinematics.gripper_center(joint_angle, model_trans, model_scale))

        glTranslatef(0.0, gripper_pos[0], 0.0)
        self.shapes["grip_r"].draw()
        self._draw_axes(60)
        glColor3f(*color)

        glTranslatef(0.0, -14.0, 0.0)
        glTranslatef(0.0, gripper_pos[1], 0.0)
        self.shapes["grip_l"].draw()
        self._draw_axes(60)

        glPopMatrix()

    def _draw_trail(self):
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 1.0)
        for x, y, z in self.trail:
            glPushMatrix()
            glTranslatef(x, y, z)
            gluSphere(self.quadric, 1.5, 8, 8)
            glPopMatrix()
        glEnable(GL_LIGHTING)

    def _draw_axes(self, length):
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(length, 0, 0)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, -length, 0)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, length)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_floor_grid(self):
        """Reference grid on the local Z=0 plane (the plane the arm's base
        sits on), with minor lines every GRID_MINOR mm, brighter major lines
        + numeric mm labels every GRID_MAJOR mm, for a sense of scale."""
        glDisable(GL_LIGHTING)
        n = int(GRID_EXTENT / GRID_MINOR)

        glLineWidth(1.0)
        glBegin(GL_LINES)
        for i in range(-n, n + 1):
            if i == 0:
                continue  # world axes already draw the center lines
            is_major = (i * GRID_MINOR) % GRID_MAJOR == 0
            c = 0.55 if is_major else 0.25
            glColor3f(c, c, c)
            glVertex3f(-GRID_EXTENT, i * GRID_MINOR, 0)
            glVertex3f(GRID_EXTENT, i * GRID_MINOR, 0)
            glVertex3f(i * GRID_MINOR, -GRID_EXTENT, 0)
            glVertex3f(i * GRID_MINOR, GRID_EXTENT, 0)
        glEnd()

        if _GLUT_OK:
            glColor3f(0.75, 0.75, 0.75)
            n_major = int(GRID_EXTENT / GRID_MAJOR)
            for i in range(-n_major, n_major + 1):
                if i == 0:
                    continue
                val = i * GRID_MAJOR
                self._draw_text(val, 6, 0, f"{val:g}")
                self._draw_text(6, val, 0, f"{val:g}")
            self._draw_text(GRID_EXTENT + 15, 0, 0, "X (mm)")
            self._draw_text(0, GRID_EXTENT + 15, 0, "Y (mm)")

        glEnable(GL_LIGHTING)

    def _draw_text(self, x, y, z, text):
        if not _GLUT_OK:
            return
        glRasterPos3f(x, y, z)
        for ch in text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, ord(ch))

    # ---------------- input callbacks ----------------
    def _on_mouse_button(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self._drag_last = glfw.get_cursor_pos(window)
            else:
                self._drag_last = None

    def _on_cursor_pos(self, window, xpos, ypos):
        if self._drag_last is None:
            return
        last_x, last_y = self._drag_last
        dx = xpos - last_x
        dy = ypos - last_y
        with self.state.lock:
            self.state.world_rot[0] -= dx * 2.0
            self.state.world_rot[1] -= dy * 2.0
        self._drag_last = (xpos, ypos)

    def _on_scroll(self, window, xoffset, yoffset):
        with self.state.lock:
            self.state.model_scale_factor += yoffset * 0.01

    def _on_key(self, window, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        with self.state.lock:
            if key == glfw.KEY_Q:
                self.state.model_trans[0] -= 50
            elif key == glfw.KEY_A:
                self.state.model_trans[0] += 50
            elif key == glfw.KEY_W:
                self.state.model_trans[1] += 50
            elif key == glfw.KEY_S:
                self.state.model_trans[1] -= 50
            elif key == glfw.KEY_E:
                self.state.model_trans[2] -= 50
            elif key == glfw.KEY_D:
                self.state.model_trans[2] += 50
            elif key == glfw.KEY_I:
                self.state.model_trans[0] = 0.0
                self.state.model_trans[1] = 0.0
                self.state.model_trans[2] = 0.0
                self.state.model_scale_factor = 0.0
                self.state.world_rot[0] = 0.0
                self.state.world_rot[1] = 0.0
