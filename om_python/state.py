import threading


class SharedState:
    """Thread-safe shared state between the serial link, the 3D viewer
    (renderer.py, its own thread) and the control panel (control_panel.py,
    main thread). Mirrors the globals in the original Processing sketch."""

    def __init__(self):
        self.lock = threading.RLock()

        # Pose reported by the real arm, from "angle,.." / "tool,.." lines.
        self.receive_joint_angle = [0.0, 0.0, 0.0, 0.0]
        self.receive_gripper_pos = [30.0, -60.0]

        # Target/ghost pose driven by the control panel.
        self.ctrl_joint_angle = [0.0, 0.0, 0.0, 0.0]
        self.ctrl_gripper_pos = [0.0, 0.0]

        self.tab_flag = 1
        self.controller_on = False
        self.connected = False

        # 3D view state (model_trans/model_scale_factor/world_rot in the
        # original .pde).
        self.model_trans = [0.0, 0.0, 0.0]
        self.model_scale_factor = 0.0
        self.world_rot = [0.0, 0.0]
