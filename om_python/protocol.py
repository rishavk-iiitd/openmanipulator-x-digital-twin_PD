"""Wire protocol shared with the OpenCR firmware (open_manipulator_chain.ino
+ processing.h). These strings are exactly what the original Processing
sketch wrote to/read from the serial port — replicated here so this Python
client talks to the same, unmodified firmware.

Outgoing (PC -> OpenCR):
    opm,ready | opm,end
    joint,<j1>,<j2>,<j3>,<j4>
    gripper,<radians>
    grip,on | grip,off
    task,forward|back|left|right|up|down
    torque,on | torque,off
    get,clear
    get,pose,<n>
    get,on | get,off
    hand,once | hand,repeat | hand,stop
    motion,1 | motion,2

Incoming (OpenCR -> PC):
    angle,<j1>,<j2>,<j3>,<j4>
    tool,<angle>
"""


def opm_ready():
    return "opm,ready"


def opm_end():
    return "opm,end"


def joint(angles):
    return "joint," + ",".join(str(a) for a in angles)


def gripper_cmd(angle_rad):
    return f"gripper,{angle_rad}"


def grip(on):
    return "grip," + ("on" if on else "off")


def task(direction):
    return f"task,{direction}"


def torque(on):
    return "torque," + ("on" if on else "off")


def get_clear():
    return "get,clear"


def get_pose(motion_num):
    return f"get,pose,{motion_num}"


def get_gripper(on):
    return "get," + ("on" if on else "off")


def hand(mode):
    return f"hand,{mode}"


def motion(n):
    return f"motion,{n}"
