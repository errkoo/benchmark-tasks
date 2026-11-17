"""On the station table, set up as a clinic desk for taking blood pressure on a training arm.

- The arm: a life-size training arm (a male display-mannequin arm, rigid PVC), a left arm held straight out on the
  midline between the two robot arms, palm up, at heart level. Its cut shoulder end is screwed to a wooden block on the
  table 3 cm in front of the arm bases; its forearm and hand lie on a thin foam pad that starts 2 cm past the elbow, so
  the upper arm is free all round and 1.5 cm clear of the table under it. The elbow crease is 33 cm in front of the arm
  bases; the upper arm's axis is 5.8 cm above the table. The upper arm is 26 cm round at the elbow crease, 28 cm round
  10 cm above it and 31 cm round at the shoulder end.
- The cuff: a clinic blood-pressure cuff (Welch Allyn FlexiPort reusable cuff, adult size 11, for arms 25 to 34 cm
  round), a navy nylon band 52 x 14.5 cm and 3 mm thick, lying open on the table under the upper arm, inner face up,
  its length across the arm. Its two ends keep the curl of a cuff stored rolled up: each end's last 8 cm curls up off
  the table. Its bladder fills the 27 cm at the right (-y) end; the white artery arrow on the outer face marks the
  bladder's middle and points at the cuff's lower edge (the edge toward the elbow). The outer face is loop fabric all
  over; the last 12 cm of the inner face, at the left end, is the hook strip. A short tube leaves the lower edge 10 cm
  from the right end, joined to the monitor's hose. The cuff lies 2.75 cm too far up the arm: its lower edge is 5.25
  cm above the elbow crease.
- The monitor: an automatic clinic blood-pressure monitor lying face up to the right of the arm, joined to the cuff by
  its hose (57 cm free; the rest is coiled and tied at the monitor). Two buttons on top: the green INFLATE button
  pumps the cuff's bladder up to its holding pressure (2 seconds), which takes up the cuff's slack and draws it tight
  round the arm; the blue START button then takes the reading, which needs the cuff left alone for 2 seconds.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    the same with --policy yam_scripted (plays demo() below).
"""

from __future__ import annotations

import math
import os
import pathlib
import sys
from collections import namedtuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import bodykit  # noqa: E402
from bodykit import Body, BoxShape, CylShape, SphereShape, as_numpy, glue, on_reset, on_step  # noqa: E402
from bodykit import rotation, set_glue_on, slider  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 2.0
TOP = TABLE_TOP[SCENE]

INSTRUCTION = "Put the blood-pressure cuff on the arm, inflate it and start the monitor."

# -- the arm (world frame: x forward from the arm bases, y left, z up; metres) -------------------------------------------
X_SHOULDER = 0.030                      # the upper arm's cut shoulder end, against its mounting block
X_CREASE = 0.330                        # the elbow crease, on the midline
ARM_Z = TOP + 0.058                     # the upper arm's axis
R_CREASE, R_SHOULDER = 0.0414, 0.0493   # the upper arm's radius at the crease and at the shoulder end (26 and 31 cm)
R_ELBOW, R_WRIST = 0.039, 0.028         # the forearm's
X_WRIST = X_CREASE + 0.250
REST_TOP = ARM_Z - 0.040                # the foam pad's top: the forearm and the hand lie on it
REST_X = (X_CREASE + 0.020, X_WRIST + 0.200)
REST_W = 0.130
BLOCK_X = (X_SHOULDER - 0.060, X_SHOULDER)
BLOCK_W = 0.120
BLOCK_TOP = ARM_Z + 0.050

# -- the cuff ----------------------------------------------------------------------------------------------------------
CUFF_N, SEG = 13, 0.040                 # 13 stiff strips of 4 cm, hinged to each other: 52 cm
CUFF_W, CUFF_T = 0.145, 0.003
CUFF_L = CUFF_N * SEG
CUFF_X = 0.205                          # the middle of its width: its lower edge at x 0.2775, its upper edge at 0.1325
CUFF_Y0 = -0.270                        # its right end (s = 0); s runs along +y to the left end at y 0.25
CUFF_Z = TOP + CUFF_T / 2 + 0.0003      # the strips' middle plane, lying on the table
BLADDER = (0.010, 0.280)                # s range of the bladder inside the cuff
MARKER_S = 0.145                        # the artery arrow: the bladder's middle
PORT_S = 0.100                          # the tube leaves the lower edge here
STUB_L, STUB_R = 0.015, 0.003
HOOK_S = 0.400                          # the hook strip: s 0.40 to 0.52, on the inner face
TAIL = (10, 11, 12)                     # the strips carrying it
UNDER = tuple(range(0, 9))              # strips whose loop face it can land on
CUFF_MASS = 0.091
PORT_MASS = 0.003
BEND_SPRING, BEND_TARGET = 1.0e-4, 15.0     # N m per degree toward a slight curl: the nylon's give
CURL_SPRING, CURL_TARGET = 1.0e-3, 45.0     # the curl kept at each end (the last two joints at each end)
CURLED = (0, 1, 10, 11)                     # those joints (bend i joins strips i and i + 1)
BEND_DAMP = 2.0e-5                          # N m s per degree
BEND_LIMIT = 150.0                          # degrees either way at each strip joint
# The bladder inflating: the strip joints inside it (s 0.04 to 0.28) take up the band's slack, at most SQUEEZE each,
# pulling with at most PUMP_N of band tension, so the wrap closes onto the arm and stops there.
PUMP_BENDS = tuple(range(0, 7))
SQUEEZE = 0.020                             # m per joint: 14 cm in all
PUMP_K, PUMP_D, PUMP_N = 1000.0, 10.0, 5.0  # N/m, N s/m, N
PUMP_S = 2.0                                # seconds to inflate

# -- the monitor and its hose --------------------------------------------------------------------------------------------
MON_XY = (0.270, -0.420)
MON_L, MON_W, MON_H = 0.140, 0.100, 0.050     # along x, y, z
MON_MASS = 0.45
BUTTONS = {"inflate": (-0.035, 0.025), "start": (-0.035, -0.025)}     # each button's centre from the monitor's centre
BUTTON_R, BUTTON_H, BUTTON_SINK = 0.011, 0.005, 0.001
BUTTON_TRAVEL, BUTTON_K = 0.003, 1000.0       # 3 mm of travel; 1000 N/m: 1.5 mm takes 1.5 N
PRESSED = 0.0015                              # pushed down this far: the button acts
HOSE_R = 0.003
HOSE_Z = TOP + HOSE_R + 0.0008                # the hose's axis, lying on the table
STUB_TIP = (CUFF_X + CUFF_W / 2 + STUB_L, CUFF_Y0 + PORT_S)
SOCKET = (MON_XY[0] + MON_L / 2, MON_XY[1])   # where the hose enters the monitor, on its front (+x) face
HOSE_PATH = (STUB_TIP, (0.460, STUB_TIP[1]), (0.500, STUB_TIP[1] - 0.040), (0.500, -0.380), (0.460, MON_XY[1]),
             SOCKET)
HOSE_N = 14
HOSE_MASS = 0.004                             # per piece
HOSE_BEND = 2.0e-4                            # N m per degree
HOSE_DAMP = 5.0e-5                            # N m s per degree

# -- the rules ---------------------------------------------------------------------------------------------------------
PRESS_N = 3.0           # hook pressed onto loop with at least this force (N): they stick
FACE_DEG = 40.0         # ... lying face to face, within this angle
TOUCH_N = 0.3           # a robot link on the cuff with more than this (N) touches it
STICK_MIN = 2           # hook strips (4 cm each) stuck for the cuff to count as closed: 8 of the 12 cm
WRAP_GAP = 0.070        # wrapped round the arm: every strip within this (m) of the skin (before it is inflated)
TIGHT_GAP = 0.015       # inflated tight: every strip of the closed loop within this (m) of the skin
EDGE_ABOVE = (0.015, 0.040)     # the lower edge this far above the elbow crease (m), all along
READ_S = 2.0            # the reading: the cuff left alone this long (s) after START
POINTS = {"closed": 0.40, "height": 0.20, "inflated": 0.20, "reading": 0.20}

# -- colours -----------------------------------------------------------------------------------------------------------
SKIN, CREASE = (0.86, 0.70, 0.58), (0.66, 0.48, 0.38)
WOOD, FOAM = (0.62, 0.45, 0.28), (0.56, 0.60, 0.66)
NAVY, BINDING, HOOK, WHITE = (0.11, 0.15, 0.30), (0.19, 0.24, 0.42), (0.72, 0.72, 0.70), (0.95, 0.95, 0.95)
PORT_GREY, HOSE_GREY = (0.82, 0.82, 0.80), (0.22, 0.23, 0.25)
MON_GREY, SCREEN = (0.86, 0.87, 0.88), (0.10, 0.12, 0.14)
BUTTON_COLOR = {"inflate": (0.18, 0.62, 0.30), "start": (0.15, 0.42, 0.78)}     # green, blue


def upper_arm_radius(x: float) -> float:
    """The upper arm's radius at world x (m)."""
    u = min(max((X_CREASE - x) / (X_CREASE - X_SHOULDER), 0.0), 1.0)
    return R_CREASE + u * (R_SHOULDER - R_CREASE)


def _arm() -> Body:
    """The training arm, its foam pad and its mounting block: one fixed body, origin on the upper arm's axis at the
    elbow crease."""
    shapes = []
    n = 30                                                   # the upper arm in 1 cm slices, widening to the shoulder
    for k in range(n):
        x = X_SHOULDER + (k + 0.5) * (X_CREASE - X_SHOULDER) / n
        shapes.append(CylShape((x - X_CREASE, 0.0, 0.0), upper_arm_radius(x), (X_CREASE - X_SHOULDER) / n, SKIN,
                               axis="X"))
    shapes.append(SphereShape((0.004, 0.0, -0.003), 0.040, SKIN))                      # the elbow
    shapes.append(BoxShape((0.0, 0.0, R_CREASE + 0.0002), (0.003, 0.050, 0.0006), CREASE, collide=False))
    n = 10                                                   # the forearm, lying on the pad
    for k in range(n):
        u = (k + 0.5) / n
        r = R_ELBOW + u * (R_WRIST - R_ELBOW)
        shapes.append(CylShape((u * (X_WRIST - X_CREASE), 0.0, REST_TOP - ARM_Z + r), r, (X_WRIST - X_CREASE) / n,
                               SKIN, axis="X"))
    lie = REST_TOP - ARM_Z                                   # the hand, palm up, on the pad, thumb to the left (+y)
    wx = X_WRIST - X_CREASE
    shapes.append(BoxShape((wx + 0.050, 0.0, lie + 0.014), (0.100, 0.085, 0.028), SKIN))
    for dy in (-0.027, -0.009, 0.009, 0.027):
        shapes.append(BoxShape((wx + 0.140, dy, lie + 0.009), (0.080, 0.017, 0.017), SKIN))
    shapes.append(BoxShape((wx + 0.040, 0.058, lie + 0.011), (0.060, 0.020, 0.020), SKIN, yaw=0.45))   # the thumb
    rx = (REST_X[0] + REST_X[1]) / 2 - X_CREASE
    shapes.append(BoxShape((rx, 0.0, (TOP + REST_TOP) / 2 - ARM_Z), (REST_X[1] - REST_X[0], REST_W, REST_TOP - TOP),
                           FOAM))
    bx = (BLOCK_X[0] + BLOCK_X[1]) / 2 - X_CREASE
    shapes.append(BoxShape((bx, 0.0, (TOP + BLOCK_TOP) / 2 - ARM_Z), (BLOCK_X[1] - BLOCK_X[0], BLOCK_W, BLOCK_TOP - TOP),
                           WOOD))
    return Body.build("arm", pos=(X_CREASE, 0.0, ARM_Z), mass=2.0, shapes=shapes, kinematic=True, color=SKIN,
                      friction=(0.6, 0.5))


def strip_of(s: float) -> int:
    return min(max(int(s // SEG), 0), CUFF_N - 1)


def strip_y(i: int) -> float:
    """World y of strip i's middle at the start."""
    return CUFF_Y0 + (i + 0.5) * SEG


STUB_Z = HOSE_Z - CUFF_Z                    # the tube's axis above the strip's middle plane


def _strip(i: int) -> Body:
    """Strip i of the cuff: origin in its middle; local x across the cuff (toward the lower edge), local y along it
    (s), local z out of the inner face."""
    s0 = i * SEG
    shapes = [BoxShape((0.0, 0.0, 0.0), (CUFF_W, SEG, CUFF_T), NAVY)]
    for side in (1.0, -1.0):                                 # the bound edges, both faces
        for face in (1.0, -1.0):
            shapes.append(BoxShape((side * (CUFF_W / 2 - 0.004), 0.0, face * (CUFF_T / 2 + 0.0002)),
                                   (0.008, SEG, 0.0004), BINDING, collide=False))
    if i in TAIL:                                            # the hook strip, on the inner face
        shapes.append(BoxShape((0.0, 0.0, CUFF_T / 2 + 0.0002), (CUFF_W - 0.020, SEG, 0.0004), HOOK, collide=False))
    if s0 <= MARKER_S < s0 + SEG:                            # the artery arrow on the outer face, pointing at the lower edge
        ly, z = MARKER_S - s0 - SEG / 2, -CUFF_T / 2 - 0.0003
        shapes.append(BoxShape((0.0125, ly, z), (0.065, 0.005, 0.0004), WHITE, collide=False))
        for sign in (1.0, -1.0):
            a = math.radians(35.0)
            shapes.append(BoxShape((0.045 - 0.008 * math.cos(a), ly + sign * 0.008 * math.sin(a), z),
                                   (0.018, 0.004, 0.0004), WHITE, yaw=-sign * a, collide=False))
    mass = CUFF_MASS / CUFF_N
    if s0 <= PORT_S < s0 + SEG:                              # the tube leaving the lower edge
        shapes.append(CylShape((CUFF_W / 2 + STUB_L / 2, PORT_S - s0 - SEG / 2, STUB_Z), STUB_R, STUB_L, PORT_GREY,
                               axis="X"))
        mass += PORT_MASS
    return Body.build(f"cuff{i}", pos=(CUFF_X, strip_y(i), CUFF_Z), mass=mass, shapes=shapes, color=NAVY,
                      friction=(0.5, 0.4), damping=(0.05, 0.1), contact_offset=0.001, solver_iterations=32)


def _monitor() -> Body:
    shapes = [BoxShape((0.0, 0.0, 0.0), (MON_L, MON_W, MON_H), MON_GREY),
              BoxShape((0.025, 0.0, MON_H / 2 + 0.0003), (0.060, 0.070, 0.0006), SCREEN, collide=False)]
    for dx, dy in BUTTONS.values():                          # a dark ring round each button
        shapes.append(CylShape((dx, dy, MON_H / 2 + 0.0002), BUTTON_R + 0.003, 0.0004, SCREEN, collide=False))
    return Body.build("monitor", pos=(*MON_XY, TOP + MON_H / 2 + 0.0005), mass=MON_MASS, shapes=shapes,
                      color=MON_GREY, friction=(0.8, 0.7))


BUTTON_Z = TOP + 0.0005 + MON_H + BUTTON_H / 2 - BUTTON_SINK


def button_pos(name: str) -> tuple:
    dx, dy = BUTTONS[name]
    return MON_XY[0] + dx, MON_XY[1] + dy, BUTTON_Z


def _button(name: str) -> Body:
    color = BUTTON_COLOR[name]
    return Body.build(f"{name}_button", pos=button_pos(name), mass=0.005,
                      shapes=[CylShape((0.0, 0.0, 0.0), BUTTON_R, BUTTON_H, color)], color=color, friction=(0.8, 0.7))


def hose_points() -> np.ndarray:
    """HOSE_N + 1 points (world) at equal spacing along HOSE_PATH: the hose's joints, from the tube to the socket."""
    path = np.array(HOSE_PATH, float)
    legs = np.linalg.norm(np.diff(path, axis=0), axis=1)
    at = np.concatenate([[0.0], np.cumsum(legs)])
    out = []
    for d in np.linspace(0.0, at[-1], HOSE_N + 1):
        k = min(int(np.searchsorted(at, d, side="right")) - 1, len(legs) - 1)
        out.append([*(path[k] + (path[k + 1] - path[k]) * (d - at[k]) / legs[k]), HOSE_Z])
    return np.array(out)


def _hose() -> list:
    pts, out = hose_points(), []
    for k in range(HOSE_N):
        a, b = pts[k], pts[k + 1]
        d = b - a
        out.append(Body.build(f"hose{k}", pos=tuple(((a + b) / 2).tolist()), mass=HOSE_MASS,
                              shapes=[CylShape((0.0, 0.0, 0.0), HOSE_R, float(np.linalg.norm(d)), HOSE_GREY, axis="X")],
                              yaw=math.atan2(d[1], d[0]), color=HOSE_GREY, friction=(0.6, 0.5), damping=(0.05, 0.1),
                              contact_offset=0.001, solver_iterations=32))
    return out


ARM, MONITOR = _arm(), _monitor()
BUTTON_BODIES = [_button(n) for n in BUTTONS]
BUTTON_NAMES = [b.name for b in BUTTON_BODIES]
CUFF = [_strip(i) for i in range(CUFF_N)]
HOSE = _hose()
OBJECTS = [ARM, MONITOR, *BUTTON_BODIES, *CUFF, *HOSE]
PORT_STRIP = strip_of(PORT_S)
CUFF_NAMES = [f"cuff{i}" for i in range(CUFF_N)]

# -- joints ------------------------------------------------------------------------------------------------------------
# Strip i and strip i + 1 meet at a PhysX D6 joint at their shared edge: a hinge about the strip's x (across the cuff),
# sprung toward a slight curl (the ends' strong curl at CURLED); inside the bladder (PUMP_BENDS) it can also close up
# along the band by up to SQUEEZE, held open by a drive until the monitor inflates the cuff.
Bend = namedtuple("Bend", "name body0 body1 spring target squeeze")
BENDS = [Bend(f"bend{i}", f"cuff{i}", f"cuff{i + 1}", CURL_SPRING if i in CURLED else BEND_SPRING,
              CURL_TARGET if i in CURLED else BEND_TARGET, i in PUMP_BENDS) for i in range(CUFF_N - 1)]
PUMP_NAMES = [b.name for b in BENDS if b.squeeze]
for _name in BUTTONS:
    slider(_name, "monitor", f"{_name}_button", anchor=button_pos(_name), axis="Z", limits=(-BUTTON_TRAVEL, 0.0),
           stiffness=BUTTON_K, damping=3.0, target=0.0, max_force=200.0)
STICKS = tuple((k, j) for k in TAIL for j in UNDER)        # the hook strip on strip k sticking to strip j's loop face


def _stick_anchor(k: int) -> np.ndarray:
    return np.array([CUFF_X, strip_y(k), CUFF_Z])


for _k, _j in STICKS:
    glue(f"stick_{_k}_{_j}", f"cuff{_j}", f"cuff{_k}", anchor=tuple(_stick_anchor(_k).tolist()))


# A flexible joint between two bodies that may start turned (the hose's pieces): a PhysX D6 joint, its slides locked,
# free to bend and twist every way against a spring toward the starting angle (bend, N m per degree) and damping (damp,
# N m s per degree). The anchor is in the world at the start; body0 comes first in OBJECTS.
Link = namedtuple("Link", "name body0 body1 anchor bend damp")


def _start_rotation(spec: Body) -> np.ndarray:
    q = spec.start_xyzw if spec.start_xyzw is not None else (0.0, 0.0, math.sin(spec.yaw / 2), math.cos(spec.yaw / 2))
    return rotation(q)


def link_frames(link: Link, specs: dict) -> tuple:
    """(local position, local rotation as a matrix) of the joint frame in body0 and in body1: the frame sits at the
    anchor, lined up with the world, at the start."""
    out = []
    for name in (link.body0, link.body1):
        spec = specs[name]
        r = _start_rotation(spec)
        out += [r.T @ (np.asarray(link.anchor, float) - np.asarray(spec.pos, float)), r.T]
    return tuple(out)


LINKS = []
_pts = hose_points()
LINKS.append(Link("hose_port", f"cuff{PORT_STRIP}", "hose0", tuple(_pts[0].tolist()), HOSE_BEND, HOSE_DAMP))
for _k in range(1, HOSE_N):
    LINKS.append(Link(f"hose_{_k}", f"hose{_k - 1}", f"hose{_k}", tuple(_pts[_k].tolist()), HOSE_BEND, HOSE_DAMP))
LINKS.append(Link("hose_socket", "monitor", f"hose{HOSE_N - 1}", tuple(_pts[-1].tolist()), HOSE_BEND, HOSE_DAMP))
SPECS = {o.name: o for o in OBJECTS}


def _quatf(m: np.ndarray):
    from pxr import Gf

    x, y, z, w = bodykit._quat_from_matrix(m)
    return Gf.Quatf(w, x, y, z)


def _author_links(stage, prim_path: str, spec) -> None:
    from pxr import Gf, UsdPhysics

    parent = prim_path.rsplit("/", 1)[0]
    for link in LINKS:
        if link.body1 != spec.name:
            continue
        p0, r0, p1, r1 = link_frames(link, SPECS)
        joint = UsdPhysics.Joint.Define(stage, f"{prim_path}/{link.name}")
        joint.CreateBody0Rel().SetTargets([f"{parent}/{link.body0}/body"])
        joint.CreateBody1Rel().SetTargets([f"{prim_path}/body"])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in p0]))
        joint.CreateLocalRot0Attr(_quatf(r0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*[float(v) for v in p1]))
        joint.CreateLocalRot1Attr(_quatf(r1))
        joint.CreateCollisionEnabledAttr(False)
        joint.CreateExcludeFromArticulationAttr(True)
        prim = joint.GetPrim()
        for dof in ("transX", "transY", "transZ"):
            limit = UsdPhysics.LimitAPI.Apply(prim, dof)  # low > high: locked
            limit.CreateLowAttr(1.0)
            limit.CreateHighAttr(-1.0)
        for dof in ("rotX", "rotY", "rotZ"):
            drive = UsdPhysics.DriveAPI.Apply(prim, dof)  # USD angular drives take degrees
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(float(link.bend))
            drive.CreateDampingAttr(float(link.damp))
            drive.CreateTargetPositionAttr(0.0)
            drive.CreateMaxForceAttr(1.0e6)


_BEND_PATHS: dict = {}
_SQUEEZE_ATTRS: dict = {}


def _lock(prim, dof: str, low: float = 1.0, high: float = -1.0) -> None:
    from pxr import UsdPhysics

    limit = UsdPhysics.LimitAPI.Apply(prim, dof)  # low > high: locked
    limit.CreateLowAttr(float(low))
    limit.CreateHighAttr(float(high))


def _drive(prim, dof: str, stiffness: float, damping: float, target: float, max_force: float) -> None:
    from pxr import UsdPhysics

    drive = UsdPhysics.DriveAPI.Apply(prim, dof)  # USD angular drives take degrees
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(float(stiffness))
    drive.CreateDampingAttr(float(damping))
    drive.CreateTargetPositionAttr(float(target))
    drive.CreateMaxForceAttr(float(max_force))


def _author_bends(stage, prim_path: str, spec) -> None:
    from pxr import Gf, UsdPhysics

    parent = prim_path.rsplit("/", 1)[0]
    for b in BENDS:
        if b.body1 != spec.name:
            continue
        joint = UsdPhysics.Joint.Define(stage, f"{prim_path}/{b.name}")
        joint.CreateBody0Rel().SetTargets([f"{parent}/{b.body0}/body"])
        joint.CreateBody1Rel().SetTargets([f"{prim_path}/body"])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, SEG / 2, 0.0))       # the strips start side by side, unrotated
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, -SEG / 2, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateCollisionEnabledAttr(False)
        joint.CreateExcludeFromArticulationAttr(True)
        prim = joint.GetPrim()
        for dof in ("transX", "transZ", "rotY", "rotZ"):
            _lock(prim, dof)
        _lock(prim, "rotX", -BEND_LIMIT, BEND_LIMIT)
        _drive(prim, "rotX", b.spring, BEND_DAMP, b.target, 1.0e6)
        if b.squeeze:
            _lock(prim, "transY", -SQUEEZE, 0.0)
            _drive(prim, "transY", PUMP_K, PUMP_D, 0.0, PUMP_N)
        else:
            _lock(prim, "transY")
        _BEND_PATHS[b.name] = f"{prim_path}/{b.name}"
        _SQUEEZE_ATTRS.pop(b.name, None)


def squeeze_target(since: float | None) -> float:
    """Where the bladder joints are driven (m along the band; negative closes them up), ``since`` seconds after the
    monitor started inflating (None: not inflating)."""
    if since is None:
        return 0.0
    return -SQUEEZE * min(max(since / PUMP_S, 0.0), 1.0)


def _set_squeeze(target: float) -> None:
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import UsdPhysics

    for name in PUMP_NAMES:
        if name not in _BEND_PATHS:
            return
        if name not in _SQUEEZE_ATTRS:
            prim = get_current_stage().GetPrimAtPath(_BEND_PATHS[name])
            _SQUEEZE_ATTRS[name] = UsdPhysics.DriveAPI(prim, "transY").GetTargetPositionAttr()
        _SQUEEZE_ATTRS[name].Set(float(target))


_PATHS: dict = {}                                          # each cuff strip's rigid-body prim


def _author_contacts(stage, prim_path: str, spec) -> None:
    """The cuff strips report their contacts (for the hook-and-loop rule and for touches by the robot)."""
    if spec.name in CUFF_NAMES:
        from pxr import PhysxSchema

        prim = stage.GetPrimAtPath(f"{prim_path}/body")
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        _PATHS[spec.name] = f"{prim_path}/body"


if not getattr(bodykit, "_bp_links", False):                # bodykit authors a body's joints as it spawns it
    _plain_sliders = bodykit._author_sliders

    def _author_sliders_and_links(stage, prim_path: str, spec) -> None:
        _plain_sliders(stage, prim_path, spec)
        _author_bends(stage, prim_path, spec)
        _author_links(stage, prim_path, spec)
        _author_contacts(stage, prim_path, spec)

    bodykit._author_sliders = _author_sliders_and_links
    bodykit._bp_links = True


# -- geometry (pure; tested offline) -----------------------------------------------------------------------------------

def world(state, local) -> np.ndarray:
    """A point given in a body's frame, in the world."""
    return np.asarray(state.pos, float) + rotation(state.quat_xyzw) @ np.asarray(local, float)


def around(p) -> tuple[float, float, float]:
    """(distance from the elbow crease up the arm toward the shoulder, angle round the arm from the top toward the body
    (-y) in degrees, distance from its axis) of a world point."""
    d = np.asarray(p, float) - np.array([X_CREASE, 0.0, ARM_Z])
    return float(-d[0]), math.degrees(math.atan2(-d[1], d[2])), float(math.hypot(d[1], d[2]))


def strip_local(s: float, across: float = 0.0, face: float = 0.0) -> tuple[int, np.ndarray]:
    """(strip, point in its frame) of the cuff point at length s, ``across`` from the middle of its width toward the
    lower edge, ``face`` out of the middle plane (+ the inner face)."""
    i = strip_of(s)
    return i, np.array([across, s - i * SEG - SEG / 2, face])


def cuff_point(cuff: list, s: float, across: float = 0.0, face: float = 0.0) -> np.ndarray:
    i, local = strip_local(s, across, face)
    return world(cuff[i], local)


def marker_point(cuff: list) -> np.ndarray:
    return cuff_point(cuff, MARKER_S, 0.0, -CUFF_T / 2)


def tube_point(cuff: list) -> np.ndarray:
    """The tube's end, where the hose is joined."""
    return cuff_point(cuff, PORT_S, CUFF_W / 2 + STUB_L, STUB_Z)


def button_travel(monitor, button) -> float:
    """How far (m) a button is pushed in from where it rests."""
    rest = BUTTON_Z - (TOP + MON_H / 2 + 0.0005)
    return float(rest - (rotation(monitor.quat_xyzw).T @ (np.asarray(button.pos, float)
                                                         - np.asarray(monitor.pos, float)))[2])


def winding(cuff: list) -> float:
    """Turns the cuff makes round the upper arm's axis from its right end to its left end (its strips' middles, seen
    along the arm; either way round)."""
    ang = [math.atan2(-(float(c.pos[1])), float(c.pos[2]) - ARM_Z) for c in cuff]
    total = 0.0
    for a, b in zip(ang, ang[1:]):
        total += (b - a + math.pi) % (2 * math.pi) - math.pi
    return abs(total) / (2 * math.pi)


def skin_gap(c) -> float:
    """How far (m) a strip's middle is from the upper arm's skin."""
    return around(c.pos)[2] - upper_arm_radius(float(c.pos[0]))


def on_arm(cuff: list) -> bool:
    """Wrapped round the upper arm: at least one full turn about its axis, every strip's middle within WRAP_GAP of its
    skin and between the shoulder and the elbow crease."""
    for c in cuff:
        if not (X_SHOULDER < float(c.pos[0]) < X_CREASE) or skin_gap(c) > WRAP_GAP:
            return False
    return winding(cuff) >= 1.0


def tight(cuff: list, stuck: dict) -> bool:
    """Drawn tight round the arm: every strip of the closed loop (up to the last stuck hook strip; the free end past it
    is a flap) within TIGHT_GAP of the skin."""
    last = max(stuck) if stuck else CUFF_N - 1
    return all(skin_gap(c) <= TIGHT_GAP for c in cuff[:last + 1])


def edge_heights(cuff: list) -> list:
    """Each strip's lower edge (its middle), as the distance above the elbow crease up the arm (m)."""
    return [around(world(c, (CUFF_W / 2, 0.0, 0.0)))[0] for c in cuff]


def facing(hook, loop) -> bool:
    """The hook strip's inner face lies on the other strip's outer (loop) face: the two flat to each other, the hook
    strip on the outer side."""
    rl, rh = rotation(loop.quat_xyzw), rotation(hook.quat_xyzw)
    if float(rl[:, 2] @ rh[:, 2]) < math.cos(math.radians(FACE_DEG)):
        return False
    d = np.asarray(hook.pos, float) - np.asarray(loop.pos, float)
    inward = (rl[:, 2] + rh[:, 2]) / np.linalg.norm(rl[:, 2] + rh[:, 2])     # the two inner faces' mean direction
    local = rl.T @ d
    return float(d @ inward) < -0.001 and abs(float(local[1])) < SEG and abs(float(local[0])) < 0.03


# -- what happens during a trial ---------------------------------------------------------------------------------------

_STATE: dict = {}
_VIEWS: dict = {"views": None, "err": ""}


def _clear() -> None:
    _STATE.update(steps=0, t=0.0, stuck={}, inflated_at=None, start_at=None, down=dict.fromkeys(BUTTONS, False),
                  touched=False, fpair=0.0, ftouch=0.0, toucher="")


_clear()


def _quat_wxyz(m: np.ndarray):
    x, y, z, w = bodykit._quat_from_matrix(m)
    return w, x, y, z


def _set_frames(name: str, pos0, rot0, pos1, rot1) -> None:
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, UsdPhysics

    joint = UsdPhysics.FixedJoint(get_current_stage().GetPrimAtPath(bodykit._GLUE_PATHS[name]))
    joint.GetLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in pos0]))
    joint.GetLocalRot0Attr().Set(Gf.Quatf(*_quat_wxyz(rot0)))
    joint.GetLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in pos1]))
    joint.GetLocalRot1Attr().Set(Gf.Quatf(*_quat_wxyz(rot1)))


@on_reset
def _new_episode() -> None:
    _clear()
    _set_squeeze(0.0)                                       # the bladder let down
    for k, j in STICKS:                                     # back to the frames they were authored with, and unstuck
        name = f"stick_{k}_{j}"
        a = _stick_anchor(k)
        _set_frames(name, a - np.asarray(CUFF[j].pos), np.eye(3), a - np.asarray(CUFF[k].pos), np.eye(3))
        set_glue_on(name, False)


def _np(a) -> np.ndarray:
    if hasattr(a, "numpy") and not hasattr(a, "detach"):
        return np.asarray(a.numpy(), float)
    return as_numpy(a)


def _views():
    """PhysX rigid-contact views, made on the first control step: (1) each hook strip against each strip it can land
    on, (2) every cuff strip against every link of both robot arms."""
    if _VIEWS["views"] is None and not _VIEWS["err"] and len(_PATHS) == CUFF_N:
        try:
            from isaaclab_physx.physics import PhysxManager
            from yam_arena.robot import asset_link_paths

            sim = PhysxManager.get_physics_sim_view()
            env = _PATHS["cuff0"].rsplit("/", 2)[0]
            pair = sim.create_rigid_contact_view([_PATHS[f"cuff{k}"] for k in TAIL],
                                                 filter_patterns=[[_PATHS[f"cuff{j}"] for j in UNDER] for _ in TAIL],
                                                 max_contact_data_count=0)
            links = [f"{env}/{arm}{p}" for arm in ("LeftArm", "RightArm") for p in asset_link_paths().values()]
            hands = sim.create_rigid_contact_view([_PATHS[n] for n in CUFF_NAMES], filter_patterns=[links] * CUFF_N,
                                                  max_contact_data_count=0)
            _VIEWS["views"] = (pair, hands)
        except Exception as e:  # noqa: BLE001
            _VIEWS["err"] = repr(e)[:300]
    return _VIEWS["views"]


def _contacts():
    """(force (N) between each hook strip and each strip it can land on, as a len(TAIL) x len(UNDER) array; the largest
    force between any robot link and any cuff strip; which strip) or None."""
    v = _views()
    if v is None:
        return None
    pair, hands = v
    p = np.linalg.norm(_np(pair.get_contact_force_matrix(dt=1.0 / 240.0)).reshape(len(TAIL), len(UNDER), 3), axis=2)
    h = np.linalg.norm(_np(hands.get_contact_force_matrix(dt=1.0 / 240.0)).reshape(CUFF_N, -1, 3), axis=2).max(axis=1)
    i = int(np.argmax(h))
    return p, float(h[i]), CUFF_NAMES[i]


def _stick(k: int, j: int, hook, loop) -> None:
    """Hook strip k sticks to strip j where they lie now: the glue between them re-anchored at their present poses."""
    a = (np.asarray(hook.pos, float) + np.asarray(loop.pos, float)) / 2
    rl, rh = rotation(loop.quat_xyzw), rotation(hook.quat_xyzw)
    name = f"stick_{k}_{j}"
    _set_frames(name, rl.T @ (a - np.asarray(loop.pos, float)), rl.T, rh.T @ (a - np.asarray(hook.pos, float)), rh.T)
    set_glue_on(name, True)


def _step_dt(environment) -> float:
    for e in (environment, getattr(environment, "unwrapped", None)):
        dt = getattr(e, "step_dt", None)
        if isinstance(dt, (int, float)) and dt > 0:
            return float(dt)
    return 0.05


def _monitor_rules(down: dict, ok: bool, ftouch: float) -> None:
    """The monitor, one control step: ``down`` says which buttons are pushed in now, ``ok`` whether the cuff is closed
    round the arm, ``ftouch`` the largest force (N) of a robot link on the cuff. INFLATE pressed (a fresh press) with
    the cuff closed round the arm starts pumping it up; START pressed after that, with it still closed, takes a reading,
    which a touch on the cuff in its first READ_S seconds spoils."""
    s = _STATE
    fresh = {n: down[n] and not s["down"][n] for n in BUTTONS}
    if fresh["inflate"] and ok and s["inflated_at"] is None:
        s["inflated_at"] = s["t"]
    if fresh["start"] and ok and s["inflated_at"] is not None:
        s["start_at"], s["touched"] = s["t"], False
    if s["start_at"] is not None and s["t"] - s["start_at"] <= READ_S and ftouch > TOUCH_N:
        s["touched"] = True
    s["down"] = dict(down)


@on_step
def _follow(environment) -> None:
    from yam_arena.scene import object_states  # importable only once the simulator has started

    s = _STATE
    s["steps"] += 1
    s["t"] += _step_dt(environment)
    st = object_states(environment, ["monitor", *BUTTON_NAMES, *CUFF_NAMES])
    cuff = [st[n] for n in CUFF_NAMES]
    c = _contacts()
    if c is not None:
        p, s["ftouch"], s["toucher"] = c
        s["fpair"] = float(p.max())
        for a, k in enumerate(TAIL):                        # hook pressed flat onto loop: it sticks, for good
            if k in s["stuck"]:
                continue
            for b, j in enumerate(UNDER):
                if p[a, b] >= PRESS_N and facing(cuff[k], cuff[j]):
                    _stick(k, j, cuff[k], cuff[j])
                    s["stuck"][k] = (j, s["steps"])
                    break
    _monitor_rules({n: button_travel(st["monitor"], st[f"{n}_button"]) >= PRESSED for n in BUTTONS}, closed(cuff),
                   s["ftouch"])
    if s["inflated_at"] is not None and s["t"] - s["inflated_at"] <= PUMP_S + 0.2:    # the bladder filling
        _set_squeeze(squeeze_target(s["t"] - s["inflated_at"]))
    if os.environ.get("BP_DEBUG") and s["steps"] % 10 == 0:
        _debug(cuff, st)


def closed(cuff: list) -> bool:
    return len(_STATE["stuck"]) >= STICK_MIN and on_arm(cuff)


def parts(objects) -> dict:
    """Which parts of the score the final state earns."""
    cuff = [objects[n] for n in CUFF_NAMES]
    mon = objects["monitor"]
    out = dict.fromkeys(POINTS, False)
    upright = rotation(mon.quat_xyzw)[2, 2] > math.cos(math.radians(30.0))
    if float(mon.pos[2]) < TOP or not upright or not on_arm(cuff):  # the monitor knocked over or off: nothing
        return out
    edges = edge_heights(cuff)
    s = _STATE
    out["height"] = EDGE_ABOVE[0] <= min(edges) and max(edges) <= EDGE_ABOVE[1]
    out["closed"] = closed(cuff)
    out["inflated"] = out["closed"] and s["inflated_at"] is not None and tight(cuff, s["stuck"])
    out["reading"] = (out["inflated"] and s["start_at"] is not None and s["t"] - s["start_at"] >= READ_S
                      and not s["touched"])
    return out


def score(objects) -> float:
    """0.40 closed round the upper arm, 0.20 its lower edge at the right height, 0.20 inflated once closed and drawn
    tight on the arm, 0.20 a reading taken (START pressed once inflated and the cuff left alone for 2 s); nothing if it
    is not round the arm or the monitor is knocked over or off the table."""
    got = parts(objects)
    return round(sum(POINTS[k] for k, v in got.items() if v), 4)


def success(objects) -> bool:
    return score(objects) >= 1.0


def _phase_label(t: float) -> str:
    for seconds, name, _, _ in demo_phases():
        if t < seconds:
            return name
        t -= seconds
    return "after the demo"


def _debug(cuff, st: dict) -> None:
    """A line on stderr every 10 control steps (BP_DEBUG=1), and each strip's place every 40."""
    s = _STATE
    steps, mon = s["steps"], st["monitor"]
    _, deg, _ = around(marker_point(cuff))
    edges = edge_heights(cuff)
    vmax = max(float(np.linalg.norm(c.lin_vel)) for c in cuff)
    stuck = ",".join(f"{k}>{j}@{t}" for k, (j, t) in sorted(s["stuck"].items())) or "-"
    got = parts(st)
    travel = " ".join(f"{n} {1000 * button_travel(mon, st[f'{n}_button']):.1f}mm" for n in BUTTONS)
    times = " ".join(f"{k} {'-' if s[k] is None else round(s[k], 2)}" for k in ("inflated_at", "start_at"))
    print(f"[bp] {steps} t {s['t']:.2f} {_phase_label(s['t'])} | turns {winding(cuff):.2f} on {int(on_arm(cuff))} "
          f"max-gap {100 * max(skin_gap(c) for c in cuff):.1f}cm tight {int(tight(cuff, s['stuck']))} "
          f"squeeze {100 * squeeze_target(None if s['inflated_at'] is None else s['t'] - s['inflated_at']):.1f}cm "
          f"| stuck {stuck} | fpair {s['fpair']:.1f} touch "
          f"{s['ftouch']:.1f} {s['toucher']} | arrow {deg:.0f}deg | edge {100 * min(edges):.1f}-{100 * max(edges):.1f}cm "
          f"| {travel} | {times} touched {int(s['touched'])} | monitor {','.join(f'{100 * v:.1f}' for v in mon.pos)} "
          f"| vmax {vmax:.2f} | score {sum(POINTS[k] for k, v in got.items() if v):.2f} "
          f"{','.join(k for k, v in got.items() if v)} {_VIEWS['err']}", file=sys.stderr, flush=True)
    if steps % 40 == 0:
        shape = " ".join(f"{i}:{100 * float(c.pos[1]):.1f},{100 * (float(c.pos[2]) - TOP):.1f},{100 * skin_gap(c):.1f},"
                         f"{math.degrees(math.atan2(rotation(c.quat_xyzw)[2, 1], rotation(c.quat_xyzw)[1, 1])):.0f}"
                         for i, c in enumerate(cuff))
        print(f"[bp-shape] {steps} y,z cm above table,gap cm,tilt deg: {shape} | x {100 * float(cuff[0].pos[0]):.1f}",
              file=sys.stderr, flush=True)


# -- scripted demo ---------------------------------------------------------------------------------------------------
# Both hands pinch a curled end (fingers closing across the strip). The right hand takes the bladder end up the body side
# of the arm, over the top and down the outer side, and lets go: the end's curl closes it round the arm. The left hand
# takes the hook end up the outer side, over the top and down the body side, which lays the hook strip on the loop face,
# lets go, lifts clear of the end with its fingers open and presses the hook strip down on the top. The right hand
# presses INFLATE, waits while the bladder draws the cuff tight, presses START, and both hands stay off while the monitor
# reads.

CURL_TILT = (43.0, 87.0)        # the settled tilt (deg) of each end's second-last and last strip (from a statics model)
GRIP_DOWN = 0.022               # the pad tips this far below a curled end's top edge
XC = X_CREASE - 0.025 - CUFF_W / 2      # the cuff's middle once moved down the arm: its lower edge 2.5 cm above the crease
HOVER = TOP + 0.130
OVER = TOP + 0.170              # the tool point carrying an end over the top of the arm
SIDE_OUT = 0.075                # the tool point's distance from the arm's axis, beside it, where it lets an end go
SIDE_Z = ARM_Z + 0.040
LET_GO = -0.010                 # the right fingers opened just 2 cm beside the arm
RELEASE = -0.016                # the left fingers opened 3.2 cm, then kept open until the hand is above the arm
CLEAR_Z = SIDE_Z + 0.050        # the left hand lifts this high before it moves over the arm: its pads above the hook end
PRESS_IN = 0.003                # pressing the closed fingers this far into the cuff on the arm
BUTTON_IN = 0.006               # the pad tips this far below a button's top (it stops at 3 mm: the rest allows for sag)
SLACK = 0.030                   # how far the left hand brings the hook end in while the right hand wraps
PARK_R = (0.120, -0.200, TOP + 0.150)
PARK_L = (0.120, 0.200, TOP + 0.150)


def curled_end(side: int) -> tuple[float, float]:
    """(y, z) of the top edge of the cuff's curled end, side -1 the right end, +1 the left end, as it settles."""
    y = CUFF_Y0 + 2 * SEG if side < 0 else CUFF_Y0 + CUFF_L - 2 * SEG
    z = CUFF_Z
    for tilt in CURL_TILT:
        y += side * SEG * math.cos(math.radians(tilt))
        z += SEG * math.sin(math.radians(tilt))
    return y, z


def demo_phases() -> list:
    """(seconds, label, left target, right target): a target is (x, y, tool point z, finger) or None (stay)."""
    o, c = FINGER_OPEN, FINGER_CLOSED
    yr, zr = curled_end(-1)
    yl, zl = curled_end(+1)
    gr, gl = zr - GRIP_DOWN + PAD_TIP_PAST_TCP, zl - GRIP_DOWN + PAD_TIP_PAST_TCP
    top = ARM_Z + upper_arm_radius(XC) + 2 * CUFF_T          # the top of the arm with both layers of the cuff on it
    press_top = top - PRESS_IN + PAD_TIP_PAST_TCP
    press = BUTTON_Z + BUTTON_H / 2 - BUTTON_IN + PAD_TIP_PAST_TCP
    ix, iy, _ = button_pos("inflate")
    sx, sy, _ = button_pos("start")
    lift = press + 0.030
    return [
        (3.0, "settle, then over the curled ends", (CUFF_X, yl, HOVER, o), (CUFF_X, yr, HOVER, o)),
        (1.5, "down round the ends", (CUFF_X, yl, gl, o), (CUFF_X, yr, gr, o)),
        (0.8, "pinch the ends", (CUFF_X, yl, gl, c), (CUFF_X, yr, gr, c)),
        (2.0, "lift them and move the cuff down the arm", (XC, yl - SLACK, gl + 0.02, c), (XC, yr + 0.02, HOVER, c)),
        (2.0, "right: the bladder end up the body side and over", None, (XC, -0.020, OVER, c)),
        (1.5, "right: down the outer side", None, (XC, SIDE_OUT, SIDE_Z, c)),
        (0.8, "right: let go", None, (XC, SIDE_OUT, SIDE_Z, LET_GO)),
        (1.0, "right: up", None, (XC, SIDE_OUT - 0.010, SIDE_Z + 0.035, LET_GO)),
        (1.5, "right: out of the way", None, (*PARK_R, o)),
        (2.0, "left: the hook end up the outer side", (XC, 0.090, OVER, c), None),
        (2.0, "left: over the top", (XC, -0.010, OVER, c), None),
        (1.5, "left: down the body side, snug", (XC, -SIDE_OUT, SIDE_Z, c), None),
        (0.8, "left: let go", (XC, -SIDE_OUT, SIDE_Z, RELEASE), None),
        (1.0, "left: up off the hook end", (XC, -SIDE_OUT + 0.005, CLEAR_Z, RELEASE), None),
        (1.2, "left: over the top of the arm, fingers open", (XC, -0.005, OVER, RELEASE), None),
        (0.4, "left: close the fingers", (XC, -0.005, OVER, c), None),
        (1.2, "left: press the hook strip on the top", (XC, 0.0, press_top, c), None),
        (1.0, "left: hold the press", (XC, 0.0, press_top, c), None),
        (0.8, "left: up", (XC, 0.0, OVER, c), None),
        (1.5, "left: out of the way", (*PARK_L, c), (ix, iy, HOVER, c)),
        (1.2, "right: press inflate", None, (ix, iy, press, c)),
        (0.8, "right: hold inflate", None, (ix, iy, press, c)),
        (0.8, "right: up from inflate", None, (ix, iy, lift, c)),
        (1.5, "right: wait while the cuff inflates", None, (ix, iy, lift, c)),
        (0.8, "right: over start", None, (sx, sy, lift, c)),
        (1.2, "right: press start", None, (sx, sy, press, c)),
        (0.8, "right: hold start", None, (sx, sy, press, c)),
        (1.0, "right: up from start", None, (sx, sy, HOVER, c)),
        (1.5, "right: out of the way, hands off", None, (*PARK_R, c)),
        (4.0, "hands off while it reads", None, None),
    ]


def demo():
    plans = {"left": [], "right": []}
    last = {"left": None, "right": None}
    for seconds, label, left, right in demo_phases():
        for arm, target in (("left", left), ("right", right)):
            target = target if target is not None else last[arm]
            last[arm] = target
            x, y, z, finger = target
            plans[arm].append(Waypoint(seconds, (float(x), float(y), float(z)), float(finger), label, 0.0))
    return plans
