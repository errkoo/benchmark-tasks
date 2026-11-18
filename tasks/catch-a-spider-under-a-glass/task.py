"""On the station table, used as a desk pushed against the wall under a wide window that stands open: a spider stand-in
sitting on the desk, an empty drinking glass standing upside down, and a stiff card lying flat.

- The window: the wall's inner face runs across the desk 28.5 cm in front of the arm bases. The opening is 90 cm wide
  and its frame's sill is 3 cm deep and 2 cm above the desk. Outside are a stone outer sill and a window box.
- The spider: a live spider can't be used in repeatable trials, so the stand-in is a small off-the-shelf robot dressed
  as one: an Ozobot Evo (Titanium Black, 3.35 cm across, 3.3 cm tall, 28 g) with eight black pipe-cleaner legs. It runs
  a short program (README): it backs away from anything its two front proximity sensors see within about 5 cm, drives
  forward from anything its two back sensors see, at 8 cm/s, and otherwise sits still. Its program is all that drives
  it; the glass and the card only push it, as they would the real one.
- The glass: a straight 250 mL tumbler, 7.2 cm across and 9 cm tall, 0.22 kg, upside down.
- The card: a 1.5 mm greyboard card, 25 x 15 cm, one short edge (on the right) folded up 2 cm (a doubled strip).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    the same with --policy yam_scripted (plays demo() below).
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, SphereShape, as_numpy, on_reset, on_step, rotation, segment  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP, TCP_OFFSET  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
TOP = TABLE_TOP[SCENE]

INSTRUCTION = "Catch the spider on the desk and put it out of the open window, without touching or hurting it."

# -- layout (world frame: x forward from the arm bases, y left, z up; metres) -------------------------------------------
X_IN, X_OUT = 0.285, 0.315              # the wall's inner and outer faces (the window frame's sill spans them)
SILL_H = 0.020                          # the sill's top above the desk
OPENING = 0.45                          # the opening runs from y -0.45 to +0.45
WALL_TOP, WALL_END = 0.50, 0.62         # the wall either side: this high above the desk, out to |y| 0.62
SLAB_X1, SLAB_H = 0.44, 0.012           # outside: a stone outer sill from X_OUT to x 0.44, 1.2 cm thick
PLANTER_X, PLANTER_HALF_Y, PLANTER_H = (0.45, 0.65), 0.32, 0.09   # and a window box beyond it

# the spider stand-in: an Ozobot Evo dressed as a spider, facing +x (the window)
SPIDER_XY = (0.200, 0.060)
BOT_R, BOT_H = 0.01675, 0.033           # 33.5 mm across, 33 mm tall
FOOT_R, FOOT_AT = 0.0045, 0.012         # it stands on four round contacts (its two wheels and two skids)
SHELL_Z0 = 0.0035                       # the shell's lower edge, this high off the floor it stands on
SPIDER_MASS = 0.028
SPIDER_COM = (0.0, 0.0, 0.008)          # battery and motors sit low in the shell
SPIDER_FRICTION = (0.30, 0.25)          # slick (the lower friction of any pair counts): wheels roll, the shell is smooth
LEG_REACH = 0.010                       # its pipe-cleaner legs reach this far out past the shell, clear of the floor

GLASS_XY = (0.170, 0.240)
GLASS_R, GLASS_WALL, GLASS_H, GLASS_BASE = 0.036, 0.0035, 0.090, 0.008
GLASS_R_IN = GLASS_R - GLASS_WALL
GLASS_MASS = 0.22

CARD_XY = (0.190, -0.140)               # middle of the flat face
CARD_X, CARD_Y, CARD_T = 0.150, 0.250, 0.0015
LIP_T, LIP_H = 0.003, 0.020             # the folded-up strip along its right-hand short edge (-y): doubled card
CARD_MASS = 0.040

# -- the stand-in's program (README: "The spider stand-in") ------------------------------------------------------------
SENSE_RANGE = 0.05                      # its infrared proximity sensors see about this far (about 2 in)
SENSE_HALF = math.radians(40.0)         # ... within this angle either side of straight ahead (front pair) or behind
SENSE_TOP = 0.035                       # ... from the floor it stands on up to this height
DRIVE = 0.08                            # m/s: the speed its program drives at
DRIVE_FF = SPIDER_FRICTION[1] * SPIDER_MASS * 9.81   # N: its wheels' force that balances the model's sliding friction
DRIVE_K = 0.5                           # N per m/s: and the force that brings it to the program's speed
DRIVE_F_MAX = 0.30                      # N: at most about its weight (rubber wheels)

# -- scoring -----------------------------------------------------------------------------------------------------------
TOUCH = 0.010           # a hand point this close to the shell of a stand-in that is not shut in touches it (its legs)
TRAP_GAP = 0.008        # shut in: inside the glass, its rim at most this far above whatever is under it, all round
SEAL_GAP = 0.004        # card on: the whole rim over the card's face, at most this far above it
LIFTED = 0.030          # lifted: the card's underside this far above the desk
OUT_PAST = 0.010        # out: the stand-in's centre this far past the wall's outer face, not shut in
DESK_X0, DESK_Y = 0.05, 0.50            # escaped: its centre gets behind x 0.05 or past |y| 0.50 (inside the room)
FELL = 0.030            # the glass's rim this far below the desk top: it fell off the desk or out of the window
UPRIGHT = math.radians(25.0)            # the glass counts as standing on its rim within this tilt
R_MID = GLASS_R - GLASS_WALL / 2
SQUASH = GLASS_WALL / 2 + BOT_R - 0.004     # the rim's middle line this close (horizontally) to the stand-in's centre
RIM = np.array([[R_MID * math.cos(a), R_MID * math.sin(a), 0.0] for a in np.linspace(0.0, 2 * math.pi, 48,
                                                                                         endpoint=False)])

# -- objects -----------------------------------------------------------------------------------------------------------
FRAME, WALL, STONE, POT, SOIL, LEAF = (0.95, 0.95, 0.93), (0.90, 0.87, 0.80), (0.62, 0.60, 0.57), (0.72, 0.38, 0.24), \
    (0.25, 0.17, 0.11), (0.25, 0.50, 0.20)
PANE = (0.72, 0.84, 0.90)
GLASS_EDGE, GLASS_GLINT = (0.78, 0.87, 0.90), (0.90, 0.95, 0.97)
SHELL, BROWN, DARK, LEG = (0.11, 0.11, 0.12), (0.33, 0.22, 0.13), (0.20, 0.13, 0.08), (0.05, 0.05, 0.05)
GREY_CARD, PRINT = (0.62, 0.61, 0.58), (0.30, 0.52, 0.70)


def _ring(r_mid, thick, z0, h, n, color, collide=True):
    """``n`` flat boxes round the z axis: the wall of a cylinder (the first faces +x, so two face +-x and two +-y)."""
    chord = 2 * (r_mid + thick / 2) * math.tan(math.pi / n) + 0.0005
    return [BoxShape((r_mid * math.cos(a), r_mid * math.sin(a), z0 + h / 2), (thick, chord, h), color, yaw=a,
                     collide=collide) for a in (2 * math.pi * i / n for i in range(n))]


def _spider() -> Body:
    """Origin under the middle of the robot, on the floor; it faces +x. It stands on four contact spheres; its shell is
    a ring of 12 flat walls and a lid (all collide). The dome, the spider's body on top and its legs are drawn only."""
    shapes = [SphereShape((FOOT_AT * math.cos(a), FOOT_AT * math.sin(a), FOOT_R), FOOT_R, None)
              for a in (0.0, math.pi / 2, math.pi, 1.5 * math.pi)]
    shapes += _ring(BOT_R - 0.001, 0.002, SHELL_Z0, BOT_H - SHELL_Z0 - 0.003, 12, SHELL)
    shapes += [BoxShape((0.0, 0.0, BOT_H - 0.0015), (0.022, 0.022, 0.003), SHELL),                     # the lid
               SphereShape((0.0, 0.0, BOT_H - 0.012), 0.0145, SHELL, collide=False),                   # the dome
               SphereShape((-0.004, 0.0, BOT_H + 0.0015), 0.0055, BROWN, collide=False),               # abdomen
               SphereShape((0.0055, 0.0, BOT_H + 0.001), 0.0035, DARK, collide=False)]                 # head
    for side in (1.0, -1.0):
        for deg in (50.0, 78.0, 104.0, 132.0):          # four a side, none in front of or behind the sensors
            a = math.radians(deg)
            c, s = math.cos(a), side * math.sin(a)
            root = (0.85 * BOT_R * c, 0.85 * BOT_R * s, BOT_H - 0.004)
            knee = ((BOT_R + 0.004) * c, (BOT_R + 0.004) * s, BOT_H + 0.003)
            foot = ((BOT_R + LEG_REACH) * c, (BOT_R + LEG_REACH) * s, 0.006)
            shapes += [segment(root, knee, 0.0014, LEG, collide=False), segment(knee, foot, 0.0012, LEG, collide=False)]
    return Body.build("spider", pos=(*SPIDER_XY, TOP + 0.0003), mass=SPIDER_MASS, shapes=shapes, color=SHELL,
                      friction=SPIDER_FRICTION, slick=True, damping=(0.2, 2.0), contact_offset=0.001, com=SPIDER_COM)


def _glass() -> Body:
    """Origin in the middle of the rim (the rim on the desk, its base 9 cm up). The walls and base collide but are not
    drawn, so the stand-in shows through; the rim, the base's edge and a few glints are drawn."""
    wall_h = GLASS_H - GLASS_BASE
    shapes = _ring(R_MID, GLASS_WALL, 0.0, wall_h, 24, None)
    shapes.append(BoxShape((0.0, 0.0, GLASS_H - GLASS_BASE / 2), (2 * GLASS_R_IN, 2 * GLASS_R_IN, GLASS_BASE), None))
    shapes += _ring(R_MID, GLASS_WALL, 0.0, 0.003, 24, GLASS_EDGE, collide=False)
    shapes += _ring(R_MID, GLASS_WALL, wall_h, GLASS_BASE, 24, GLASS_EDGE, collide=False)
    for i in range(6):
        a = 2 * math.pi * (i + 0.5) / 6
        shapes.append(BoxShape((R_MID * math.cos(a), R_MID * math.sin(a), GLASS_H / 2), (GLASS_WALL, 0.0015, GLASS_H),
                               GLASS_GLINT, yaw=a, collide=False))
    return Body.build("glass", pos=(*GLASS_XY, TOP + 0.0005), mass=GLASS_MASS, shapes=shapes, color=GLASS_EDGE,
                      friction=(0.5, 0.4))


def _card() -> Body:
    """Origin in the middle of the flat face's underside. The fold runs along the -y (right-hand) short edge."""
    shapes = [BoxShape((0.0, 0.0, CARD_T / 2), (CARD_X, CARD_Y, CARD_T), GREY_CARD),
              BoxShape((0.004, 0.0, CARD_T + 0.0002), (CARD_X - 0.020, CARD_Y - 0.012, 0.0004), PRINT, collide=False),
              BoxShape((0.0, -CARD_Y / 2 + LIP_T / 2, CARD_T + LIP_H / 2), (CARD_X, LIP_T, LIP_H), GREY_CARD)]
    return Body.build("card", pos=(*CARD_XY, TOP + 0.0003), mass=CARD_MASS, shapes=shapes, color=GREY_CARD,
                      friction=(0.45, 0.40), contact_offset=0.001)


def _window() -> Body:
    """The wall with its open window, the outer sill and the window box: one fixed body (origin on the desk top)."""
    xm, depth = (X_IN + X_OUT) / 2, X_OUT - X_IN
    shapes = [BoxShape((xm, 0.0, SILL_H / 2), (depth, 2 * OPENING, SILL_H), FRAME)]
    for side in (1.0, -1.0):
        shapes.append(BoxShape((xm, side * (OPENING + WALL_END) / 2, WALL_TOP / 2), (depth, WALL_END - OPENING,
                                                                                    WALL_TOP), WALL))
        shapes.append(BoxShape((xm, side * (OPENING + 0.012), (SILL_H + WALL_TOP) / 2),
                               (depth + 0.006, 0.024, WALL_TOP - SILL_H), FRAME, collide=False))      # the jambs
    # the casement, swung open outward on its hinge at the left jamb
    cx, cz, ch = X_OUT + 0.20, (SILL_H + WALL_TOP) / 2 + 0.01, WALL_TOP - SILL_H - 0.03
    shapes += [BoxShape((cx, OPENING - 0.03, cz), (0.40, 0.014, ch), FRAME, collide=False),
               BoxShape((cx, OPENING - 0.03, cz), (0.34, 0.016, ch - 0.06), PANE, collide=False)]
    shapes.append(BoxShape(((X_OUT + SLAB_X1) / 2, 0.0, SLAB_H / 2), (SLAB_X1 - X_OUT, 2 * OPENING + 0.10, SLAB_H),
                           STONE))
    px, pdx = (PLANTER_X[0] + PLANTER_X[1]) / 2, PLANTER_X[1] - PLANTER_X[0]
    shapes += [BoxShape((px, 0.0, PLANTER_H / 2), (pdx, 2 * PLANTER_HALF_Y, PLANTER_H), POT),
               BoxShape((px, 0.0, PLANTER_H + 0.0005), (pdx - 0.02, 2 * PLANTER_HALF_Y - 0.02, 0.001), SOIL,
                        collide=False)]
    for i, y in enumerate((-0.24, -0.12, 0.0, 0.12, 0.24)):
        x = px + (0.03 if i % 2 else -0.03)
        shapes += [SphereShape((x, y, PLANTER_H + 0.035), 0.040, LEAF, collide=False),
                   SphereShape((x + 0.02, y + 0.03, PLANTER_H + 0.06), 0.028, LEAF, collide=False)]
    return Body.build("window", pos=(0.0, 0.0, TOP), mass=10.0, shapes=shapes, kinematic=True, color=WALL)


SPIDER, GLASS, CARD, WINDOW = _spider(), _glass(), _card(), _window()
OBJECTS = [WINDOW, CARD, GLASS, SPIDER]


# -- geometry (pure; tested offline) -----------------------------------------------------------------------------------

def spider_centre(spider) -> np.ndarray:
    """The middle of the stand-in's shell (half its height above the floor it stands on)."""
    return np.asarray(spider.pos, float) + rotation(spider.quat_xyzw) @ np.array([0.0, 0.0, BOT_H / 2])


def rim_points(glass) -> np.ndarray:
    return np.asarray(glass.pos, float) + RIM @ rotation(glass.quat_xyzw).T


def tilt(quat) -> float:
    return math.acos(max(-1.0, min(1.0, float(rotation(quat)[2, 2]))))


def to_card(card, p) -> np.ndarray:
    return rotation(card.quat_xyzw).T @ (np.asarray(p, float) - np.asarray(card.pos, float))


def over_card(q) -> bool:
    return abs(q[0]) <= CARD_X / 2 and abs(q[1]) <= CARD_Y / 2


def surface_z(x: float, y: float) -> float:
    """Height of the fixed surface under (x, y): the desk, the window's sill, the outer sill or the window box (-1 m:
    nothing to rest on)."""
    if x < X_IN:
        return TOP
    if x <= X_OUT:
        return TOP + SILL_H
    if x <= SLAB_X1:
        return TOP + SLAB_H
    if PLANTER_X[0] <= x <= PLANTER_X[1] and abs(y) <= PLANTER_HALF_Y:
        return TOP + PLANTER_H
    return -1.0


def rim_gap(glass, card) -> tuple[float, bool]:
    """The largest gap under the glass's rim (each of 48 points over the card's face measured to the face, the others
    to the fixed surface below), and whether the whole rim is over the card's face."""
    gaps, all_on_card = rim_gaps(glass, card)
    return float(max(gaps)), all_on_card


def rim_gaps(glass, card) -> tuple[list, bool]:
    gaps, all_on_card = [], True
    for p in rim_points(glass):
        q = to_card(card, p)
        if over_card(q) and -0.004 < q[2] - CARD_T < 0.05:
            gaps.append(q[2] - CARD_T)
        else:
            all_on_card = False
            gaps.append(p[2] - surface_z(p[0], p[1]))
    return gaps, all_on_card


def in_glass(glass, point) -> tuple[float, float]:
    """(distance from the glass's axis, height above its rim's plane) of a point, in the glass's frame."""
    u = rotation(glass.quat_xyzw).T @ (np.asarray(point, float) - np.asarray(glass.pos, float))
    return float(math.hypot(u[0], u[1])), float(u[2])


def shut_in(spider, glass, card) -> bool:
    """Inside the upright glass, whose rim is at most TRAP_GAP above whatever is under it all round."""
    if tilt(glass.quat_xyzw) > UPRIGHT:
        return False
    r, h = in_glass(glass, spider_centre(spider))
    return r < GLASS_R_IN and -0.02 < h < GLASS_H - GLASS_BASE and rim_gap(glass, card)[0] <= TRAP_GAP


def on_card(spider, card) -> bool:
    """Standing on the card's face."""
    q = to_card(card, spider_centre(spider))
    return over_card(q) and abs(q[2] - CARD_T - BOT_H / 2) < 0.006


def sealed(spider, glass, card) -> bool:
    """Shut in with the card under the whole rim (rim at most SEAL_GAP above the card's face) and the stand-in on it."""
    if not shut_in(spider, glass, card):
        return False
    gap, all_on_card = rim_gap(glass, card)
    return all_on_card and gap <= SEAL_GAP and on_card(spider, card)


def squashed_now(spider, glass) -> bool:
    """The glass's rim on the stand-in: its middle line within SQUASH (horizontally, in the glass's frame) of the
    stand-in's centre, at or below the top of its shell."""
    r, h = in_glass(glass, spider_centre(spider))
    return abs(r - R_MID) < SQUASH and h > -(BOT_H / 2 + 0.0015)


def touch_gap(spider, hands) -> float:
    """The smallest distance from a hand point to the stand-in's shell (a cylinder; 9 m with no hands)."""
    base, gaps = np.asarray(spider.pos, float), [9.0]
    for h in hands:
        p = np.asarray(h, float)
        across = max(float(math.hypot(p[0] - base[0], p[1] - base[1])) - BOT_R, 0.0)
        up = max(float(p[2]) - (float(base[2]) + BOT_H), float(base[2]) - float(p[2]), 0.0)
        gaps.append(math.hypot(across, up))
    return min(gaps)


def inside(state) -> bool:
    """An object back in the room: its origin short of the wall's inner face and not below the desk."""
    return float(state.pos[0]) < X_IN and float(state.pos[2]) > TOP - 0.02


# -- what its sensors see ----------------------------------------------------------------------------------------------

WALL_POINTS = np.array([[X_IN, y, TOP + z] for y in np.arange(-0.62, 0.6201, 0.01) for z in (0.005, 0.015)])
FOLD_POINTS = np.array([[x, -CARD_Y / 2 + LIP_T / 2, CARD_T + z] for x in np.linspace(-CARD_X / 2, CARD_X / 2, 16)
                        for z in (0.006, 0.016)])


def obstacle_points(card, hands) -> np.ndarray:
    """Points on everything its infrared proximity sensors can see: the card's fold, the wall under the window and the
    grippers. Not the clear glass: infrared goes through it (so it sees a gripper through the glass too)."""
    pts = [np.asarray(card.pos, float) + FOLD_POINTS @ rotation(card.quat_xyzw).T, WALL_POINTS]
    if len(hands):
        pts.append(np.asarray(hands, float).reshape(-1, 3))
    return np.concatenate(pts)


def heading(spider) -> np.ndarray:
    """The way it faces, on the desk (unit vector)."""
    d = rotation(spider.quat_xyzw)[:2, 0]
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-9 else np.array([1.0, 0.0])


def sensed(spider, points) -> tuple[bool, bool]:
    """(front, back): whether its front pair or its back pair of proximity sensors sees any of ``points``: within
    SENSE_RANGE of that end of its shell, within SENSE_HALF of straight ahead (or behind), up to SENSE_TOP high."""
    base, h = np.asarray(spider.pos, float), heading(spider)
    pts = np.asarray(points, float)
    up = pts[:, 2] - base[2]
    pts = pts[(up > -0.002) & (up < SENSE_TOP)]
    seen = []
    for d in (h, -h):
        v = pts[:, :2] - (base[:2] + d * BOT_R)
        dist = np.linalg.norm(v, axis=1)
        seen.append(bool(np.any((dist <= SENSE_RANGE) & (v @ d >= dist * math.cos(SENSE_HALF)))))
    return seen[0], seen[1]


# -- what happens during a trial ---------------------------------------------------------------------------------------

_STATE: dict = {}


FATAL = ("touched", "squashed", "flung", "escaped", "broken")
FLAGS = ("shut", "trapped", "sealed", "lifted", "carry", "past", "out", *FATAL)


def _clear() -> None:
    _STATE.update({k: False for k in FLAGS}, steps=0, driving=False, front=False, back=False, near=9.0)


_clear()


@on_reset
def _new_episode() -> None:
    _clear()


def _stop():
    """Its program stops the wheels: (0, 0) once, when it was driving; None after (physics alone moves it)."""
    if _STATE["driving"]:
        _STATE["driving"] = False
        return np.zeros(2)
    return None


def update(spider, glass, card, hands, dt: float):
    """One control step: record what happened, then run the stand-in's program. ``hands`` are the grippers' tool points
    and fingertips (world). Returns the horizontal velocity (vx, vy) its wheels drive it at, (0, 0) when it has just
    stopped, or None while it stands still. Pure apart from _STATE: tested offline."""
    s = _STATE
    s["steps"] += 1
    if any(s[k] for k in FATAL) or s["out"]:
        return _stop()
    if float(np.min(rim_points(glass)[:, 2])) < TOP - FELL:
        s["broken"] = True
        return _stop()
    centre = spider_centre(spider)
    shut = shut_in(spider, glass, card)
    s["shut"] = shut
    if tilt(glass.quat_xyzw) < math.radians(60.0) and squashed_now(spider, glass):
        s["squashed"] = True
        return _stop()
    s["near"] = touch_gap(spider, hands)
    if not shut and s["near"] < TOUCH:
        s["touched"] = True
        return _stop()

    # carried out: shut in on the card without a break from inside the room until its centre is past the wall
    contained = shut and on_card(spider, card)
    if shut:
        s["trapped"] = True
        if sealed(spider, glass, card):
            s["sealed"] = True
        if contained and float(card.pos[2]) >= TOP + LIFTED:
            s["lifted"] = True
    if contained and centre[0] <= X_OUT:
        s["carry"] = True                       # shut in on the card, still in the room: a carry starts (or goes on)
    elif s["carry"] and not contained and not s["past"]:
        s["carry"] = False                      # let out before it was past the wall: that carry is over
    if s["carry"] and contained and centre[0] > X_OUT:
        s["past"] = True                        # carried past the wall's outer face, still shut in on the card
    if s["past"] and not contained and centre[0] <= X_OUT:
        s["carry"] = s["past"] = False          # brought back into the room uncovered
    if centre[0] > X_OUT + OUT_PAST and not shut:
        s["out" if s["past"] else "flung"] = True
        return _stop()
    if centre[0] < DESK_X0 or abs(centre[1]) > DESK_Y or centre[2] < TOP - 0.03:
        s["escaped"] = True
        return _stop()

    # its program: back away from what the front sensors see, drive on from what the back ones see, else stand still
    front, back = sensed(spider, obstacle_points(card, hands))
    s["front"], s["back"] = front, back
    if front == back:
        return _stop()
    s["driving"] = True
    return heading(spider) * (-DRIVE if front else DRIVE)


def _points(objects) -> float:
    s = _STATE
    if any(s[k] for k in FATAL):
        return 0.0
    if s["out"]:
        return 1.0 if inside(objects["glass"]) and inside(objects["card"]) else 0.75
    if s["lifted"]:
        return 0.75
    if s["sealed"]:
        return 0.5
    return 0.25 if s["trapped"] else 0.0


def score(objects) -> float:
    """0 for the whole trial if the stand-in was ever touched (a gripper's tool point or fingertip within 1 cm of its
    shell while it was not shut in), squashed (the glass's rim came down on it), knocked out of the window (past the wall
    without having been carried there shut in), or escaped (got behind x 5 cm or past 50 cm left or right inside the
    room, or off the desk), or if the glass fell off the desk or out of the window. Otherwise 1.0 once it is out
    (carried out shut in on the card, then its centre 1 cm past the wall's outer face and not shut in) with the glass and
    the card back inside the room; 0.75 if it is out but the glass or the card is still outside, or if it was shut in on
    the card with the card's underside 3 cm above the desk; 0.5 if it was shut in with the card under the whole rim (at
    most 4 mm below it all round); 0.25 if it was ever shut in under the glass (inside it, the rim at most 8 mm above the
    desk or card all round); else 0. Carried out means shut in on the card without a break from inside the room until
    its centre is past the wall's outer face: let out of the glass before that, it was not carried out."""
    points = _points(objects)
    k = _STATE["steps"]
    fine = 230 <= k <= 300 or 430 <= k <= 480                    # the slide and the release: every 2nd step
    if os.environ.get("SPIDER_DEBUG") and (k % 10 == 0 or (fine and k % 2 == 0)):
        _debug(objects, points)
    return points


def _debug(objects, points) -> None:
    """One line on stderr: the demo phase (at 20 Hz), the flags set, its program's state, where things are (cm), the
    rim's gap."""
    s, t, label = _STATE, _STATE["steps"] * 0.05, "after the demo"
    for seconds, name, _, _ in demo_phases():
        if t < seconds:
            label = name
            break
        t -= seconds
    sp, gl, cd = objects["spider"], objects["glass"], objects["card"]
    gaps, under = rim_gaps(gl, cd)
    cm = lambda v: ",".join(f"{100 * float(x):.1f}" for x in v)  # noqa: E731
    deg = lambda q: f"{math.degrees(tilt(q)):.1f}"  # noqa: E731
    sees = ("F" if s["front"] else "-") + ("B" if s["back"] else "-") + (" drive" if s["driving"] else "")
    print(f"[spider] {s['steps']} {label} | {points} {' '.join(k for k in FLAGS if s[k]) or '-'} | sees {sees} | "
          f"bot {cm(sp.pos)} tilt {deg(sp.quat_xyzw)} on_card {on_card(sp, cd)} | glass {cm(gl.pos)} tilt "
          f"{deg(gl.quat_xyzw)} | card {cm(cd.pos)} tilt {deg(cd.quat_xyzw)} | gap {1000 * max(gaps):.1f}/"
          f"{1000 * min(gaps):.1f}mm under {under} | hand {100 * s['near']:.1f}cm", file=sys.stderr, flush=True)


def success(objects) -> bool:
    """Full score, with the glass and the card at rest."""
    return _points(objects) >= 1.0 and objects["glass"].resting() and objects["card"].resting()


def report(objects) -> dict:
    out = {k: _STATE[k] for k in (*FLAGS, "steps")}
    out["score"] = _points(objects)
    for name in ("spider", "glass", "card"):
        out[f"{name}_cm"] = [round(100 * float(v), 1) for v in objects[name].pos]
    return out


# -- the simulator side (needs Isaac Sim; not run offline) -------------------------------------------------------------

def hand_points(environment) -> list:
    """Each gripper's tool point and the tips and middles of its two fingers (world, relative to the env origin)."""
    base = environment.unwrapped
    origin = as_numpy(base.scene.env_origins[0])
    pts = []
    for arm in ("left_robot", "right_robot"):
        robot = base.scene[arm]
        pose = as_numpy(robot.data.body_link_pose_w)[0][list(robot.body_names).index("gripper")]
        rot = rotation(pose[3:7])
        tcp = pose[:3] - origin + rot @ np.array(TCP_OFFSET, float)
        names, q = list(robot.joint_names), as_numpy(robot.data.joint_pos)[0]
        half = sum(abs(float(q[names.index(j)])) for j in ("joint7", "joint8")) / 2
        tip = tcp + rot[:, 2] * PAD_TIP_PAST_TCP
        pts += [tcp, tip, tcp + rot[:, 0] * half, tcp - rot[:, 0] * half, tip + rot[:, 0] * half,
                tip - rot[:, 0] * half]
    return pts


def drive_force(target, vel) -> np.ndarray:
    """Its wheels: the horizontal force (world, N) that drives it at ``target`` (m/s) from its velocity ``vel``: a
    feed-forward that balances the model's sliding friction plus a proportional term, at most DRIVE_F_MAX. Zero for a
    zero target (stopped)."""
    t = np.asarray(target, float)[:2]
    n = float(np.linalg.norm(t))
    if n < 1e-9:
        return np.zeros(2)
    f = DRIVE_FF * t / n + DRIVE_K * (t - np.asarray(vel, float)[:2])
    m = float(np.linalg.norm(f))
    return f * (DRIVE_F_MAX / m) if m > DRIVE_F_MAX else f


def _push(environment, fxy) -> None:
    """Hold a horizontal force (world) on the stand-in for every physics step until the next control step."""
    import torch

    asset = environment.unwrapped.scene["spider"]
    f = torch.tensor([[[float(fxy[0]), float(fxy[1]), 0.0]]], dtype=torch.float32, device=asset.device)
    asset.permanent_wrench_composer.set_forces_and_torques_index(forces=f, torques=torch.zeros_like(f), is_global=True)


@on_step
def _watch(environment) -> None:
    from yam_arena.scene import object_states  # importable only once the simulator has started

    st = object_states(environment, ["spider", "glass", "card"])
    dt = float(getattr(environment.unwrapped, "step_dt", 0.05))
    v = update(st["spider"], st["glass"], st["card"], hand_points(environment), dt)
    if v is not None:
        _push(environment, drive_force(v, st["spider"].lin_vel))


# -- scripted demo -----------------------------------------------------------------------------------------------------
# The left hand takes the glass by its upturned base, fingers closing along x, puts it down over the stand-in, then
# lifts its rim a few millimetres. The right hand takes the middle of the card's fold (fingers along y) and slides the
# card in under the glass from the right while the left hand moves the glass 1.5 cm toward it: the glass's far wall
# pushes the stand-in onto the card's edge. The glass is set down on the card, and the two hands lift and carry glass,
# card and stand-in together out over the sill. There they lower them until the card is just over the window's sill,
# the right hand pulls the card away, the stand-in drops 1.75 cm onto the outer sill with the glass still round its top,
# the glass is raised clear of it, and both hands bring the glass and the card back in and set them down.
# Every target comes from the layout above: nothing is read from the simulator.

HOVER = TOP + 0.16
GRIP_BELOW_TOP = 0.010          # the left tool point this far below the glass's (upturned) base
PINCH_ABOVE = 0.005             # the right fingertips this far above the card's face
LIFT_RIM = 0.009                # the rim raised this much while the card goes under (the hand sags ~3 mm under it)
GLASS_PUSH = 0.020              # while the card slides in, the glass moves this far toward it (-y)
RELEASE_UP = 0.028              # outside, glass and card come down until the card's underside is this high (8 mm over
#                                 the window's sill): the stand-in drops 1.75 cm, its top still inside the glass
CLEAR_RIM = 0.057               # then the glass is raised until its rim is this high (1.2 cm over the stand-in's lid;
#                                 higher, the left arm leaves its reach)
CARRY_UP = 0.040                # the glass and card lifted this much to carry them (card's underside 2 cm over the sill)
RELEASE_XY = (0.375, 0.080)     # where the glass is held outside (its back edge 2.4 cm past the wall)
PULL = 0.130                    # the card pulled this far (-y) from under it
SLIDE_PAST = 0.010              # the card slid in until its leading edge is this far past the glass's far side
LEFT_SAG = 0.0                  # extra height for the left hand while it holds the glass on the card (tune on the sim)
LEFT_AIM = (0.014, -0.0045)     # the left hand holding the glass falls this far short (x, y) far out: aim past it
# The left grip closes its fingers along x (the wrist turned the way that keeps joint 6 inside its +-120 degrees);
# the right one along y, across the fold, so the card cannot swing in its fingers.
YAW = {"left": -math.pi / 2, "right": 0.0}


def _glass_tcp(rim_z: float) -> float:
    return rim_z + GLASS_H - GRIP_BELOW_TOP


def _pinch_tcp(card_bottom: float) -> float:
    return card_bottom + CARD_T + PINCH_ABOVE + PAD_TIP_PAST_TCP


def demo_phases() -> list:
    """(seconds, label, left target, right target): a target is (x, y, tool point z, finger) or None (stay)."""
    gx, gy = GLASS_XY
    sx, sy = SPIDER_XY
    tx, ty = sx, sy - GLASS_PUSH                                       # the glass after it has pushed
    rx, ry = RELEASE_XY
    px, py = CARD_XY[0], CARD_XY[1] - CARD_Y / 2 + LIP_T / 2          # the middle of the fold
    slide = ty + GLASS_R + SLIDE_PAST - (CARD_XY[1] + CARD_Y / 2)   # just far enough: the right hand stays clear
    o, c = FINGER_OPEN, FINGER_CLOSED
    g_rest, g_carry, c_rest = _glass_tcp(TOP + 0.0005), _glass_tcp(TOP + 0.060), _pinch_tcp(TOP)
    g_card = _glass_tcp(TOP + CARD_T + 0.0005) + LEFT_SAG
    dx, dy = rx - tx, ry - ty
    ax, ay = LEFT_AIM                                                   # the left hand's aim only
    sx, sy, tx, ty, rx, ry = sx + ax, sy + ay, tx + ax, ty + ay, rx + ax, ry + ay
    return [
        (2.0, "to the glass / to the card's fold", (gx, gy, HOVER, o), (px, py, HOVER, o)),
        (1.5, "down round the glass / onto the fold", (gx, gy, g_rest, o), (px, py, c_rest, o)),
        (0.8, "grip both", (gx, gy, g_rest, c), (px, py, c_rest, c)),
        (1.2, "lift the glass", (gx, gy, g_carry, c), None),
        (2.0, "carry it over the spider", (sx, sy, g_carry, c), None),
        (2.5, "lower it over the spider", (sx, sy, g_rest, c), None),
        (0.5, "wait", None, None),
        (1.2, "raise the rim", (sx, sy, g_rest + LIFT_RIM, c), None),
        (3.0, "slide the card in; the glass pushes the spider onto it", (tx, ty, g_rest + LIFT_RIM, c),
         (px, py + slide, c_rest, c)),
        (0.5, "wait", None, None),
        (0.8, "rim down onto the card", (tx, ty, g_card, c), None),
        (1.5, "lift glass and card together", (tx, ty, g_card + CARRY_UP, c), (px, py + slide, c_rest + CARRY_UP, c)),
        (3.0, "carry them out over the sill", (rx, ry, g_card + CARRY_UP, c),
         (px + dx, py + slide + dy, c_rest + CARRY_UP, c)),
        (1.2, "lower them toward the outer sill", (rx, ry, g_card + RELEASE_UP, c),
         (px + dx, py + slide + dy, c_rest + RELEASE_UP, c)),
        (2.0, "pull the card away", None, (px + dx, py + slide + dy - PULL, c_rest + RELEASE_UP, c)),
        (1.0, "let the spider drop", None, None),
        (1.0, "raise the glass off it", (rx, ry, _glass_tcp(TOP + CLEAR_RIM), c), None),
        (2.5, "bring the glass and the card back in", (gx, gy, g_carry, c), (px, py, c_rest + CARRY_UP, c)),
        (1.5, "set them down", (gx, gy, g_rest + 0.001, c), (px, py, c_rest + 0.002, c)),
        (0.6, "let go", (gx, gy, g_rest + 0.001, o), (px, py, c_rest + 0.002, o)),
        (1.2, "up and clear", (gx, gy, HOVER, o), (px, py, HOVER, o)),
    ]


def demo():
    plans = {"left": [], "right": []}
    last = {"left": None, "right": None}
    for seconds, label, left, right in demo_phases():
        for arm, target in (("left", left), ("right", right)):
            target = target if target is not None else last[arm]
            last[arm] = target
            x, y, z, finger = target
            plans[arm].append(Waypoint(seconds, (float(x), float(y), float(z)), float(finger), label, YAW[arm]))
    return plans
