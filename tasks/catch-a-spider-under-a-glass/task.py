"""On the station table, used as a desk pushed against the wall under a wide window that stands open: a house spider
sitting on the desk, an empty drinking glass standing upside down, and a stiff card lying flat.

- The window: the wall's inner face runs across the desk 28.5 cm in front of the arm bases. The opening is 90 cm wide
  and its frame's sill is 3 cm deep and 2 cm above the desk. Outside are a stone outer sill and a window box.
- The spider: a house spider, body 1.6 cm long, legs spanning 4 cm. It bolts if a hand comes within 5 cm of it, and it
  does not stay on anything that is moved or lifted unless it is shut in.
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

SPIDER_XY = (0.200, 0.060)
SPIDER_R = 0.0045                       # its body's collision sphere (radius); the centre is this high off the floor
SPIDER_MASS = 0.003                     # heavier than a real one (about 0.1 g) so the solver stays stable

GLASS_XY = (0.170, 0.240)
GLASS_R, GLASS_WALL, GLASS_H, GLASS_BASE = 0.036, 0.0035, 0.090, 0.008
GLASS_R_IN = GLASS_R - GLASS_WALL
GLASS_MASS = 0.22

CARD_XY = (0.190, -0.140)               # middle of the flat face
CARD_X, CARD_Y, CARD_T = 0.150, 0.250, 0.0015
LIP_T, LIP_H = 0.003, 0.020             # the folded-up strip along its right-hand short edge (-y): doubled card
CARD_MASS = 0.040

# -- the spider's behaviour --------------------------------------------------------------------------------------------
STARTLE = 0.05          # a hand point (tool point or fingertip) this close makes a free spider bolt
CALM = 0.10             # it keeps running while a hand point is this close
RUN_SPEED = 0.08        # m/s
RUN_S = 1.0             # once startled it runs at least this long
CARD_STILL = 0.03       # m/s: an uncovered spider gets off a card moving faster than this ...
CARD_UP = 0.010         # ... or lifted this far off the desk
STEP_ON = 0.003         # shut in, it steps onto the card when the card's edge comes this close to its body

# -- scoring -----------------------------------------------------------------------------------------------------------
TOUCH = 0.020           # a hand point this close to the centre of a spider that is not shut in touches it (legs 2 cm)
TRAP_GAP = 0.008        # shut in: inside the glass, its rim at most this far above whatever is under it, all round
SEAL_GAP = 0.004        # card on: the whole rim over the card's face, at most this far above it
LIFTED = 0.030          # lifted: the card's underside this far above the desk
OUT_PAST = 0.010        # out: the spider's centre this far past the wall's outer face, not shut in
DESK_X0, DESK_Y = 0.05, 0.50            # escaped: the spider gets behind x 0.05 or past |y| 0.50 (inside the room)
FELL = 0.030            # the glass's rim this far below the desk top: it fell off the desk or out of the window
UPRIGHT = math.radians(25.0)            # the glass counts as standing on its rim within this tilt
R_MID = GLASS_R - GLASS_WALL / 2
SQUASH = GLASS_WALL / 2 + SPIDER_R - 0.0015   # the rim's middle line this close (horizontally) to the spider's centre
RIM = np.array([[R_MID * math.cos(a), R_MID * math.sin(a), 0.0] for a in np.linspace(0.0, 2 * math.pi, 48,
                                                                                         endpoint=False)])

# -- objects -----------------------------------------------------------------------------------------------------------
FRAME, WALL, STONE, POT, SOIL, LEAF = (0.95, 0.95, 0.93), (0.90, 0.87, 0.80), (0.62, 0.60, 0.57), (0.72, 0.38, 0.24), \
    (0.25, 0.17, 0.11), (0.25, 0.50, 0.20)
PANE = (0.72, 0.84, 0.90)
GLASS_EDGE, GLASS_GLINT = (0.78, 0.87, 0.90), (0.90, 0.95, 0.97)
BROWN, DARK, LEG = (0.33, 0.22, 0.13), (0.20, 0.13, 0.08), (0.24, 0.16, 0.10)
GREY_CARD, PRINT = (0.62, 0.61, 0.58), (0.30, 0.52, 0.70)


def _ring(r_mid, thick, z0, h, n, color, collide=True):
    """``n`` flat boxes round the z axis: the wall of a cylinder (the first faces +x, so two face +-x and two +-y)."""
    chord = 2 * (r_mid + thick / 2) * math.tan(math.pi / n) + 0.0005
    return [BoxShape((r_mid * math.cos(a), r_mid * math.sin(a), z0 + h / 2), (thick, chord, h), color, yaw=a,
                     collide=collide) for a in (2 * math.pi * i / n for i in range(n))]


def _spider() -> Body:
    """Origin under the middle of its body, on the floor; it faces +x. One collision sphere, the rest drawn only."""
    shapes = [SphereShape((0.0, 0.0, SPIDER_R), SPIDER_R, None),
              SphereShape((-0.0042, 0.0, 0.0047), 0.0046, BROWN, collide=False),     # abdomen
              SphereShape((0.0036, 0.0, 0.0040), 0.0033, DARK, collide=False)]       # head and thorax
    for side in (1.0, -1.0):
        for deg, reach in ((40.0, 0.020), (72.0, 0.017), (105.0, 0.017), (140.0, 0.019)):
            a = math.radians(deg)
            c, s = math.cos(a), side * math.sin(a)
            root = (0.0036 + 0.0015 * c, 0.0022 * s / max(abs(s), 1e-9), 0.0042)
            knee = (0.0036 + 0.010 * c, 0.010 * s, 0.0105)
            foot = (0.0036 + reach * c, reach * s, 0.0006)
            shapes += [segment(root, knee, 0.0010, LEG, collide=False), segment(knee, foot, 0.0008, LEG, collide=False)]
    return Body.build("spider", pos=(*SPIDER_XY, TOP + 0.0003), mass=SPIDER_MASS, shapes=shapes, color=BROWN,
                      friction=(0.8, 0.7), damping=(0.5, 8.0), contact_offset=0.001)


def _glass() -> Body:
    """Origin in the middle of the rim (the rim on the desk, its base 9 cm up). The walls and base collide but are not
    drawn, so the spider shows through; the rim, the base's edge and a few glints are drawn."""
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
    return np.asarray(spider.pos, float) + rotation(spider.quat_xyzw) @ np.array([0.0, 0.0, SPIDER_R])


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


def sealed(spider, glass, card) -> bool:
    """Shut in with the card under the whole rim (rim at most SEAL_GAP above the card's face) and the spider on it."""
    if not shut_in(spider, glass, card):
        return False
    gap, all_on_card = rim_gap(glass, card)
    q = to_card(card, spider_centre(spider))
    return all_on_card and gap <= SEAL_GAP and over_card(q) and q[2] > CARD_T


def squashed_now(spider, glass) -> bool:
    """The glass's rim on the spider's body: its middle line within SQUASH (horizontally, in the glass's frame) of the
    spider's centre, at or below the top of the body."""
    r, h = in_glass(glass, spider_centre(spider))
    return abs(r - R_MID) < SQUASH and h > -(SPIDER_R + 0.0015)


def card_face_z(card, x: float, y: float) -> float:
    """Height of the plane of the card's face at (x, y) (-1 m when the card stands on edge)."""
    n, c = rotation(card.quat_xyzw)[:, 2], np.asarray(card.pos, float)
    if n[2] < 0.5:
        return -1.0
    return float(c[2] + (CARD_T - n[0] * (x - c[0]) - n[1] * (y - c[1])) / n[2])


def on_card(spider, card) -> bool:
    q = to_card(card, spider_centre(spider))
    return over_card(q) and abs(q[2] - CARD_T - SPIDER_R) < 0.006


def inside(state) -> bool:
    """An object back in the room: its origin short of the wall's inner face and not below the desk."""
    return float(state.pos[0]) < X_IN and float(state.pos[2]) > TOP - 0.02


# -- what happens during a trial ---------------------------------------------------------------------------------------

_STATE: dict = {}


FATAL = ("touched", "squashed", "flung", "escaped", "broken")
FLAGS = ("shut", "trapped", "sealed", "lifted", "out", *FATAL)


def _clear() -> None:
    _STATE.update({k: False for k in FLAGS}, run=0.0, dir=(-1.0, 0.0), steps=0, rel=None)


_clear()


@on_reset
def _new_episode() -> None:
    _clear()


def _blocked(xy, here, glass) -> bool:
    """A running spider stops at the wall (it never climbs out of the window on its own) and at a glass standing near
    the desk (unless it is already under it)."""
    if xy[0] > X_IN - 0.010:
        return True
    lo, hi = np.asarray(glass.low, float), np.asarray(glass.high, float)
    if lo[2] < TOP + 0.02:
        r = SPIDER_R + 0.002
        within = lambda p: lo[0] - r <= p[0] <= hi[0] + r and lo[1] - r <= p[1] <= hi[1] + r  # noqa: E731
        return within(xy) and not within(here)
    return False


def _rotate(d, deg) -> np.ndarray:
    a = math.radians(deg)
    return np.array([d[0] * math.cos(a) - d[1] * math.sin(a), d[0] * math.sin(a) + d[1] * math.cos(a)])


def _floor_under(xy, card) -> float:
    """Where a spider on the desk at ``xy`` stands: on the card's face if the card lies flat there, else the desk."""
    q = to_card(card, (xy[0], xy[1], TOP + 0.005))
    if over_card(q) and float(card.pos[2]) < TOP + 0.004 and tilt(card.quat_xyzw) < math.radians(5.0):
        return float(card.pos[2]) + CARD_T
    return TOP


def _sit_in_glass(spider, glass, card):
    """Shut in, it keeps its place under the glass (it goes where the glass goes) and stands on whatever is under it.
    When the card's edge comes in under the rim to within STEP_ON of its body, it steps onto the card."""
    s = _STATE
    rg, g = rotation(glass.quat_xyzw), np.asarray(glass.pos, float)
    if s["rel"] is None:
        s["rel"] = rg.T @ (spider_centre(spider) - g)
    p = g + rg @ s["rel"]
    floor = surface_z(p[0], p[1])
    z = card_face_z(card, p[0], p[1])
    q = to_card(card, (p[0], p[1], z))
    reach = SPIDER_R + STEP_ON
    if (abs(q[0]) <= CARD_X / 2 + reach and abs(q[1]) <= CARD_Y / 2 + reach
            and floor - 0.002 < z <= float(np.max(rim_points(glass)[:, 2])) + 0.003):
        if over_card(q):
            floor = z
        else:                                       # the card's edge is at it: it steps onto the card
            inset = np.array([CARD_X / 2 - 0.001, CARD_Y / 2 - 0.001])
            w = np.asarray(card.pos, float) + rotation(card.quat_xyzw) @ np.array([*np.clip(q[:2], -inset, inset),
                                                                                    CARD_T])
            if in_glass(glass, (w[0], w[1], p[2]))[0] < GLASS_R_IN - SPIDER_R:
                p[:2] = w[:2]
                floor = card_face_z(card, p[0], p[1])
                s["rel"] = rg.T @ (np.array([p[0], p[1], floor + SPIDER_R]) - g)
    r = rotation(spider.quat_xyzw)
    return np.array([p[0], p[1], max(floor, 0.0) + 0.0003]), math.atan2(r[1, 0], r[0, 0])


def update(spider, glass, card, hands, dt: float):
    """One control step: record what happened and move the spider on its own. ``hands`` are the grippers' tool points
    and fingertips (world). Returns the spider's new (position, yaw) when it moves itself, the string "gone" when it
    has escaped off the desk, or None. Pure apart from _STATE: tested offline."""
    s = _STATE
    s["steps"] += 1
    if any(s[k] for k in FATAL):
        return None
    if float(np.min(rim_points(glass)[:, 2])) < TOP - FELL:
        s["broken"] = True
        return None
    if s["out"]:                                       # let go outside: it is gone, nothing more to watch
        return None
    centre = spider_centre(spider)
    shut = shut_in(spider, glass, card)
    s["shut"] = shut
    if not shut:
        s["rel"] = None
    if tilt(glass.quat_xyzw) < math.radians(60.0) and squashed_now(spider, glass):
        s["squashed"] = True
        return None
    near = min((float(np.linalg.norm(np.asarray(h, float) - centre)) for h in hands), default=9.0)
    s["near"] = near
    s["left_tcp"] = hands[0] if len(hands) > 6 else None
    if not shut and near < TOUCH:
        s["touched"] = True
        return None
    if shut:
        s["trapped"] = True
        if sealed(spider, glass, card):
            s["sealed"] = True
        if on_card(spider, card) and float(card.pos[2]) >= TOP + LIFTED:
            s["lifted"] = True
        return _sit_in_glass(spider, glass, card)
    if centre[0] > X_OUT + OUT_PAST:                   # past the wall: carried out shut in, or knocked out
        s["out" if s["lifted"] else "flung"] = True
        return None
    if centre[0] < DESK_X0 or abs(centre[1]) > DESK_Y or centre[2] < TOP - 0.03:
        s["escaped"] = True
        return "gone"

    # not shut in, still in the room: it gets off a card that is moved or lifted ...
    if on_card(spider, card) and (float(card.pos[2]) > TOP + CARD_UP
                                  or float(np.linalg.norm(card.lin_vel[:2])) > CARD_STILL):
        q = to_card(card, centre)
        edge = np.array([0.0, 0.0])
        if CARD_X / 2 - abs(q[0]) < CARD_Y / 2 - abs(q[1]):
            edge[0] = math.copysign(CARD_X / 2 + 0.015, q[0])
            edge[1] = q[1]
        else:
            edge[0] = q[0]
            edge[1] = math.copysign(CARD_Y / 2 + 0.015, q[1])
        p = np.asarray(card.pos, float) + rotation(card.quat_xyzw) @ np.array([edge[0], edge[1], 0.0])
        if X_IN - 0.012 < p[0] <= X_OUT:                     # it would land on the window's sill: it hops back in
            p[0] = X_IN - 0.012
        z = surface_z(p[0], p[1])
        if p[0] > X_OUT:
            s["out" if s["lifted"] else "flung"] = True
        else:
            d = p[:2] - centre[:2]
            s["dir"] = tuple(d / max(float(np.linalg.norm(d)), 1e-9))
            s["run"] = RUN_S
        return np.array([p[0], p[1], max(z, 0.0) + 0.0003]), math.atan2(s["dir"][1], s["dir"][0])

    # ... and it bolts from a hand that comes close
    if near < STARTLE or (s["run"] > 0.0 and near < CALM):
        nearest = min(hands, key=lambda h: float(np.linalg.norm(np.asarray(h, float) - centre)))
        away = centre[:2] - np.asarray(nearest, float)[:2]
        n = float(np.linalg.norm(away))
        s["dir"] = tuple(away / n) if n > 1e-6 else s["dir"]
        s["run"] = max(s["run"], RUN_S)
    if s["run"] <= 0.0:
        return None
    s["run"] -= dt
    for turn in (0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0):
        d = _rotate(s["dir"], turn)
        xy = centre[:2] + d * RUN_SPEED * dt
        if not _blocked(xy, centre[:2], glass):
            break
    else:
        return None
    s["dir"] = (float(d[0]), float(d[1]))
    if xy[0] < DESK_X0 or abs(xy[1]) > DESK_Y:
        s["escaped"] = True
        return "gone"
    return np.array([xy[0], xy[1], _floor_under(xy, card) + 0.0003]), math.atan2(d[1], d[0])


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
    """0 for the whole trial if the spider was ever touched (a gripper's tool point or fingertip within 2 cm of its
    centre while it was not shut in), squashed (the glass's rim came down on its body), knocked out of the window
    (past the wall without having been carried there shut in), or escaped (got behind x 5 cm or past 50 cm left or
    right inside the room, or off the desk), or if the glass fell off the desk or out of the window. Otherwise 1.0 once
    the spider is out (carried out shut in, then its centre 1 cm past the wall's outer face and not shut in) with the
    glass and the card back inside the room; 0.75 if it is out but the glass or the card is still outside, or if it was
    shut in on the card with the card's underside 3 cm above the desk; 0.5 if it was shut in with the card under the
    whole rim (at most 4 mm below it all round); 0.25 if it was ever shut in under the glass (inside it, the rim at most
    8 mm above the desk or card all round); else 0."""
    points = _points(objects)
    k = _STATE["steps"]
    if os.environ.get("SPIDER_DEBUG") and (k % 10 == 0 or (250 <= k <= 300 and k % 2 == 0)):
        _debug(objects, points)
    return points


def _debug(objects, points) -> None:
    """One line on stderr: the demo phase (at 20 Hz), the flags set, where things are (cm), the rim's gap."""
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
    lt = s.get("left_tcp")
    print(f"[spider] {s['steps']} {label} | {points} {' '.join(k for k in FLAGS if s[k]) or '-'} | "
          f"spider {cm(spider_centre(sp))} | glass {cm(gl.pos)} tilt {deg(gl.quat_xyzw)} | "
          f"card {cm(cd.pos)} tilt {deg(cd.quat_xyzw)} | gap {1000 * max(gaps):.1f}/{1000 * min(gaps):.1f}mm "
          f"under {under} | hand {100 * s.get('near', 9.0):.1f}cm | left tcp {cm(lt) if lt is not None else '-'}",
          file=sys.stderr, flush=True)


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


def _write_spider(environment, pos, yaw: float) -> None:
    """Put the spider at ``pos`` (relative to the env origin), facing ``yaw``, at rest."""
    import torch

    base = environment.unwrapped
    asset = base.scene["spider"]
    origin = as_numpy(base.scene.env_origins[0])
    data = asset.data.root_link_pose_w
    dev = getattr(data, "torch", data).device
    quat = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    pose = torch.tensor([[*(np.asarray(pos, float) + origin), *quat]], dtype=torch.float32, device=dev)
    vel = torch.zeros((1, 6), dtype=torch.float32, device=dev)
    for stem, kw in (("write_root_pose_to_sim", {"root_pose": pose}),
                     ("write_root_velocity_to_sim", {"root_velocity": vel})):
        fn = getattr(asset, f"{stem}_index", None)
        if fn is not None:
            fn(env_ids=[0], **kw)
        else:
            getattr(asset, stem)(*kw.values())


@on_step
def _watch(environment) -> None:
    from yam_arena.scene import object_states  # importable only once the simulator has started

    st = object_states(environment, ["spider", "glass", "card"])
    dt = float(getattr(environment.unwrapped, "step_dt", 0.05))
    move = update(st["spider"], st["glass"], st["card"], hand_points(environment), dt)
    if move is None:
        return
    if isinstance(move, str):                       # escaped: it ran off the desk onto the floor, out of sight
        c = spider_centre(st["spider"])
        _write_spider(environment, (float(c[0]), math.copysign(0.75, float(c[1]) or 1.0), 0.0003), 0.0)
        return
    _write_spider(environment, move[0], move[1])


# -- scripted demo -----------------------------------------------------------------------------------------------------
# The left hand takes the glass by its upturned base, fingers closing along x, puts it down over the spider, then lifts
# its rim a few millimetres. The right hand takes the middle of the card's fold (fingers along y) and slides the card
# in under the glass from the right. The glass is set down on the card, and the two hands lift and carry glass, card and spider together out
# over the sill. There the right hand pulls the card away, the spider drops onto the outer sill, and both hands bring
# the glass and the card back in and set them down. Every target comes from the layout above: nothing is read from
# the simulator.

HOVER = TOP + 0.16
GRIP_BELOW_TOP = 0.010          # the left tool point this far below the glass's (upturned) base
PINCH_ABOVE = 0.005             # the right fingertips this far above the card's face
LIFT_RIM = 0.009                # the rim raised this much while the card goes under (the hand sags ~3 mm under it)
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
    rx, ry = RELEASE_XY
    px, py = CARD_XY[0], CARD_XY[1] - CARD_Y / 2 + LIP_T / 2          # the middle of the fold
    slide = sy + GLASS_R + SLIDE_PAST - (CARD_XY[1] + CARD_Y / 2)   # just far enough: the right hand stays clear
    o, c = FINGER_OPEN, FINGER_CLOSED
    g_rest, g_carry, c_rest = _glass_tcp(TOP + 0.0005), _glass_tcp(TOP + 0.060), _pinch_tcp(TOP)
    g_card = _glass_tcp(TOP + CARD_T + 0.0005) + LEFT_SAG
    dx, dy = rx - sx, ry - sy
    sx, sy, rx, ry = sx + LEFT_AIM[0], sy + LEFT_AIM[1], rx + LEFT_AIM[0], ry + LEFT_AIM[1]   # left hand's aim only
    return [
        (2.0, "to the glass / to the card's fold", (gx, gy, HOVER, o), (px, py, HOVER, o)),
        (1.5, "down round the glass / onto the fold", (gx, gy, g_rest, o), (px, py, c_rest, o)),
        (0.8, "grip both", (gx, gy, g_rest, c), (px, py, c_rest, c)),
        (1.2, "lift the glass", (gx, gy, g_carry, c), None),
        (2.0, "carry it over the spider", (sx, sy, g_carry, c), None),
        (2.5, "lower it over the spider", (sx, sy, g_rest, c), None),
        (0.5, "wait", None, None),
        (1.2, "raise the rim", (sx, sy, g_rest + LIFT_RIM, c), None),
        (3.0, "slide the card in under the glass", None, (px, py + slide, c_rest, c)),
        (0.5, "wait", None, None),
        (0.8, "rim down onto the card", (sx, sy, g_card, c), None),
        (1.5, "lift glass and card together", (sx, sy, g_card + CARRY_UP, c), (px, py + slide, c_rest + CARRY_UP, c)),
        (3.0, "carry them out over the sill", (rx, ry, g_card + CARRY_UP, c),
         (px + dx, py + slide + dy, c_rest + CARRY_UP, c)),
        (2.0, "pull the card away", None, (px + dx, py + slide + dy - PULL, c_rest + CARRY_UP, c)),
        (1.0, "let the spider drop", None, None),
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
