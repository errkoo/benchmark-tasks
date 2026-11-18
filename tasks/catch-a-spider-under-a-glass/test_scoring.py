"""Offline checks of the geometry, the stand-in's program, the scoring rule and the scripted demo (no simulator):
python -m pytest test_scoring.py"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import task as T  # noqa: E402
from bodykit import state_from_pose  # noqa: E402

UP = (0.0, 0.0, 0.0, 1.0)
FAR = [np.array([0.0, 0.3, 1.2])]                       # hands up out of the way
DT = 0.05
SX, SY = T.SPIDER_XY
ON_CARD = T.TOP + 0.0003 + T.CARD_T                     # the stand-in's floor on a card lying on the desk


def yaw(a):
    return (0.0, 0.0, math.sin(a / 2), math.cos(a / 2))


def st(spec, xy, z, quat=UP, vel=(0.0, 0.0, 0.0)):
    return state_from_pose(spec, (xy[0], xy[1], z), quat, vel)


def spider(xy=T.SPIDER_XY, z=T.TOP + 0.0003, quat=UP):
    return st(T.SPIDER, xy, z, quat)


def glass(xy=T.SPIDER_XY, rim=T.TOP + 0.0005, quat=UP):
    return st(T.GLASS, xy, rim, quat)


def card(xy=T.CARD_XY, z=T.TOP + 0.0003, vel=(0.0, 0.0, 0.0)):
    return st(T.CARD, xy, z, vel=vel)


def near(dx, dy, dz=0.02):
    """A hand point (dx, dy) from the stand-in's start, dz above the desk."""
    return [np.array([SX + dx, SY + dy, T.TOP + dz])]


GLASS_AWAY = glass(T.GLASS_XY)


@pytest.fixture(autouse=True)
def fresh():
    T._clear()
    yield
    T._clear()


def objects(g=GLASS_AWAY, c=None, s=None):
    return {"glass": g, "card": c or card(), "spider": s or spider(), "window": None}


def stopped(v) -> bool:
    return v is None or not np.any(v)


# -- the task file loads --------------------------------------------------------------------------------------------

def test_task_file_loads():
    from yam_arena import task_file

    loaded = task_file.load(pathlib.Path(T.__file__))
    assert loaded.names == ("window", "card", "glass", "spider") and loaded.demo is not None
    assert set(loaded.free_objects) == {"window", "card", "glass", "spider"}     # bodykit bodies all carry a mass


# -- geometry -------------------------------------------------------------------------------------------------------

def test_free_at_the_start_and_shut_in_under_the_glass():
    assert not T.shut_in(spider(), GLASS_AWAY, card())
    assert T.shut_in(spider(), glass(), card())


def test_a_raised_rim_lets_it_out():
    assert T.shut_in(spider(), glass(rim=T.TOP + 0.007), card())         # 7 mm: still shut in
    assert not T.shut_in(spider(), glass(rim=T.TOP + 0.010), card())     # 1 cm: not


def test_glass_on_its_side_shuts_nothing_in():
    side = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))      # 90 degrees about x
    assert not T.shut_in(spider(), glass(quat=side), card())


def test_rim_on_its_shell_squashes_it_but_a_wall_against_it_does_not():
    on_top = glass((SX + T.R_MID, SY), rim=T.TOP + 0.0003 + T.BOT_H)               # rim sitting on its lid
    assert T.squashed_now(spider(), on_top)
    against_in = glass((SX + T.GLASS_R_IN - T.BOT_R + 0.001, SY))                 # pressed 1 mm into it from inside
    against_out = glass((SX + T.GLASS_R + T.BOT_R + 0.0005, SY))                  # touching it from outside
    assert not T.squashed_now(spider(), against_in) and not T.squashed_now(spider(), against_out)
    assert not T.squashed_now(spider(), glass((SX + T.R_MID, SY), rim=T.TOP + 0.05))   # rim well above it


def test_card_under_the_glass_seals_it():
    c = card(T.SPIDER_XY)
    s = spider(z=ON_CARD)
    g = glass(rim=ON_CARD + 0.0005)
    assert T.sealed(s, g, c) and T.on_card(s, c)
    part = card((SX, SY - 0.15))                                                  # its edge only just under the rim
    assert T.shut_in(spider(), glass(rim=T.TOP + 0.006), part) and not T.sealed(spider(), glass(), part)


def test_surfaces():
    assert T.surface_z(0.2, 0.0) == T.TOP
    assert T.surface_z(0.30, 0.0) == pytest.approx(T.TOP + T.SILL_H)
    assert T.surface_z(0.36, 0.0) == pytest.approx(T.TOP + T.SLAB_H)
    assert T.surface_z(0.80, 0.0) < 0


# -- the stand-in's program -----------------------------------------------------------------------------------------

def test_sits_still_while_nothing_is_near():
    assert T.update(spider(), GLASS_AWAY, card(), FAR, DT) is None
    assert not T._STATE["front"] and not T._STATE["back"]


def test_backs_away_from_a_hand_within_5_cm_in_front():
    v = T.update(spider(), GLASS_AWAY, card(), near(T.BOT_R + 0.03, 0.0), DT)
    assert v == pytest.approx([-T.DRIVE, 0.0]) and T._STATE["driving"]


def test_drives_on_away_from_a_hand_within_5_cm_behind():
    v = T.update(spider(), GLASS_AWAY, card(), near(-T.BOT_R - 0.03, 0.0), DT)
    assert v == pytest.approx([T.DRIVE, 0.0])


def test_does_not_see_a_hand_6_cm_away_above_its_sensors_or_at_its_side():
    assert T.update(spider(), GLASS_AWAY, card(), near(T.BOT_R + 0.06, 0.0), DT) is None
    assert T.update(spider(), GLASS_AWAY, card(), near(T.BOT_R + 0.02, 0.0, dz=0.05), DT) is None
    assert T.update(spider(), GLASS_AWAY, card(), near(0.0, T.BOT_R + 0.03), DT) is None


def test_runs_along_the_way_it_faces():
    v = T.update(spider(quat=yaw(math.pi / 2)), GLASS_AWAY, card(), near(0.0, T.BOT_R + 0.03), DT)
    assert v == pytest.approx([0.0, -T.DRIVE], abs=1e-9)


def test_stops_once_the_hand_has_gone():
    T.update(spider(), GLASS_AWAY, card(), near(T.BOT_R + 0.03, 0.0), DT)
    v = T.update(spider(), GLASS_AWAY, card(), FAR, DT)
    assert v is not None and not np.any(v) and not T._STATE["driving"]          # the wheels stop, once
    assert T.update(spider(), GLASS_AWAY, card(), FAR, DT) is None


def test_backs_away_from_the_window_wall_and_never_drives_into_it():
    at_wall = spider((T.X_IN - T.BOT_R - 0.02, 0.0))
    v = T.update(at_wall, GLASS_AWAY, card(), FAR, DT)
    assert v == pytest.approx([-T.DRIVE, 0.0])
    T._clear()
    chased = [np.array([T.X_IN - 2 * T.BOT_R - 0.04, 0.0, T.TOP + 0.02])]          # a hand behind it as well
    assert stopped(T.update(at_wall, GLASS_AWAY, card(), chased, DT))


def test_it_does_not_see_the_clear_glass_but_sees_a_hand_through_it():
    assert T.update(spider(), glass((SX + T.BOT_R + 0.045, SY)), card(), FAR, DT) is None   # 1 cm in front of it
    assert stopped(T.update(spider(), glass(), card(), near(0.0, 0.0, dz=0.05), DT))       # a hand on the glass
    v = T.update(spider(), glass(), card(), near(T.BOT_R + 0.04, 0.0), DT)                 # a hand low outside it
    assert v == pytest.approx([-T.DRIVE, 0.0])


def test_its_wheels_hold_the_programs_speed():
    assert T.drive_force([0.0, 0.0], [0.05, 0.0, 0.0]) == pytest.approx([0.0, 0.0])
    assert T.drive_force([T.DRIVE, 0.0], [T.DRIVE, 0.0, 0.0]) == pytest.approx([T.DRIVE_FF, 0.0])
    assert np.linalg.norm(T.drive_force([T.DRIVE, 0.0], [-1.0, 0.0, 0.0])) == pytest.approx(T.DRIVE_F_MAX)
    # a 28 g block with the model's sliding friction, the force updated at 20 Hz, physics at 240 Hz
    m, g, v = T.SPIDER_MASS, 9.81, 0.0
    for _ in range(20):
        f = T.drive_force([T.DRIVE, 0.0], [v, 0.0, 0.0])[0]
        for _ in range(12):
            if v > 1e-9:
                v = max(v + (f - T.SPIDER_FRICTION[1] * m * g) / m / 240, 0.0)
            elif f > T.SPIDER_FRICTION[0] * m * g:
                v += (f - T.SPIDER_FRICTION[1] * m * g) / m / 240
    assert v == pytest.approx(T.DRIVE, rel=0.05)


def test_getting_past_the_side_or_the_back_of_the_desk_is_an_escape():
    T.update(spider((0.20, T.DESK_Y + 0.005)), GLASS_AWAY, card(), FAR, DT)
    assert T._STATE["escaped"] and T.score(objects()) == 0.0
    T._clear()
    T.update(spider((T.DESK_X0 - 0.005, 0.0)), GLASS_AWAY, card(), FAR, DT)
    assert T._STATE["escaped"]


# -- scoring --------------------------------------------------------------------------------------------------------

LIFT = T.CARRY_UP
OUT = T.RELEASE_XY


def _run(steps):
    """Feed (spider, glass, card, hands) states through update()."""
    for s, g, c, h in steps:
        T.update(s, g, c, h, DT)


def _story():
    """The demo's story as states: shut in, card going in, sealed, lifted, carried out, let go outside."""
    up = LIFT
    return [
        (spider(), glass(), card(), FAR),                                                      # 0 shut in on the desk
        (spider(), glass(rim=T.TOP + 0.006), card((SX, SY - 0.15)), FAR),                      # 1 card going in
        (spider(z=ON_CARD), glass(rim=ON_CARD + 0.0005), card(T.SPIDER_XY), FAR),              # 2 sealed
        (spider(z=ON_CARD + up), glass(rim=ON_CARD + 0.0005 + up), card(T.SPIDER_XY, z=T.TOP + 0.0003 + up), FAR),
        (spider(OUT, z=ON_CARD + up), glass(OUT, rim=ON_CARD + 0.0005 + up), card(OUT, z=T.TOP + 0.0003 + up), FAR),
        (spider(OUT, z=T.TOP + T.SLAB_H + 0.0003), glass(OUT, rim=T.TOP + 0.09),
         card((OUT[0], OUT[1] - 0.13), z=T.TOP + 0.0003 + up), FAR),                         # 5 card away, glass up
    ]


def test_full_story_scores_one_once_glass_and_card_are_back_in():
    _run(_story())
    assert T._STATE["past"] and T._STATE["out"]
    assert T.score(objects(glass(OUT, rim=T.TOP + 0.09), card((0.36, -0.05), z=T.TOP + 0.044))) == 0.75
    assert T.score(objects(GLASS_AWAY, card())) == 1.0


def test_milestones():
    story = _story()
    _run(story[:1])
    assert T.score(objects()) == 0.25
    _run(story[1:3])
    assert T.score(objects()) == 0.5
    _run(story[3:4])
    assert T.score(objects()) == 0.75 and T._STATE["carry"] and not T._STATE["past"]


def test_uncontained_spider_leaving_on_a_moving_card_scores_zero():
    """The reviewer's case: lifted shut in on the card, then the glass comes off inside the room, and the stand-in
    rides out of the window on the moving card with nothing over it. That is not a contained carry: 0."""
    _run(_story()[:4])
    assert T._STATE["lifted"] and T._STATE["carry"]
    up = LIFT
    _run([(spider(z=ON_CARD + up), GLASS_AWAY, card(T.SPIDER_XY, z=T.TOP + 0.0003 + up), FAR)])   # glass off, inside
    assert not T._STATE["carry"]
    for x in (0.26, 0.30, 0.34):                                                   # the card moves out with it on it
        _run([(spider((x, SY), z=ON_CARD + up), GLASS_AWAY, card((x, SY), z=T.TOP + 0.0003 + up), FAR)])
    assert T._STATE["flung"] and not T._STATE["out"] and T.score(objects()) == 0.0


def test_a_card_carried_out_with_nothing_over_it_scores_zero():
    up = LIFT
    for x in (0.20, 0.26, 0.30, 0.34):
        _run([(spider((x, SY), z=ON_CARD + up), GLASS_AWAY, card((x, SY), z=T.TOP + 0.0003 + up), FAR)])
    assert T._STATE["flung"] and T.score(objects()) == 0.0


def test_covered_again_inside_the_room_the_carry_starts_over():
    story = _story()
    up = LIFT
    _run(story[:4])
    _run([(spider(z=ON_CARD + up), GLASS_AWAY, card(T.SPIDER_XY, z=T.TOP + 0.0003 + up), FAR)])   # glass off
    _run(story[3:])                                                                               # back on, then out
    assert T._STATE["out"] and T.score(objects()) == 1.0


def test_shut_in_once_is_kept_as_partial_credit():
    _run(_story()[:1])
    _run([(spider(), GLASS_AWAY, card(), FAR)])
    assert T.score(objects()) == 0.25


def test_knocked_out_of_the_window_without_being_carried_scores_zero():
    _run([(spider((0.34, 0.0), z=T.TOP + T.SLAB_H + 0.0003), GLASS_AWAY, card(), FAR)])
    assert T._STATE["flung"] and T.score(objects()) == 0.0


def test_touching_it_scores_zero():
    _run(_story()[:1])
    _run([(spider(), GLASS_AWAY, card(), near(T.BOT_R + 0.005, 0.0, dz=0.015))])   # 5 mm from its shell
    assert T._STATE["touched"] and T.score(objects()) == 0.0


def test_a_hand_on_the_glass_does_not_touch_it_shut_in_under_it():
    _run([(spider(), glass(), card(), near(0.0, 0.0, dz=0.04))])                   # 7 mm above its lid
    assert not T._STATE["touched"] and T.score(objects()) == 0.25


def test_squashing_it_scores_zero():
    _run([(spider(), glass((SX + T.R_MID, SY), rim=T.TOP + 0.0003 + T.BOT_H), card(), FAR)])
    assert T._STATE["squashed"] and T.score(objects()) == 0.0


def test_glass_off_the_desk_scores_zero():
    _run(_story()[:4])
    _run([(spider(), glass((0.5, 0.0), rim=0.2), card(), FAR)])
    assert T._STATE["broken"] and T.score(objects()) == 0.0


def test_nothing_done_scores_zero():
    _run([(spider(), GLASS_AWAY, card(), FAR)] * 5)
    assert T.score(objects()) == 0.0 and not T.success(objects())


# -- the scripted demo ----------------------------------------------------------------------------------------------

def _phase(prefix):
    for p in T.demo_phases():
        if p[1].startswith(prefix):
            return p
    raise KeyError(prefix)


def _rim(left_target):
    return left_target[2] - T.GLASS_H + T.GRIP_BELOW_TOP


def test_demo_story_fits_the_layout():
    _, _, left, right = _phase("slide the card in")
    ty = SY - T.GLASS_PUSH
    card_y = right[1] + T.CARD_Y / 2 - T.LIP_T / 2
    assert card_y + T.CARD_Y / 2 >= ty + T.GLASS_R + 0.009                    # its leading edge past the far side
    assert card_y - T.CARD_Y / 2 <= ty - T.GLASS_R - 0.03                     # its other edge well short of the near side
    assert abs(T.CARD_XY[0] - SX) + T.GLASS_R < T.CARD_X / 2 - 0.02           # the rim over it in x, 2 cm to spare
    rx, ry = T.RELEASE_XY
    assert rx - T.GLASS_R > T.X_OUT + 0.02                                   # the whole glass past the wall
    assert rx - T.GLASS_R_IN + T.BOT_R > T.X_OUT + T.OUT_PAST + 0.01         # wherever the stand-in is inside it
    assert rx + T.GLASS_R_IN < T.SLAB_X1 - 0.03                              # and it drops onto the outer sill
    carry = _phase("carry them out")
    assert _rim(carry[2]) - T.CARD_T - T.TOP > T.SILL_H + 0.015               # card's underside clears the sill
    pull = _phase("pull the card away")[3]
    assert pull[1] + T.CARD_Y - T.LIP_T / 2 < ry - T.GLASS_R                 # card's far edge past the rim
    low = _phase("lower them toward the outer sill")
    assert _rim(low[2]) - T.CARD_T - T.TOP > T.SILL_H + 0.005                 # card's underside over the window's sill
    assert T.RELEASE_UP + T.CARD_T - T.SLAB_H < 0.02                         # the stand-in drops less than 2 cm
    dropped_top = T.TOP + T.SLAB_H + T.BOT_H
    assert dropped_top > _rim(low[2]) + 0.01                                 # its top still 1 cm inside the glass
    assert _rim(_phase("raise the glass off it")[2]) > dropped_top + 0.01    # the glass lifts clear of it
    assert _rim(_phase("bring the glass")[2]) > dropped_top + 0.01           # and passes over it coming back in


def test_the_glass_push_brings_its_far_wall_against_the_stand_in():
    """The glass moves toward the card by GLASS_PUSH while the card slides in: its far inner wall ends up 0 to 5 mm
    past where the stand-in's side was, so it pushes the stand-in onto the card's edge rather than the card's edge
    pushing it along the desk."""
    wall = SY - T.GLASS_PUSH + T.GLASS_R_IN
    assert 0.0 < SY + T.BOT_R - wall < 0.005
    _, _, left, _ = _phase("slide the card in")
    assert left[1] - T.LEFT_AIM[1] == pytest.approx(SY - T.GLASS_PUSH)


def _hand_points(tcp, half_gap, axis):
    """Tool point, fingertip middle and the two fingertips of a straight-down grip closing along ``axis``."""
    tcp, side = np.asarray(tcp, float), np.asarray(axis, float) * half_gap
    tip = tcp - [0.0, 0.0, T.PAD_TIP_PAST_TCP]
    return [tcp, tip, tip + side, tip - side]


def test_demo_neither_sets_it_off_nor_touches_it_before_the_glass_is_down():
    """Until the glass is down over it, no hand point (tool points, fingertips, open or closed) comes within sight of
    its sensors or within TOUCH of its shell, at 11 points along every move; and while the glass comes down and the
    card goes in, no hand point comes within sight of it either (so it never drives during the demo)."""
    s, last = spider(), {"left": None, "right": None}
    for seconds, label, left, right in T.demo_phases():
        if label.startswith("rim down onto the card"):
            break
        starts = dict(last)
        last = {"left": left or last["left"], "right": right or last["right"]}
        for t in np.linspace(0.0, 1.0, 11):
            pts = []
            for arm, axis in (("left", (1, 0, 0)), ("right", (0, 1, 0))):
                a, b = starts[arm] or last[arm], last[arm]
                p = (1 - t) * np.asarray(a[:3], float) + t * np.asarray(b[:3], float)
                pts += _hand_points(p, 0.047, axis) + _hand_points(p, 0.0, axis)
            assert T.sensed(s, T.obstacle_points(card(), pts)) == (False, False), (label, t)
            assert T.touch_gap(s, pts) > T.TOUCH, (label, t)


def test_demo_is_within_reach():
    """Both arms follow every waypoint with the IK the scripted policy uses, within 2 mm and 1 degree."""
    from yam_arena import station
    from yam_arena.demo import ArmPlan

    plans = T.demo()
    for arm, waypoints in plans.items():
        plan = ArmPlan(waypoints, station.BASES[arm])
        for k in range(int(plan.duration / DT) + 2):
            plan.action7(k * DT)
        assert plan.worst_ik_mm < 2.0 and plan.worst_ik_deg < 1.0, (arm, plan.worst_ik_mm, plan.worst_ik_deg)
