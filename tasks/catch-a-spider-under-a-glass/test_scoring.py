"""Offline checks of the geometry, the spider's behaviour, the scoring rule and the scripted demo's reach (no simulator):
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


def st(spec, xy, z, quat=UP, vel=(0.0, 0.0, 0.0)):
    return state_from_pose(spec, (xy[0], xy[1], z), quat, vel)


def spider(xy=T.SPIDER_XY, z=T.TOP + 0.0003):
    return st(T.SPIDER, xy, z)


def glass(xy=T.SPIDER_XY, rim=T.TOP + 0.0005, quat=UP):
    return st(T.GLASS, xy, rim, quat)


def card(xy=T.CARD_XY, z=T.TOP + 0.0003, vel=(0.0, 0.0, 0.0)):
    return st(T.CARD, xy, z, vel=vel)


GLASS_AWAY = glass(T.GLASS_XY)


@pytest.fixture(autouse=True)
def fresh():
    T._clear()
    yield
    T._clear()


def objects(g=GLASS_AWAY, c=None, s=None):
    return {"glass": g, "card": c or card(), "spider": s or spider(), "window": None}


# -- the task file loads --------------------------------------------------------------------------------------------

def test_task_file_loads():
    from yam_arena import task_file

    loaded = task_file.load(pathlib.Path(T.__file__))
    assert loaded.names == ("window", "card", "glass", "spider") and loaded.demo is not None
    assert set(loaded.free_objects) == {"window", "card", "glass", "spider"}     # bodykit bodies all carry a mass


# -- geometry -------------------------------------------------------------------------------------------------------

def test_spider_free_at_the_start_and_shut_in_under_the_glass():
    assert not T.shut_in(spider(), GLASS_AWAY, card())
    assert T.shut_in(spider(), glass(), card())


def test_a_raised_rim_lets_it_out():
    assert T.shut_in(spider(), glass(rim=T.TOP + 0.007), card())         # 7 mm: still shut in
    assert not T.shut_in(spider(), glass(rim=T.TOP + 0.010), card())     # 1 cm: it can get out


def test_glass_on_its_side_shuts_nothing_in():
    side = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))      # 90 degrees about x
    assert not T.shut_in(spider(), glass(quat=side), card())


def test_rim_on_the_body_squashes_it_but_a_wall_beside_it_does_not():
    on_top = glass((T.SPIDER_XY[0] + T.R_MID, T.SPIDER_XY[1]), rim=T.TOP + 2 * T.SPIDER_R)   # rim sitting on its back
    assert T.squashed_now(spider(), on_top)
    beside_in = glass((T.SPIDER_XY[0] + T.GLASS_R_IN - T.SPIDER_R - 0.0005, T.SPIDER_XY[1]))  # touching inside
    beside_out = glass((T.SPIDER_XY[0] + T.GLASS_R + T.SPIDER_R + 0.0005, T.SPIDER_XY[1]))    # touching outside
    assert not T.squashed_now(spider(), beside_in) and not T.squashed_now(spider(), beside_out)
    assert not T.squashed_now(spider(), glass(rim=T.TOP + 0.03, xy=(T.SPIDER_XY[0] + T.R_MID, T.SPIDER_XY[1])))


def test_card_under_the_glass_seals_it():
    c = card(T.SPIDER_XY)
    s = spider(z=T.TOP + 0.0003 + T.CARD_T)
    g = glass(rim=T.TOP + 0.0003 + T.CARD_T + 0.0005)
    assert T.sealed(s, g, c) and T.on_card(s, c)
    half = card((T.SPIDER_XY[0], T.SPIDER_XY[1] - T.CARD_Y / 2))         # card only part way in
    assert T.shut_in(spider(), glass(rim=T.TOP + 0.006), half) and not T.sealed(spider(), glass(), half)


def test_surfaces():
    assert T.surface_z(0.2, 0.0) == T.TOP
    assert T.surface_z(0.30, 0.0) == pytest.approx(T.TOP + T.SILL_H)
    assert T.surface_z(0.36, 0.0) == pytest.approx(T.TOP + T.SLAB_H)
    assert T.surface_z(0.80, 0.0) < 0


# -- the spider's behaviour -----------------------------------------------------------------------------------------

def test_sits_still_while_no_hand_is_near():
    assert T.update(spider(), GLASS_AWAY, card(), FAR, DT) is None


def test_bolts_away_from_a_hand_within_5_cm():
    hand = np.array([T.SPIDER_XY[0] + 0.03, T.SPIDER_XY[1], T.TOP + 0.02])         # 3.6 cm, from the front
    pos, yaw = T.update(spider(), GLASS_AWAY, card(), [hand], DT)
    assert pos[0] < T.SPIDER_XY[0] and abs(math.cos(yaw) + 1.0) < 1e-6           # ran straight back, away from it
    assert pos[0] == pytest.approx(T.SPIDER_XY[0] - T.RUN_SPEED * DT)


def test_stops_at_the_wall_and_never_climbs_out_by_itself():
    at_wall = spider((T.X_IN - 0.011, 0.0))
    hand = np.array([T.X_IN - 0.05, 0.0, T.TOP + 0.02])                             # chasing it toward the window
    pos, _ = T.update(at_wall, GLASS_AWAY, card(), [hand], DT)
    assert pos[0] <= T.X_IN - 0.010 and not T._STATE["out"]


def test_running_past_the_side_is_an_escape():
    edge = spider((0.20, T.DESK_Y - 0.002))
    hand = np.array([0.20, T.DESK_Y - 0.04, T.TOP + 0.02])
    assert T.update(edge, GLASS_AWAY, card(), [hand], DT) == "gone"
    assert T._STATE["escaped"] and T.score(objects()) == 0.0


def test_gets_off_an_uncovered_card_that_is_lifted():
    c = card(T.SPIDER_XY, z=T.TOP + 0.02)
    s = spider(z=T.TOP + 0.02 + T.CARD_T)
    pos, _ = T.update(s, GLASS_AWAY, c, FAR, DT)
    assert pos[2] == pytest.approx(T.TOP + 0.0003) and T._STATE["run"] > 0       # down on the desk, running


def test_shut_in_it_keeps_its_place_under_the_glass_and_goes_with_it():
    sx, sy = T.SPIDER_XY
    T.update(spider(), glass(), card(), FAR, DT)
    pos, _ = T.update(spider(), glass((sx + 0.004, sy)), card(), FAR, DT)          # the glass slid 4 mm
    assert pos[0] == pytest.approx(sx + 0.004) and pos[2] == pytest.approx(T.TOP + 0.0003)


def test_shut_in_it_steps_onto_the_card_sliding_in_under_the_rim():
    sx, sy = T.SPIDER_XY
    g = glass(rim=T.TOP + 0.006)
    T.update(spider(), g, card(), FAR, DT)
    edge_at = lambda d: card((sx, sy - d - T.CARD_Y / 2))                          # noqa: E731  its +y edge d short
    pos, _ = T.update(spider(), g, edge_at(0.02), FAR, DT)                         # 2 cm short: stays on the desk
    assert pos[1] == pytest.approx(sy) and pos[2] == pytest.approx(T.TOP + 0.0003)
    pos, _ = T.update(spider(), g, edge_at(0.005), FAR, DT)                        # 5 mm short: steps onto it
    assert pos[1] == pytest.approx(sy - 0.006) and pos[2] == pytest.approx(T.TOP + 0.0003 + T.CARD_T + 0.0003)
    s = spider((sx, sy - 0.006), z=T.TOP + 0.0006 + T.CARD_T)
    pos, _ = T.update(s, g, edge_at(-0.03), FAR, DT)                               # the card goes on: it stays put
    assert pos[1] == pytest.approx(sy - 0.006) and T.on_card(spider(pos[:2], pos[2]), edge_at(-0.03))


def test_gets_off_an_uncovered_card_that_is_dragged():
    c = card(T.SPIDER_XY, vel=(0.0, 0.05, 0.0))
    s = spider(z=T.TOP + 0.0003 + T.CARD_T)
    assert T.update(s, GLASS_AWAY, c, FAR, DT) is not None


# -- scoring --------------------------------------------------------------------------------------------------------

def _run(steps):
    """Feed (spider, glass, card, hands) states through update()."""
    for s, g, c, h in steps:
        T.update(s, g, c, h, DT)


def _story(release_gx=T.RELEASE_XY[0]):
    """The demo's story as states: shut in, card in, sealed, lifted, carried out, let go."""
    sx, sy = T.SPIDER_XY
    lift = T.CARRY_UP
    c_in, c_up = card(T.SPIDER_XY), card(T.SPIDER_XY, z=T.TOP + 0.0003 + lift)
    s_on, s_up = spider(z=T.TOP + 0.0003 + T.CARD_T), spider(z=T.TOP + 0.0003 + T.CARD_T + lift)
    g_on, g_up = glass(rim=T.TOP + 0.0003 + T.CARD_T + 0.0005), glass(rim=T.TOP + 0.0003 + T.CARD_T + 0.0005 + lift)
    out = (release_gx, T.RELEASE_XY[1])
    return [
        (spider(), glass(), card(), FAR),                                 # shut in on the desk
        (spider(), glass(rim=T.TOP + 0.006), card((sx, sy - 0.06)), FAR),   # card half in
        (s_on, g_on, c_in, FAR),                                          # sealed
        (s_up, g_up, c_up, FAR),                                          # lifted
        (spider(out, z=T.TOP + T.SLAB_H + 0.0003), glass(out, rim=T.TOP + 0.045), card((out[0], out[1] - 0.13),
                                                                                       z=T.TOP + 0.044), FAR),
    ]


def test_full_story_scores_one_once_glass_and_card_are_back_in():
    _run(_story())
    assert T._STATE["out"]
    assert T.score(objects(glass(T.RELEASE_XY, rim=T.TOP + 0.045), card((0.36, -0.05), z=T.TOP + 0.044))) == 0.75
    assert T.score(objects(GLASS_AWAY, card())) == 1.0


def test_milestones():
    story = _story()
    _run(story[:1])
    assert T.score(objects()) == 0.25
    _run(story[1:3])
    assert T.score(objects()) == 0.5
    _run(story[3:4])
    assert T.score(objects()) == 0.75


def test_shut_in_once_is_kept_as_partial_credit():
    _run(_story()[:1])
    _run([(spider(), GLASS_AWAY, card(), FAR)])
    assert T.score(objects()) == 0.25


def test_knocked_out_of_the_window_without_being_carried_scores_zero():
    _run([(spider((0.34, 0.0), z=T.TOP + T.SLAB_H + 0.0003), GLASS_AWAY, card(), FAR)])
    assert T._STATE["flung"] and T.score(objects()) == 0.0


def test_touching_it_scores_zero():
    _run(_story()[:1])
    hand = np.array([*T.SPIDER_XY, T.TOP + 0.01])
    _run([(spider(), GLASS_AWAY, card(), [hand])])
    assert T._STATE["touched"] and T.score(objects()) == 0.0


def test_a_hand_on_the_glass_does_not_touch_a_spider_shut_in_under_it():
    hand = np.array([*T.SPIDER_XY, T.TOP + 0.015])
    _run([(spider(), glass(), card(), [hand])])
    assert not T._STATE["touched"] and T.score(objects()) == 0.25


def test_squashing_it_scores_zero():
    on_top = glass((T.SPIDER_XY[0] + T.R_MID, T.SPIDER_XY[1]), rim=T.TOP + 2 * T.SPIDER_R)
    _run([(spider(), on_top, card(), FAR)])
    assert T._STATE["squashed"] and T.score(objects()) == 0.0


def test_glass_off_the_desk_scores_zero():
    _run(_story()[:4])
    _run([(spider(), glass((0.5, 0.0), rim=0.2), card(), FAR)])
    assert T._STATE["broken"] and T.score(objects()) == 0.0


def test_nothing_done_scores_zero():
    _run([(spider(), GLASS_AWAY, card(), FAR)] * 5)
    assert T.score(objects()) == 0.0 and not T.success(objects())


# -- the scripted demo ----------------------------------------------------------------------------------------------

def _hand_points(tcp, half_gap, axis):
    """Tool point, fingertip middle and the two fingertips of a straight-down grip closing along ``axis``."""
    tcp, side = np.asarray(tcp, float), np.asarray(axis, float) * half_gap
    tip = tcp - [0.0, 0.0, T.PAD_TIP_PAST_TCP]
    return [tcp, tip, tip + side, tip - side]


def test_demo_story_fits_the_layout():
    phases = T.demo_phases()
    labels = [p[1] for p in phases]
    slide = phases[labels.index("slide the card in under the glass")][3]
    pinch_end_y = slide[1]
    card_y = pinch_end_y + T.CARD_Y / 2 - T.LIP_T / 2
    sx, sy = T.SPIDER_XY
    assert card_y + T.CARD_Y / 2 >= sy + T.GLASS_R + 0.009                    # its leading edge past the far side
    assert card_y - T.CARD_Y / 2 <= sy - T.GLASS_R - 0.03                     # its other edge well short of the near side
    assert abs(T.CARD_XY[0] - sx) + T.GLASS_R < T.CARD_X / 2 - 0.02           # the rim over it in x, 2 cm to spare
    rx = T.RELEASE_XY[0]
    assert rx - T.GLASS_R > T.X_OUT + 0.02                                   # the whole glass past the wall
    assert rx - T.GLASS_R_IN + T.SPIDER_R > T.X_OUT + T.OUT_PAST + 0.01      # wherever the spider is inside it
    carry = phases[labels.index("carry them out over the sill")]
    rim = carry[2][2] - T.GLASS_H + T.GRIP_BELOW_TOP
    assert rim - T.CARD_T - T.TOP > T.SILL_H + 0.015                         # card's underside clears the sill
    pull = phases[labels.index("pull the card away")][3]
    assert pull[1] + T.CARD_Y - T.LIP_T / 2 < T.RELEASE_XY[1] - T.GLASS_R   # card's far edge past the rim


def test_demo_hands_never_come_near_the_free_spider():
    """Until the glass is down over it, no fingertip or tool point comes within STARTLE of the spider."""
    centre = np.array([*T.SPIDER_XY, T.TOP + 0.0003 + T.SPIDER_R])
    for seconds, label, left, right in T.demo_phases():
        for target, half, axis in ((left, T.GLASS_R + 0.001, (1, 0, 0)), (right, 0.047, (0, 1, 0))):
            if target is None:
                continue
            for p in _hand_points(target[:3], half, axis):
                assert np.linalg.norm(p - centre) > T.STARTLE, label
        if label == "lower it over the spider":
            break


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
