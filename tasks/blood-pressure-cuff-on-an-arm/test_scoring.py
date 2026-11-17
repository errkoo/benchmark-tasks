"""Offline checks of the scene, the joints, the scoring geometry and the scripted demo's reach (no simulator):
python -m pytest test_scoring.py"""

from __future__ import annotations

import itertools
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import bodykit  # noqa: E402
import task as T  # noqa: E402
from bodykit import BoxShape, CylShape, SphereShape, rotation, state_from_pose  # noqa: E402

DT = 0.05


def start_states() -> dict:
    return {o.name: state_from_pose(o, o.pos, (0.0, 0.0, math.sin(o.yaw / 2), math.cos(o.yaw / 2))) for o in T.OBJECTS}


def boxes(spec) -> list:
    """World-aligned boxes (lo, hi) round each colliding shape of a body at its start pose."""
    r = rotation((0.0, 0.0, math.sin(spec.yaw / 2), math.cos(spec.yaw / 2)))
    out = []
    for s in spec.shapes:
        if not s.collide:
            continue
        if isinstance(s, BoxShape):
            half = np.abs(r @ rotation(s.quat())) @ (np.array(s.size) / 2)
        elif isinstance(s, CylShape):
            half = np.abs(r) @ s.half_extent()
        else:
            assert isinstance(s, SphereShape)
            half = s.half_extent()
        c = np.asarray(spec.pos) + r @ np.asarray(s.centre)
        out.append((c - half, c + half))
    return out


def joined() -> set:
    pairs = {(h.body0, h.body1) for h in bodykit._HINGES} | {(sl.body0, sl.body1) for sl in bodykit._SLIDERS}
    pairs |= {(k.body0, k.body1) for k in T.LINKS} | {(b.body0, b.body1) for b in T.BENDS}
    return pairs | {(b, a) for a, b in pairs}


def wrapped(s_bottom: float = 0.27, x: float | None = None, gap: float = 0.0, turn_deg: float = 0.0) -> dict:
    """Start states, with the cuff wound round the upper arm as the demo means to: its strip at length s_bottom under
    the arm, the right end's part up the body side and over, the left end's part over that (one cuff thickness out),
    the middle of its width at x, ``gap`` between the skin and the inner turn, all turned ``turn_deg`` round."""
    x = T.XC if x is None else x
    out = start_states()
    r_in = T.upper_arm_radius(x) + gap + T.CUFF_T / 2
    r_out = r_in + T.CUFF_T
    a0 = math.pi - s_bottom / r_in                           # the right end's angle, from the top toward the body (-y)
    inner = 2 * math.pi * r_in - 0.01                        # the length of the inner turn
    for i in range(T.CUFF_N):
        s = (i + 0.5) * T.SEG
        a, r = (a0 + s / r_in, r_in) if s <= inner else (a0 + inner / r_in + (s - inner) / r_out, r_out)
        a += math.radians(turn_deg)
        rot = np.array([[1.0, 0.0, 0.0], [0.0, -math.cos(a), math.sin(a)], [0.0, -math.sin(a), -math.cos(a)]])
        pos = (x, -r * math.sin(a), T.ARM_Z + r * math.cos(a))
        out[f"cuff{i}"] = state_from_pose(T.CUFF[i], pos, bodykit._quat_from_matrix(rot))
    return out


@pytest.fixture(autouse=True)
def fresh_state():
    T._clear()
    yield
    T._clear()


# -- the task file and the scene -----------------------------------------------------------------------------------------

def test_task_file_loads():
    from yam_arena import task_file

    loaded = task_file.load(pathlib.Path(T.__file__))
    assert loaded.names[:4] == ("arm", "monitor", "inflate_button", "start_button") and loaded.demo is not None
    assert len(loaded.names) == 4 + T.CUFF_N + T.HOSE_N


def test_nothing_overlaps_at_the_start():
    """No two bodies that are not joined start inside each other (world-aligned boxes round each shape, 0.3 mm grace),
    and nothing starts inside the table."""
    pairs = joined()
    for a, b in itertools.combinations(T.OBJECTS, 2):
        if (a.name, b.name) in pairs:
            continue
        for (lo1, hi1), (lo2, hi2) in itertools.product(boxes(a), boxes(b)):
            gap = np.maximum(lo1 - hi2, lo2 - hi1)
            assert gap.max() > -0.0003, (a.name, b.name, gap)
    for o in T.OBJECTS:
        for lo, _ in boxes(o):
            assert lo[2] >= T.TOP - 1e-6, o.name


def test_cuff_lies_open_under_the_arm_and_fits_it():
    ys = np.array([c.pos[1] for c in T.CUFF])
    assert np.allclose([c.pos[0] for c in T.CUFF], T.CUFF_X) and np.allclose(np.diff(ys), T.SEG)
    assert ys[0] < -0.2 and ys[-1] > 0.2                                      # it reaches out past the arm both ways
    for x in np.linspace(T.CUFF_X - T.CUFF_W / 2, T.CUFF_X + T.CUFF_W / 2, 9):
        under = T.ARM_Z - T.upper_arm_radius(x) - (T.CUFF_Z + T.CUFF_T / 2)
        assert 0.005 < under < 0.015                                         # the arm 0.5 to 1.5 cm above it
    round_mid = 2 * math.pi * T.upper_arm_radius(T.XC)
    assert 0.25 <= round_mid <= 0.34                                         # the arm is in the adult 11 cuff's range
    assert T.CUFF_L - round_mid >= T.CUFF_L - T.HOOK_S + 0.08                 # the whole hook strip lands on the cuff
    assert T.BLADDER[0] <= T.MARKER_S <= T.BLADDER[1]
    lower = T.X_CREASE - (T.CUFF_X + T.CUFF_W / 2)
    assert lower > T.EDGE_ABOVE[1] + 0.01                                    # it starts too far up the arm
    assert T.EDGE_ABOVE[0] < T.X_CREASE - (T.XC + T.CUFF_W / 2) < T.EDGE_ABOVE[1]


def test_the_cuff_strips_are_hinged_end_to_end_and_curl_at_the_ends():
    assert [b.name for b in T.BENDS] == [f"bend{i}" for i in range(T.CUFF_N - 1)]
    assert not [h for h in bodykit._HINGES if h.name.startswith("bend")]
    for i, b in enumerate(T.BENDS):
        assert b.body0 == f"cuff{i}" and b.body1 == f"cuff{i + 1}"
        assert b.target == (T.CURL_TARGET if i in (0, 1, T.CUFF_N - 3, T.CUFF_N - 2) else T.BEND_TARGET)


def test_only_the_bladder_draws_up_and_it_can_take_up_the_slack():
    for b in T.BENDS:
        s = (int(b.name[4:]) + 1) * T.SEG                    # where along the cuff the joint is
        assert b.squeeze == (T.BLADDER[0] < s <= T.BLADDER[1])
    loop = 10 * T.SEG                                         # run bp-2: the hook strip stuck on the head, 40 cm round
    round_mid = 2 * math.pi * (T.upper_arm_radius(T.XC) + T.CUFF_T / 2)
    assert len(T.PUMP_NAMES) * T.SQUEEZE >= loop - round_mid + 0.02
    assert T.SQUEEZE < T.SEG / 2 + 1e-9                       # a strip never closes up past the one beyond
    assert T.squeeze_target(None) == 0.0 and T.squeeze_target(0.0) == 0.0
    assert T.squeeze_target(T.PUMP_S / 2) == pytest.approx(-T.SQUEEZE / 2)
    assert T.squeeze_target(T.PUMP_S * 3) == pytest.approx(-T.SQUEEZE)


def test_curled_ends_stand_clear_of_the_arm():
    for side in (-1, 1):
        y, z = T.curled_end(side)
        assert abs(y) > 0.15 and T.TOP + 0.05 < z < T.TOP + 0.09


def test_hose_runs_from_the_tube_to_the_socket():
    pts = T.hose_points()
    assert np.allclose(pts[0][:2], T.HOSE_PATH[0]) and np.allclose(pts[-1][:2], T.SOCKET)
    tube = T.tube_point([start_states()[n] for n in T.CUFF_NAMES])
    assert np.linalg.norm(tube - pts[0]) < 1e-9
    length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    assert 0.50 < length < 0.65
    for k, h in enumerate(T.HOSE):                                    # each piece spans its two joints
        r = rotation((0.0, 0.0, math.sin(h.yaw / 2), math.cos(h.yaw / 2)))
        half = h.shapes[0].height / 2
        assert np.allclose(np.asarray(h.pos) - r @ [half, 0, 0], pts[k], atol=1e-9)
        assert np.allclose(np.asarray(h.pos) + r @ [half, 0, 0], pts[k + 1], atol=1e-9)
    worn = T.tube_point([wrapped()[n] for n in T.CUFF_NAMES])
    assert length - np.linalg.norm(worn - pts[-1]) > 0.10                # 10 cm of slack with the cuff on the arm


def test_link_frames_meet_at_the_anchor_lined_up():
    for link in T.LINKS:
        p0, r0, p1, r1 = T.link_frames(link, T.SPECS)
        b0, b1 = T.SPECS[link.body0], T.SPECS[link.body1]
        R0, R1 = T._start_rotation(b0), T._start_rotation(b1)
        assert np.allclose(np.asarray(b0.pos) + R0 @ p0, link.anchor, atol=1e-9)
        assert np.allclose(np.asarray(b1.pos) + R1 @ p1, link.anchor, atol=1e-9)
        assert np.allclose(R0 @ r0, np.eye(3), atol=1e-9) and np.allclose(R1 @ r1, np.eye(3), atol=1e-9)
    names = [o.name for o in T.OBJECTS]
    for link in T.LINKS:
        assert names.index(link.body0) < names.index(link.body1)


def test_every_hook_and_loop_pair_is_declared_and_ordered():
    glues = {g.name: g for g in bodykit._GLUES}
    names = [o.name for o in T.OBJECTS]
    for k, j in T.STICKS:
        g = glues[f"stick_{k}_{j}"]
        assert names.index(g.body0) < names.index(g.body1) and g.body1 == f"cuff{k}"


def test_buttons_rest_out_and_sit_on_the_monitor():
    st = start_states()
    for n in T.BUTTONS:
        assert abs(T.button_travel(st["monitor"], st[f"{n}_button"])) < 1e-9
        x, y, z = T.button_pos(n)
        assert T.MONITOR.pos[2] + T.MON_H / 2 < z + T.BUTTON_H / 2                   # standing proud of the top
        assert abs(x - T.MON_XY[0]) + T.BUTTON_R < T.MON_L / 2 and abs(y - T.MON_XY[1]) + T.BUTTON_R < T.MON_W / 2
    sliders = {sl.name: sl for sl in bodykit._SLIDERS}
    assert set(sliders) == set(T.BUTTONS)
    (_, y1, _), (_, y2, _) = T.button_pos("inflate"), T.button_pos("start")
    assert abs(y1 - y2) > 2 * T.BUTTON_R + 0.02                                  # a fingertip presses one only


# -- geometry and scoring --------------------------------------------------------------------------------------------------

def test_around_the_arm():
    a, deg, r = T.around((T.X_CREASE - 0.1, 0.0, T.ARM_Z + 0.05))
    assert abs(a - 0.1) < 1e-9 and abs(deg) < 1e-9 and abs(r - 0.05) < 1e-9
    _, deg, _ = T.around((T.X_CREASE, -0.05, T.ARM_Z))
    assert abs(deg - 90.0) < 1e-9


def test_the_open_cuff_at_the_start_scores_nothing():
    st = start_states()
    cuff = [st[n] for n in T.CUFF_NAMES]
    assert T.winding(cuff) < 0.5 and not T.on_arm(cuff) and T.score(st) == 0.0


def test_a_wrapped_cuff_turns_round_the_arm_with_its_bladder_inside():
    st = wrapped()
    cuff = [st[n] for n in T.CUFF_NAMES]
    assert 1.5 < T.winding(cuff) < 1.95 and T.on_arm(cuff) and T.tight(cuff, {10: 0, 11: 0})
    assert max(T.skin_gap(c) for c in cuff) < 0.005
    edges = T.edge_heights(cuff)
    assert abs(min(edges) - 0.025) < 1e-6 and abs(max(edges) - 0.025) < 1e-6
    for k in T.TAIL:                                             # the hook strip lies on the inner turn's loop face
        assert any(T.facing(cuff[k], cuff[j]) for j in T.UNDER), k


def test_score_parts():
    st = wrapped()
    assert T.score(st) == pytest.approx(0.20)                    # round the arm at the right height, not closed
    T._STATE["stuck"] = {10: (3, 0), 11: (4, 0)}
    assert T.score(st) == pytest.approx(0.60)                    # closed
    T._STATE["inflated_at"] = 1.0
    assert T.score(st) == pytest.approx(0.80)                    # inflated, tight on the arm
    T._STATE.update(start_at=2.0, t=3.5)
    assert T.score(st) == pytest.approx(0.80) and not T.success(st)          # reading, 1.5 s in
    T._STATE["t"] = 4.0
    assert T.score(st) == pytest.approx(1.0) and T.success(st)               # read
    T._STATE["touched"] = True
    assert T.score(st) == pytest.approx(0.80)                    # touched while it read
    T._STATE["touched"] = False
    assert T.score(wrapped(x=T.XC - 0.03)) == pytest.approx(0.80)          # 5.5 cm above the crease: no height
    assert T.score(wrapped(gap=0.03)) == pytest.approx(0.60)               # inflated but still 3 cm loose
    assert T.score(wrapped(gap=0.08)) == 0.0                               # hanging 8 cm off it: not on the arm
    flap = wrapped()
    flap["cuff12"] = wrapped(gap=0.05)["cuff12"]
    assert T.score(flap) == pytest.approx(1.0)                   # the free end past the hook strip may stand off
    T._STATE["stuck"] = {10: (3, 0)}
    assert T.score(st) == pytest.approx(0.20)                    # 4 cm of hook stuck: not closed, nothing after
    T._STATE["stuck"] = {10: (3, 0), 11: (4, 0)}
    tipped = dict(st)
    m = T.MONITOR
    tipped["monitor"] = state_from_pose(m, (m.pos[0], m.pos[1], T.TOP - 0.2), (0.0, 0.0, 0.0, 1.0))
    assert T.score(tipped) == 0.0                                # the monitor pulled off the table


def press(t: float, ok: bool = True, ftouch: float = 0.0, **down) -> None:
    T._STATE["t"] = t
    T._monitor_rules({n: down.get(n, False) for n in T.BUTTONS}, ok, ftouch)


def test_the_monitor_needs_the_cuff_closed_then_inflate_then_start():
    press(1.0, start=True)
    press(1.1)
    assert T._STATE["start_at"] is None                          # START before INFLATE: nothing
    press(2.0, ok=False, inflate=True)
    press(2.1, ok=False)
    assert T._STATE["inflated_at"] is None                       # INFLATE with the cuff open: nothing
    press(3.0, inflate=True)
    press(3.5, inflate=True)
    press(3.6)
    press(3.7, inflate=True)
    assert T._STATE["inflated_at"] == 3.0                        # it inflates once
    press(4.0, start=True)
    assert T._STATE["start_at"] == 4.0 and not T._STATE["touched"]
    press(5.0, ftouch=T.TOUCH_N + 0.1)
    assert T._STATE["touched"]                                   # a touch in the reading's 2 s spoils it
    press(6.0)
    press(6.1, start=True)
    assert T._STATE["start_at"] == 6.1 and not T._STATE["touched"]          # START again: a new reading
    press(8.5, ftouch=5.0)
    assert not T._STATE["touched"]                               # a touch after the reading is done


def test_marker_and_tube_on_the_strips_at_the_start():
    cuff = [start_states()[n] for n in T.CUFF_NAMES]
    m = T.marker_point(cuff)
    assert np.allclose(m, (T.CUFF_X, T.CUFF_Y0 + T.MARKER_S, T.CUFF_Z - T.CUFF_T / 2))


# -- the scripted demo -------------------------------------------------------------------------------------------------

def test_demo_is_within_reach():
    """Both arms follow every waypoint with the IK the scripted policy uses, within 2 mm and 1 degree."""
    from yam_arena import station
    from yam_arena.demo import ArmPlan

    for arm, waypoints in T.demo().items():
        plan = ArmPlan(waypoints, station.BASES[arm])
        for k in range(int(plan.duration / DT) + 2):
            plan.action7(k * DT)
        assert plan.worst_ik_mm < 2.0 and plan.worst_ik_deg < 1.0, (arm, plan.worst_ik_mm, plan.worst_ik_deg)


def test_demo_presses_inflate_then_start_and_ends_hands_off():
    phases = T.demo_phases()
    labels = [p[1] for p in phases]
    assert len(set(labels)) == len(labels)
    by = {p[1]: p for p in phases}
    assert labels.index("right: press inflate") < labels.index("right: press start")
    t_inflate = sum(p[0] for p in phases[:labels.index("right: press inflate") + 1])
    t_start = sum(p[0] for p in phases[:labels.index("right: press start")])
    assert t_start - t_inflate >= T.PUMP_S + 0.5                 # START once the cuff has filled
    button_top = T.BUTTON_Z + T.BUTTON_H / 2
    for name in T.BUTTONS:
        x, y, z, _ = by[f"right: press {name}"][3]
        assert np.allclose((x, y), T.button_pos(name)[:2])
        depth = button_top - (z - T.PAD_TIP_PAST_TCP)
        assert T.PRESSED + 0.003 < depth < T.BUTTON_TRAVEL + 0.004                # past "pressed" with room for sag
    after = labels.index("right: press start") + 1
    held = sum(p[0] for p in phases[after:])
    assert held >= T.READ_S + T.SUCCESS_HOLD_S + 1.0             # it reads and holds before the demo ends
    assert phases[-1][2] is None and phases[-1][3] is None
