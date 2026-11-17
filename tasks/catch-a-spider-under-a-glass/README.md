# Catch a spider under a glass, slide a card under it and let it out of the open window

A house spider sits on a desk under an open window. The robot has to get it outside without touching it and without
hurting it. On the desk are an upturned drinking glass and a stiff card. Nothing in the instruction says how.

Instruction: **"Catch the spider on the desk and put it out of the open window, without touching or hurting it."**

## Why this task is different

- **A target that runs.** The spider bolts if a hand comes within 5 cm, so it cannot be picked up or pushed. It has to
  be covered from above.
- **Gentle contact.** The glass must come down around the spider, not on it: a rim landing on its body ends the trial.
- **Bimanual insertion.** One hand lifts the rim a few millimetres while the other slides a 1.5 mm card in under it,
  without opening a gap the spider can get out through (8 mm).
- **A coordinated carry.** The two hands lift the glass and the card together and carry them over the window's sill
  without letting the rim come off the card.
- **Release and tidy up.** Outside, the card comes away, the spider drops onto the outer sill, and the glass and card
  come back in.

No accepted or pending task on the task board, and none in LIBERO, RoboCasa, SimplerEnv, BEHAVIOR, DataCenterBench or
RoboDojo, asks the robot to catch a small animal that runs from it, or to carry anything shut in between two loose
parts held by two hands. The closest on the board moves a toy ladybird that stays still from a pile of crumbs to a
garden pot: there the animal is simply picked up and carried.

## Where this comes up

Getting a spider or a bee out of a room is one of the most common small chores in a home, and one many people avoid:
they will not touch it and do not want to kill it. The glass and card is how most people do it. A home robot asked to
"get rid of the spider" has to choose the humane way and do it gently.

## Files

| File | What it is |
|---|---|
| `task.py` | the task file: scene, the spider's behaviour, scoring and the scripted demo |
| `bodykit.py` | rigid bodies made of several shapes (glass, card, spider, window), unchanged from tasks already in this repository |
| `test_scoring.py` | offline checks: geometry, the spider's behaviour, every scoring outcome, the demo's reach |
| `task.json` | the task's details |

## Requirements

The program's Brev launchable (Isaac Lab-Arena with `yam_arena` and Inspect Robots, L40S). Before every shell:
`source ~/IsaacLab-Arena/yam.env`. The Dream key sits in `~/.dreamlabs/credentials`. Nothing else: no downloaded
assets.

Tested with inspect-robots 1.0.0 and yam-arena at commit 0059d02a1a91 (`git -C ~/yam-isaac-lab-arena rev-parse
--short=12 HEAD`). `task.py` SHA-256, first 12 hex: 9bc4bc597c0e (`sha256sum task.py | cut -c1-12`).

## Run and score one Dream-1 trial

From this folder:

    inspect-robots run --embodiment yam_arena --policy dream \
      -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
      --instruction "Catch the spider on the desk and put it out of the open window, without touching or hurting it." \
      --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/spider-dream

Dream-1 refuses this instruction before the arms move, so this run records nothing and scores nothing:
`PolicyError: DreamAPI refused the inference request: DA-4000 BAD_REQUEST (malformed): unsafe instruction in this scene,
refused: The instruction involves catching a real spider, which counts as an animal.`

Scoring is automatic: the CLI prints `task_score: <value>`, and the log keeps it under `results.metrics.task_score`.
Set `SPIDER_DEBUG=1` to print the trial's state twice a second on stderr.

## Scripted demo

    inspect-robots run --embodiment yam_arena --policy yam_scripted \
      -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=1920 -E cam_height=1056 \
      --instruction "Catch the spider on the desk and put it out of the open window, without touching or hurting it." \
      --scorer task_score --max-steps 700 --store-frames --log-dir ~/yam-eval/spider-demo

The demo video is the top camera of this run. At 320 x 176 the demo scores the same.

`demo()` gives both arms Cartesian waypoints (fingers closing along x, pointing straight down), played open loop. Every
target comes from the layout constants in `task.py`; nothing is read from the simulator.

1. The left hand takes the glass by its upturned base and puts it down over the spider.
2. It lifts the rim a few millimetres. The right hand, holding the middle of the card's folded edge, slides the card in under
   the glass.
3. The glass is set down on the card. Both hands lift glass, card and spider 4 cm and carry them out over the sill.
4. Outside, the right hand pulls the card away and the spider drops onto the outer sill.
5. Both hands bring the glass and the card back in, set them down and let go.

About 29 s of motion, 580 control steps at 20 Hz.

## Scoring

Every condition is a distance or a height, measured in the simulation every control step (in a real trial: from the
video and a ruler).

- **Shut in:** the spider's centre is inside the glass, the glass stands on its rim (tilted less than 25 degrees), and
  the rim is at most 8 mm above whatever is under it (desk or card) all round.
- **0 for the whole trial** if the spider is ever touched (a gripper's tool point or fingertip within 2 cm of its
  centre while it is not shut in), squashed (the glass's rim comes down on its body), knocked out of the window (past
  the wall without having been carried there shut in) or escapes (gets behind x 5 cm, past 50 cm left or right, or off
  the desk), or if the glass falls off the desk or out of the window.
- **1.0** once the spider is out (carried out shut in, then its centre 1 cm past the wall's outer face and not shut in)
  with the glass and the card back inside the room (short of the wall's inner face, on the desk).
- **0.75** if it is out but the glass or the card is still outside, or if it was shut in on the card with the card's
  underside 3 cm above the desk.
- **0.5** if it was shut in with the card under the whole rim (the rim at most 4 mm above the card all round).
- **0.25** if it was ever shut in under the glass.
- **0** otherwise.

The trial ends with success after 1 s at full score with the glass and the card at rest.

## Verification

    python -m pytest test_scoring.py

27 checks, no simulator needed (only `yam_arena`, which the walkthrough's setup installs). They cover the geometry
(shut in, sealed, squashed, the surfaces), the spider's behaviour (sits still, bolts from a hand, stops at the wall,
escapes past the side, gets off a lifted or dragged card, keeps its place under the glass, steps onto the card slid in
under it), every scoring outcome, and that both arms reach every demo
waypoint with the scripted policy's IK (within 2 mm and 1 degree).

## Runtime and seeds

One layout, no randomness: run with `-E object_jitter=0`. Run limit 1,800 control steps (90 s at 20 Hz).

## Setup

Positions in cm from the midpoint between the two arm bases, on the table (x forward, y left).

- **Desk:** the station table, against the wall under a window.
- **Wall and window:** the wall's inner face runs across the desk at x 28.5, its outer face at x 31.5. The opening is
  90 cm wide (y -45 to 45), its frame's sill 2 cm above the desk. White frame, cream wall 50 cm high either side; the
  casement stands open outward on its hinge at the left. Outside: a grey stone outer sill (x 31.5 to 44, 1.2 cm thick)
  and a terracotta window box with plants (x 45 to 65, y -32 to 32, 9 cm tall).
- **Spider:** a house spider at x 20, y 6, facing forward: body 1.6 cm long, legs spanning 4 cm, dark brown.
- **Glass:** a straight clear 250 mL tumbler, 7.2 cm across, 9 cm tall, 0.22 kg, upside down at x 17, y 24.
- **Card:** a grey 1.5 mm greyboard card, flat face 25 cm (along y) by 15 cm (along x), with a blue picture printed over most of its top; its
  short edge on the right is folded up 2 cm and doubled (3 mm thick). Centred at x 19, y -14.
- Both arms start in the raised ready pose, grippers open.

## Fidelity

- **The spider** is a rigid model: one 9 mm collision sphere for its body, its legs drawn only. It weighs 3 g (a real
  one about 0.1 g) so the solver stays stable. Its behaviour is a rule run every control step: when not shut in, it
  bolts at 8 cm/s straight away from any gripper point within 5 cm, for at least 1 s and until none is within 10 cm;
  it stops at the wall (it never climbs out of the window by itself) and at a glass standing on the desk; and it gets
  off a card that is lifted more than 1 cm or moved faster than 3 cm/s while it is not shut in. Shut in, it keeps its
  place under the glass, and it steps onto the card when the card's edge slides in to within 3 mm of its body.
- **The glass** collides as a 24-sided wall and a base, but only its rim, its base's edge and six glints are drawn,
  so the spider shows through it. It does not break: a glass that falls off the desk scores 0.
- **The card** is rigid (it does not bend), and its fold is a second flat part.
- **Outside:** the station table continues under the outer sill and the window box.

## Real-world build

- A desk against a wall under an open window, or a desk with a 3 cm deep, 2 cm high wooden batten across it as the
  sill and a board behind it as the outer sill.
- A straight 250 mL glass tumbler (7.2 cm across, 9 cm tall, about 220 g).
- A 1.5 mm greyboard card cut 29 x 15 cm, one short edge folded up 2 cm and glued doubled.
- A soft rubber toy house spider (body 1.6 cm, legs spanning 4 cm, about 2 g) with a 0.2 mm nylon thread tied to its
  body, led away flat along the desk to an operator outside the robot's reach. Whenever a gripper comes within 5 cm of
  the spider while it is not under the glass, the operator pulls it 8 cm straight away from the gripper.
- Scoring from the video and a ruler. Nothing in the trial is hot, sharp or fast.

## Third-party assets

None. `bodykit.py` comes unchanged from tasks already in this repository.
