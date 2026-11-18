# Catch a spider under a glass, slide a card under it and let it out of the open window

A house spider sits on a desk under an open window. The robot has to get it outside without touching it and without
hurting it. On the desk are an upturned drinking glass and a stiff card. Nothing in the instruction says how.

Instruction: **"Catch the spider on the desk and put it out of the open window, without touching or hurting it."**

A live spider cannot be used in repeatable trials, so the spider is a stand-in that runs from the robot by itself: an
off-the-shelf Ozobot Evo robot dressed as a spider, running a five-line program (see "The spider stand-in" below). The
simulated spider is a model of that robot and that program.

## Why this task is different

- **A target that runs.** The stand-in backs away from a gripper its sensors see within 5 cm in front of it and drives
  on from one behind it, at 8 cm/s. It cannot be picked up or pushed into place: it has to be covered from above.
- **Gentle contact.** The glass must come down around it, not on it: a rim landing on it ends the trial.
- **Bimanual insertion.** One hand lifts the rim a few millimetres while the other slides a 1.5 mm card in under it.
  As the card comes in, the glass's far wall pushes the stand-in onto the card's edge, without opening a gap it could
  get out through (8 mm).
- **A coordinated carry.** The two hands lift the glass and the card together and carry them over the window's sill
  without letting the rim come off the card. The scorer checks the spider is shut in the whole way out.
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
| `task.py` | the task file: scene, the stand-in's program, scoring and the scripted demo |
| `bodykit.py` | rigid bodies made of several shapes (glass, card, stand-in, window), unchanged from tasks already in this repository |
| `test_scoring.py` | offline checks: geometry, the stand-in's program and wheels, every scoring outcome, the demo's layout and reach |
| `task.json` | the task's details |

## Requirements

The program's Brev launchable (Isaac Lab-Arena with `yam_arena` and Inspect Robots, L40S). Before every shell:
`source ~/IsaacLab-Arena/yam.env`. The Dream key sits in `~/.dreamlabs/credentials`. Nothing else: no downloaded
assets.

Tested with inspect-robots 1.0.0 and yam-arena at commit 0059d02a1a91 (`git -C ~/yam-isaac-lab-arena rev-parse
--short=12 HEAD`). `task.py` SHA-256, first 12 hex: f2dc05145706 (`sha256sum task.py | cut -c1-12`).

## Run and score one Dream-1 trial

From this folder:

    inspect-robots run --embodiment yam_arena --policy dream \
      -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
      --instruction "Catch the spider on the desk and put it out of the open window, without touching or hurting it." \
      --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/spider-dream

Dream-1 refuses this instruction before the arms move, so this run records nothing and scores nothing:
`PolicyError: DreamAPI refused the inference request: DA-4000 BAD_REQUEST (malformed): instruction invades privacy in
this scene, refused: Instruction requires exploring or moving outside the tabletop work area.`

Scoring is automatic: the CLI prints `task_score: <value>`, and the log keeps it under `results.metrics.task_score`.
Set `SPIDER_DEBUG=1` to print the trial's state twice a second on stderr (what the stand-in's sensors see, whether it
is driving, the carry flags, where everything is).

## Scripted demo

    inspect-robots run --embodiment yam_arena --policy yam_scripted \
      -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=1920 -E cam_height=1056 \
      --instruction "Catch the spider on the desk and put it out of the open window, without touching or hurting it." \
      --scorer task_score --max-steps 700 --store-frames --log-dir ~/yam-eval/spider-demo

The demo video is the top camera of this run. At 320 x 176 the demo scores the same.

`demo()` gives both arms Cartesian waypoints (pointing straight down), played open loop. Every target comes from the
layout constants in `task.py`; nothing is read from the simulator. The hands stay out of the stand-in's sight the whole
time (above 3.5 cm or more than 5 cm away), so it never drives during the demo.

1. The left hand takes the glass by its upturned base and puts it down over the stand-in.
2. It lifts the rim a few millimetres. The right hand, holding the middle of the card's folded edge, slides the card in
   under the glass while the left hand moves the glass 2 cm toward the card: the glass's far wall pushes the stand-in
   onto the card's edge, and the card goes on in under it.
3. The glass is set down on the card. Both hands lift glass, card and stand-in 4 cm and carry them out over the sill.
4. Outside, both hands lower them until the card is 8 mm over the window's sill. The right hand pulls the card
   away and the stand-in drops 1.8 cm onto the outer sill, its top still inside the glass, which keeps it upright.
   The glass is raised clear of it.
5. Both hands bring the glass and the card back in, set them down and let go.

About 32 s of motion, 630 control steps at 20 Hz.

## Scoring

Every condition is a distance or a height, measured in the simulation every control step (in a real trial: from the
video and a ruler).

- **Shut in:** the spider's centre is inside the glass, the glass stands on its rim (tilted less than 25 degrees), and
  the rim is at most 8 mm above whatever is under it (desk or card) all round.
- **Carried out:** shut in on the card without a break from inside the room until its centre is past the wall's outer
  face. If the glass comes off it before that, it was not carried out: a spider riding out of the window on a bare card
  counts as knocked out, however it was lifted before.
- **0 for the whole trial** if the spider is ever touched (a gripper's tool point or fingertip within 1 cm of its
  shell while it is not shut in), squashed (the glass's rim comes down on it), knocked out of the window (past the wall
  without having been carried out) or escapes (gets behind x 5 cm, past 50 cm left or right, or off the desk), or if
  the glass falls off the desk or out of the window.
- **1.0** once the spider is out (carried out, then its centre 1 cm past the wall's outer face and not shut in) with
  the glass and the card back inside the room (short of the wall's inner face, on the desk).
- **0.75** if it is out but the glass or the card is still outside, or if it was shut in on the card with the card's
  underside 3 cm above the desk.
- **0.5** if it was shut in with the card under the whole rim (the rim at most 4 mm above the card all round).
- **0.25** if it was ever shut in under the glass.
- **0** otherwise.

The trial ends with success after 1 s at full score with the glass and the card at rest.

## Verification

    python -m pytest test_scoring.py

33 checks, no simulator needed (only `yam_arena`, which the walkthrough's setup installs). They cover:

- the geometry (shut in, sealed, squashed, the surfaces);
- the stand-in's program: it sits still with nothing near; backs away from a hand 3 cm in front; drives on from one
  behind; ignores a hand 6 cm away, above its sensors or at its side; runs the way it faces; stops when the hand has
  gone; backs away from the window wall and never drives into it; does not see the glass but sees a hand through it;
- its wheels: the drive force holds 8 cm/s against the model's friction, at 20 Hz control and 240 Hz physics;
- every scoring outcome, including the stand-in lifted shut in on the card, then the glass taken off inside the room
  and the stand-in riding out of the window on the moving card with nothing over it (0), a bare card carried out (0),
  and the glass put back over it inside the room before the carry (1.0);
- the demo: the layout, the glass push, no hand ever in the stand-in's sight or within 1 cm of it, and both arms
  reaching every waypoint with the scripted policy's IK (within 2 mm and 1 degree).

## Runtime and seeds

One layout, no randomness: run with `-E object_jitter=0`. Run limit 1,800 control steps (90 s at 20 Hz).

## Setup

Positions in cm from the midpoint between the two arm bases, on the table (x forward, y left).

- **Desk:** the station table, against the wall under a window.
- **Wall and window:** the wall's inner face runs across the desk at x 28.5, its outer face at x 31.5. The opening is
  90 cm wide (y -45 to 45), its frame's sill 2 cm above the desk. White frame, cream wall 50 cm high either side; the
  casement stands open outward on its hinge at the left. Outside: a grey stone outer sill (x 31.5 to 44, 1.2 cm thick)
  and a terracotta window box with plants (x 45 to 65, y -32 to 32, 9 cm tall).
- **Spider:** the stand-in below, at x 20, y 6, facing the window (+x), its program running.
- **Glass:** a straight clear 250 mL tumbler, 7.2 cm across, 9 cm tall, 0.22 kg, upside down at x 17, y 24.
- **Card:** a grey 1.5 mm greyboard card, flat face 25 cm (along y) by 15 cm (along x), with a blue picture printed
  over most of its top; its short edge on the right is folded up 2 cm and doubled (3 mm thick). Centred at x 19, y -14.
- Both arms start in the raised ready pose, grippers open.

## The spider stand-in

**What it is.** An Ozobot Evo (Titanium Black; about 100 to 130 USD; 3.35 cm across, 3.3 cm tall, 28 g). It has two
infrared proximity sensors at the front and two at the back, two driven wheels, and stores a program made in the free
OzoBlockly editor, which it runs without a computer.

**How it is dressed.** A 12 mm brown pom-pom (the body) and an 8 mm dark brown pom-pom (the head, toward the front)
glued on its top. Four 15 cm black pipe cleaners laid across its top, held with a strip of black tape and bent into
eight legs, four a side, their tips 1 cm out from the shell and 6 mm above the desk. No leg is in front of or behind
the shell, where the sensors look. With its legs it spans 5.4 cm, the size of a large house spider.

**Its program** (OzoBlockly, loaded onto the robot once):

    repeat forever:
      if a front proximity sensor detects something and neither back sensor does:
        drive backward at 80 mm/s
      else if a back proximity sensor detects something and neither front sensor does:
        drive forward at 80 mm/s
      else:
        stop

**What that makes it do.** Its sensors see about 5 cm ahead and behind, not to the sides and not higher than about
3.5 cm above the desk. Infrared passes through the clear glass, so it does not see the glass.
- A gripper coming at it low from the front: it backs away. From behind: it drives on. From the side or from above:
  nothing.
- It stops when the gripper is out of sight, and when it sees things both ahead and behind.
- Driving toward the window, it sees the wall under it and backs away: it never leaves through the window by itself.
- Under the glass it sees nothing and sits still.

**Getting onto the card.** No hidden help. The glass is held over it with its rim a few millimetres up. As the card
slides in under the rim, the glass is moved toward the card, and its far wall pushes the robot onto the card's edge.
Its round wheels ride up the 1.5 mm edge.

**Operating procedure for each trial.**
1. Charge the robot.
2. Start its program.
3. Hold it by its sides from above, set it on its mark facing the window, and lift the hand straight up. It does not
   see a hand above it.
4. Nobody touches it during the trial.
5. After the trial, pick it up the same way.

## Fidelity

- **The stand-in** is a rigid model of the robot's size and mass: a shell of 12 flat walls and a lid on four 9 mm
  round contacts (its wheels and skids), and a low centre of mass. Its legs and pom-poms are drawn only. Its contacts
  are slick (the lower friction of a pair counts, 0.25 sliding), as its wheels roll and its shell is smooth plastic.
- **Its program** runs every control step on what its sensors would see: points on the grippers (tool points and
  fingertips), along the wall under the window and along the card's fold, within 5 cm of either end of its shell, 40
  degrees either side of straight ahead or behind, up to 3.5 cm above the floor it stands on. The glass is not seen.
- **Its wheels** are a horizontal force (world frame) held for every physics step: it balances the model's sliding
  friction and brings it to 8 cm/s, at most 0.3 N (about its weight). When its program stops, the force goes to zero.
  Nothing else drives it; the glass and the card move it only by pushing it.
- **The glass** collides as a 24-sided wall and a base, but only its rim, its base's edge and six glints are drawn,
  so the stand-in shows through it. It does not break: a glass that falls off the desk scores 0.
- **The card** is rigid (it does not bend), and its fold is a second flat part.
- **Outside:** the station table continues under the outer sill and the window box.

## Real-world build

- A desk against a wall under an open window, or a desk with a 3 cm deep, 2 cm high wooden batten across it as the
  sill and a board behind it as the outer sill.
- A straight 250 mL glass tumbler (7.2 cm across, 9 cm tall, about 220 g).
- A 1.5 mm greyboard card cut 29 x 15 cm, one short edge folded up 2 cm and glued doubled.
- The spider stand-in: an Ozobot Evo, two pom-poms, four black pipe cleaners and black tape, dressed and programmed as
  above.
- Scoring from the video and a ruler. Nothing in the trial is hot, sharp or fast.

## Third-party assets

None. `bodykit.py` comes unchanged from tasks already in this repository. The Ozobot Evo is a commercial product, used
only in real trials; the simulation models it from its published size and mass.
