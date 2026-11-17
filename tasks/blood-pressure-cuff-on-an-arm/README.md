# Put a blood-pressure cuff on a training arm: wrap it round, close the hook-and-loop, inflate it and start the monitor

A life-size training arm is held straight out on the table between the robot's two arms, palm up. An adult
blood-pressure cuff lies open under the upper arm, and an automatic monitor, joined to the cuff by its hose, lies to the
right. The robot has to wrap the cuff round the upper arm at the right height, close it, pump it up and start a reading,
then leave it alone while the monitor reads.

Instruction: **"Put the blood-pressure cuff on the arm, inflate it and start the monitor."**

## What makes it hard

- **A floppy band round a fixed limb.** The 52 cm cuff bends anywhere. One hand has to carry its bladder end up the
  inner side of the arm, over the top and down the outer side, while the other keeps the hook end clear, then bring the
  hook end round over it the other way.
- **Closing it.** The hook strip only sticks where it is laid flat on the cuff's loop face and pressed with at least
  3 N. A strip dragged across, or pressed edge-on, does not hold.
- **The right height.** The lower edge has to end up 1.5 to 4 cm above the elbow crease all the way round. The cuff
  starts 5.25 cm above it, so it has to be moved down the arm as it is wrapped.
- **Hands off.** The cuff only reads if nothing touches it for the 2 s after START, so both hands have to let go cleanly
  and stay clear.

## Where this comes up

Nursing assistants (O*NET 31-1131.00, about 1.4M US workers), medical assistants (31-9092.00, about 0.8M) and licensed
practical nurses (29-2061.00, about 0.6M) all list measuring and recording vital signs, blood pressure among them, as a
core task, done many times a shift. Every reading starts with fitting the cuff to the patient's bare upper arm.

## Files

| File | What it is |
|---|---|
| `task.py` | the task file: scene, the cuff's joints, the hook-and-loop and monitor rules, scoring and the scripted demo |
| `bodykit.py` | rigid bodies made of several shapes, hinges, sliders and glue joints, unchanged from tasks already in this repository |
| `test_scoring.py` | offline checks: layout, joints, every scoring outcome, the monitor's rules and the demo's reach |
| `task.json` | the task's details |

## Requirements

The program's Brev launchable (Isaac Lab-Arena with `yam_arena` and Inspect Robots, L40S). Before every shell:
`source ~/IsaacLab-Arena/yam.env`. The Dream key sits in `~/.dreamlabs/credentials`. Nothing else: no downloaded
assets.

Tested on that launchable on 17 Nov 2026. `task.py` SHA-256, first 12 hex: 38d6ab3afb1e (`sha256sum task.py |
cut -c1-12`).

## Run and score one Dream-1 trial

From this folder:

    inspect-robots run --embodiment yam_arena --policy dream \
      -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
      --instruction "Put the blood-pressure cuff on the arm, inflate it and start the monitor." \
      --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/bp-dream

Dream-1 refuses this instruction before the arms move, so this run records no motion and scores nothing:
`PolicyError: DreamAPI refused the inference request: DA-4000 BAD_REQUEST (malformed): unsafe instruction, refused: Attaching and inflating a cuff on a person's arm causes squeezing/force.`

Scoring is automatic: the CLI prints `task_score: <value>`, and the log keeps it under `results.metrics.task_score`.
Set `BP_DEBUG=1` to print the trial's state every half second on stderr.

## Scripted demo

    inspect-robots run --embodiment yam_arena --policy yam_scripted \
      -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
      --instruction "Put the blood-pressure cuff on the arm, inflate it and start the monitor." \
      --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/bp-demo

The demo video is the top camera of this run.

`demo()` gives both arms Cartesian waypoints (pointing straight down, fingers closing across the strip), played open
loop. Every target comes from the layout constants in `task.py`; nothing is read from the simulator.

1. Each hand pinches one curled end of the cuff, and together they lift it and move it 2.75 cm down the arm.
2. The right hand carries the bladder end up the inner side of the arm, over the top and down the outer side, and lets
   go: the end's curl closes it round the arm.
3. The left hand carries the hook end up the outer side, over the top and down the inner side, which lays the hook
   strip on the loop face, lets go, lifts clear of the end with its fingers open, and presses the hook strip down on
   the top of the arm.
4. The right hand presses INFLATE. The bladder fills and draws the cuff tight round the arm.
5. The right hand presses START and both hands stay clear while the monitor reads.

About 41 s of motion; the run ends with success after about 37.5 s (750 control steps at 20 Hz).

## Scoring

Every condition is a distance, a height or a button press, measured in the simulation every control step (in a real
trial: from the video, a ruler and the monitor's display).

- **Closed:** at least 8 of the hook strip's 12 cm is stuck to the cuff's loop face, and the cuff goes at least once
  round the upper arm, between its shoulder end and the elbow crease, with no part of it more than 7 cm off the skin.
- **0** if, at the end, the cuff is not round the upper arm like this, or the monitor is knocked over (tilted over 30
  degrees) or off the table.
- Otherwise, at the end:
  - **+0.40** closed;
  - **+0.20** the cuff's lower edge is 1.5 to 4 cm above the elbow crease all the way round;
  - **+0.20** inflated: INFLATE was pressed while the cuff was closed, and the inflated cuff now lies on the arm all
    round its closed loop (no part of it, up to the end of the stuck hook strip, more than 1.5 cm off the skin);
  - **+0.20** reading: START was pressed after that, with the cuff still closed, and no gripper touched the cuff in the
    2 s that followed.
- INFLATE pressed before the cuff is closed does nothing (a real monitor shows a cuff error), nor does START pressed
  before INFLATE. A button counts as pressed when it is pushed in 1.5 mm.
- The hook strip sticks only where it lies flat on the loop face, face to face within 40 degrees, pressed with at
  least 3 N.

The trial ends with success once the score has held at 1.0 for 2 s.

## How the simulation stands in for the cuff

- The cuff is 13 rigid strips of 4 cm, joined edge to edge by sprung hinges (the nylon's give). The last two joints at
  each end are sprung to a curl, like a cuff kept rolled up.
- Hook-and-loop is a rule: when a hook strip is pressed flat on a strip's loop face as above, a fixed joint holds the
  two where they lie, for the rest of the trial.
- The bladder is not simulated as air. With the cuff closed, INFLATE makes the 7 joints inside the bladder close up
  along the band over 2 s (up to 2 cm each, at most 5 N of band tension), so the cuff draws onto the arm and stops
  there.
- The monitor's reading is a rule: START after INFLATE with the cuff closed, then 2 s with no gripper on the cuff.

## Verification

    python -m pytest test_scoring.py

18 checks, no simulator needed (only `yam_arena`). They cover the layout (nothing overlaps, the cuff fits the arm and
starts too far up it, the curled ends stand clear), the joints (hinges, hose links, the hook-and-loop pairs, the
bladder joints and how much slack they can take up), the scoring (every part, the free end past the hook strip, the
monitor knocked off), the monitor's rules (INFLATE only with the cuff closed, START only after INFLATE, a touch while it
reads), and that both arms reach every demo waypoint with the scripted policy's IK (within 2 mm and 1 degree).

## Runtime and seeds

One layout, no randomness: run with `-E object_jitter=0`. Run limit 1,800 control steps (90 s at 20 Hz).
