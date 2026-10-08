# jumper.dance_courtship

A short comic crab courtship dance: the two front claws gesture, the body shimmies,
bows nose-up and leans, and the four other feet stay on the floor. It is about ten
seconds of clip (one second of stillness, the music, then a return to standing),
meant to be learned the same way as `jumper.dance`.

The poses live in `tools/courtship_keyframes.py`. The numbers at the top of that
file are the ones to edit: `TILT_RAD` (starts at 10 degrees), `SHIMMY_ROLL_RAD`,
`SHIMMY_Y_M`, `BOW_PITCH_RAD`, `BOW_DROP_M`, `CLAW_OPEN`. Roll this large has not
been tried on the robot. Raise `TILT_RAD` in the preview until a foot slips, then
stop.

## Regenerate the clip

From the repository root, with the editable install active:

```bash
python tools/make_courtship_dance.py
```

That rewrites `media/courtship.npz` and prints the joint speeds, how close each
joint comes to a stop, and how far the support feet drift. It refuses to write
the file if a joint exceeds the servo's corner speed, leaves its mechanical range,
or a support foot leaves its stance point by more than 2 mm. Running it twice
writes the same file.

## Preview

No trained policy is required. The zero agent holds the reference:

```bash
python scripts/play.py --task jumper.dance_courtship --agent zero
```

On macOS the live window needs `.venv/bin/mjpython` instead of `python`. Add
`--headless` to run without a window. Watch for the things listed at the end of
this file; the generator's numbers cannot see whether a gag reads.

## Train on a Windows GPU PC

The machine this was sized for is an RTX 2080 Super with 8 GB. A 1024-environment
run of `jumper.dance` used about 4.3 GB on that card, and this task uses the same
tracking environment, so start at 1024. This repository has not measured this
clip's own footprint. If the run runs out of memory, use 512.

```powershell
.venv\Scripts\Activate.ps1
python scripts/train.py --task jumper.dance_courtship --num_envs 1024
```

Leave the backend on its default so it uses the GPU. Logs land in
`logs/<model>/jumper.dance_courtship/<date-time>/`.

## Export the video with music

```bash
python scripts/export.py --task jumper.dance_courtship --checkpoint logs/<model>/jumper.dance_courtship/<run>/model_<N>.pt
```

The video is written under the export's `media/` directory, with
`voxel_dance_cut.wav` starting one second in, after the lead-in. `--no-video`
skips the render and still writes the policy. Credit Kevin MacLeod in the video
description; the wording is in `media/README.md`.

## What to look at in the preview

- The claws rise apart and open, then meet overhead. They should not pass through
  each other on the way up.
- Beat 3 is a shut clack, beat 4 an open one.
- Beats 5-6 sway left, then right, with the claws still high.
- Beat 7 freezes and the left claw drops halfway. Beat 8 is the right claw
  beckoning twice.
- Beats 9-10 are windshield-wiper claws, crossing at different heights.
- Beats 11-12 bow with the nose up and the rear low, twice. This is the opposite
  pitch from `jumper.dance`.
- Beats 13-14 lean right, then left, about 10 degrees. If a foot skates, that
  lean is too big.
- Beat 15 makes a heart overhead (the fingertip origins are about 11 mm apart, so
  the pads should touch). Beat 16 holds it, then the left claw flops. On the way
  down that claw passes about 7 mm from the right tip and may look like a brush.
- The overhead claws peak near 0.26 m. If they leave the top of the frame, the
  camera in the export is `jumper.dance`'s and was framed for a slightly lower arm.
