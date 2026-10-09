# Material for `jumper.dance_courtship`

One choreography and the track it was timed to. Training reads the `.npz` only.
`scripts/export.py` reads the wav as well and starts it one second into the clip,
which is `audio_start_in_sim`.

| file | what it is |
|---|---|
| `courtship.npz` | the dance, written by `tools/make_courtship_dance.py` |
| `voxel_dance_cut.wav` | the music, a cut of "Voxel Revolution" |

Exactly one `.npz` and one audio file. A second of either kind is an error: the
loader picks by extension and refuses to guess.

## Music credit

"Voxel Revolution" Kevin MacLeod (incompetech.com)
Licensed under Creative Commons: By Attribution 4.0 License
http://creativecommons.org/licenses/by/4.0/

The file is the cut shipped with this task (`voxel_dance_cut.wav`), 7.8688 s,
122 bpm, sixteen beats. Put the same credit in the video description.
