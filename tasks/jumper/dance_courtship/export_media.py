"""The performance video for jumper.dance_courtship.

Rendering, the camera and the music mux are `jumper.dance`'s
(`tasks/jumper/dance/export_media.py`). This module exists because that hook
reads its own `MEDIA`, and this dance's music has to come from this directory,
delayed by this clip's `audio_start_in_sim` (the one-second lead-in).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..common.dance.motion import AUDIO_SUFFIXES, find_material
from ..dance.export_media import _mux, _render, _rollout

__all__ = ["export_media"]


def export_media(out: Path, *, task_id: str, checkpoint: Path, asset, res) -> None:
    """Render the performance and ship it beside the bundle.

    Called by `scripts/export.py` through `tasks.load_export_media`.
    """
    from ..common.dance.motion import load_source
    from .env_cfg import MEDIA

    material = find_material(MEDIA)
    if material.audio is None:
        raise FileNotFoundError(
            f"no music in {material.motion.parent} (looked for "
            f"{', '.join(AUDIO_SUFFIXES)}). The policy and its contract are written; "
            f"the performance video is the dance with its music on it, so it needs "
            f"the track the choreography was made for. Put it there and export "
            f"again, or pass --no-video."
        )
    media = Path(out) / "media"
    media.mkdir(parents=True, exist_ok=True)

    print(f"\n[export] rendering the performance -> {media}")
    qpos, model, fps, frames_total, stopped_early = _rollout(
        task_id, asset, Path(checkpoint), res
    )

    if stopped_early is not None:
        print(
            f"[export] WARNING: the policy terminated at frame {stopped_early} of "
            f"{frames_total} ({stopped_early / fps:.1f} s of "
            f"{frames_total / fps:.1f} s). The video is the performance up to that "
            f"point, not the whole choreography."
        )

    raw = media / "dance.silent.mp4"
    _render(model, qpos, fps, raw)

    clip = load_source(material.motion)
    final = media / "dance.mp4"
    _mux(raw, material.audio, clip.audio_start_s, final)
    raw.unlink()

    shutil.copy2(material.audio, media / f"music{material.audio.suffix.lower()}")
    if material.eyes is not None:
        print(f"[export] face animation: {material.eyes.name}")
        shutil.copy2(material.eyes, media / f"eyes{material.eyes.suffix.lower()}")
    else:
        print(
            f"[export] no face animation found; put one in "
            f"{material.motion.parent}/ to ship it with the performance."
        )

    for f in sorted(media.iterdir()):
        print(f"    media/{f.name:<12}{f.stat().st_size / 1024 / 1024:8.1f} MB")
