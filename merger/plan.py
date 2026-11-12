from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, floor

from .analysis import RATE


@dataclass(frozen=True)
class Removed:
    clip: int
    start: float
    end: float
    reference_clip: int
    reference_start: float
    reference_end: float
    confidence: float


@dataclass(frozen=True)
class Segment:
    clip: int
    start: float
    frames: int
    transition_frames: int = 0


def available(start, end, exclusions):
    cursor = start
    for lo, hi in sorted(exclusions):
        if hi <= cursor:
            continue
        if lo >= end:
            break
        if lo > cursor:
            yield cursor, lo
        cursor = max(cursor, hi)
    if cursor < end:
        yield cursor, end


def build_plan(analyses, candidates, settings, fps):
    exclusions = [[] for _ in analyses]
    removed = []
    for match in sorted(candidates, key=lambda m: (m.later, m.later_start, -m.frames, -m.confidence, m.earlier)):
        target = match.later_start / RATE
        source = match.earlier_start / RATE
        delta = target - source
        length = match.frames / RATE
        for lo, hi in available(target, target + length, exclusions[match.later]):
            for ref_lo, ref_hi in available(lo - delta, hi - delta, exclusions[match.earlier]):
                # Round toward the interior so encoding cannot remove unverified time.
                start = ceil((ref_lo + delta) * fps - 1e-7) / fps
                end = floor((ref_hi + delta) * fps + 1e-7) / fps
                if end - start + 1 / fps < settings.minimum_seconds:
                    continue
                exclusions[match.later].append((start, end))
                removed.append(Removed(match.later, start, end, match.earlier,
                                       start - delta, end - delta, match.confidence))
    segments = []
    for index, item in enumerate(analyses):
        duration = max(item.clip.duration, item.clip.audio_end)
        for start, end in available(0, duration, exclusions[index]):
            frames = round((end - start) * fps) if end < duration else ceil((end - start) * fps - 1e-7)
            if frames > 0:
                segments.append(Segment(index, start, frames))
    return segments, removed


def to_json(analyses, segments, removed, settings, options):
    return {
        "schema": 1,
        "time_unit": "seconds relative to first selected audio or video timestamp",
        "sources": [{**asdict(item.clip), "path": str(item.clip.path)} for item in analyses],
        "settings": asdict(settings),
        "output": asdict(options),
        "keep": [asdict(segment) for segment in segments],
        "remove": [asdict(interval) for interval in sorted(removed, key=lambda r: (r.clip, r.start))],
    }

