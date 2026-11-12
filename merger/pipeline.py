from __future__ import annotations

import json
import tempfile
import uuid
from datetime import datetime
from math import ceil
from pathlib import Path

from .analysis import Transcriber, extract, release
from .detect import confirmed, cumulative_revisions
from .export import add_transitions, encode
from .media import Cancelled, MediaError, check_cancel
from .plan import Segment, build_plan, to_json


def process(clips, warnings, destination, settings, options, model_path, cancel, update, log, stats):
    if not clips:
        raise MediaError("Select a folder containing at least one playable video.")
    if not (.5 <= settings.minimum_seconds <= 30 and 0 < settings.hash_similarity <= 1
            and 0 < settings.tile_similarity <= 1 and 0 < settings.audio_correlation <= 1):
        raise MediaError("Duplicate detection settings are outside the supported range.")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.now():%Y-%m-%d_%H%M%S}-{uuid.uuid4().hex[:8]}"
    final = destination / run_id
    update("Preparing analysis", 0)
    transcriber = Transcriber(model_path) if model_path else None
    analyses = []
    warnings = list(warnings)

    with tempfile.TemporaryDirectory(prefix="video-merger-") as work, tempfile.TemporaryDirectory(
        prefix=".in-progress-", dir=destination
    ) as staging:
        work, staging = Path(work), Path(staging)
        bundle = staging / "result"
        bundle.mkdir()
        try:
            for index, clip in enumerate(clips):
                check_cancel(cancel)
                update(f"Analyzing {index + 1}/{len(clips)} · {clip.path.name}", int(43 * index / len(clips)))
                folder = work / str(index)
                folder.mkdir()
                try:
                    analyses.append(extract(clip, folder, cancel, log, transcriber))
                except Cancelled:
                    raise
                except MediaError as exc:
                    warning = f"Skipped {clip.path.name} during decoding: {exc}"
                    warnings.append(warning)
                    log(warning)
            if not analyses:
                raise MediaError("None of the selected videos could be decoded.")
            update("Checking cumulative revisions" if options.prefer_latest_revision else "Matching clips", 44)
            revisions = cumulative_revisions(analyses, settings, cancel) if options.prefer_latest_revision else []
            if revisions:
                latest = analyses[-1].clip
                frames = ceil(max(latest.duration, latest.audio_end) * options.fps - 1e-7)
                segments, removed = [Segment(len(analyses) - 1, 0, frames)], []
                log(f"Selected {latest.path.name} as the newest cumulative revision. "
                    f"Superseded {len(revisions)} earlier exports.")
            else:
                if options.prefer_latest_revision:
                    log("The clips did not meet the cumulative revision checks. Using duplicate detection.")
                matches = confirmed(analyses, settings, cancel, lambda i, n, name: update(
                    f"Matching {i + 1}/{n} · {name}", 44 + int(25 * i / n)))
                update("Planning cuts", 70)
                segments, removed = build_plan(analyses, matches, settings, options.fps)
            segments = add_transitions(analyses, segments, options)
            seconds = sum(interval.end - interval.start for interval in removed)
            revision_seconds = sum(row["superseded_seconds"] for row in revisions)
            stats(len(removed) + len(revisions), seconds + revision_seconds)
            log(f"Planned {len(removed)} removals totaling {seconds:.2f} seconds.")
            plan = to_json(analyses, segments, removed, settings, options)
            plan["superseded_revisions"] = revisions
            (bundle / "editing-plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
            update("Exporting MP4 · 0%", 72)
            duration = encode(analyses, segments, options, work, bundle / "merged.mp4", cancel,
                              lambda value: update(f"Exporting MP4 · {value:.0%}", 72 + int(27 * value)))
            report = {**plan, "completed": datetime.now().astimezone().isoformat(),
                      "warnings": warnings, "duration": duration, "removed_seconds": seconds,
                      "removed_count": len(removed), "superseded_seconds": revision_seconds,
                      "output_file": "merged.mp4"}
            (bundle / "processing-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            check_cancel(cancel)
            bundle.rename(final)
        finally:
            release(analyses)
    update("Export complete", 100)
    return final

