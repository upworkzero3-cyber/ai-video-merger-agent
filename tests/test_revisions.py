import json
from threading import Event

from merger.detect import Settings
from merger.export import ExportOptions
from merger.media import probe
from merger.pipeline import process
from test_merger import binaries, footage, make_clip


def test_opt_in_cumulative_revisions(tmp_path, binaries):
    parts = [footage(seed) for seed in (1, 2, 3, 4, 5)]
    paths = [
        make_clip(tmp_path, "clip01", parts[:3], mute=True),
        make_clip(tmp_path, "clip02", parts[:4], mute=True),
        make_clip(tmp_path, "clip03", parts[:3] + parts[4:5] * 2, mute=True),
    ]
    options = ExportOptions(320, 180, prefer_latest_revision=True)
    result = process([probe(path, Event()) for path in paths], [], tmp_path / "out", Settings(), options,
                     None, Event(), lambda *_: None, print, lambda *_: None)
    report = json.loads((result / "processing-report.json").read_text())
    assert not ExportOptions().prefer_latest_revision
    assert [entry["clip"] for entry in report["superseded_revisions"]] == [0, 1]
    assert [entry["clip"] for entry in report["keep"]] == [2]
    assert probe(result / "merged.mp4", Event()).duration == 10

