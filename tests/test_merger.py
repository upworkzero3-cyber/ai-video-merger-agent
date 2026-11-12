from dataclasses import replace
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pytest

from merger.analysis import RATE, extract, release
from merger.detect import Match, Settings, confirmed
from merger.export import ExportOptions
from merger.media import MediaError, binary, ordered_paths, probe, run, scan
from merger.pipeline import process
from merger.plan import build_plan


@pytest.fixture
def binaries():
    for name in ("ffmpeg", "ffprobe"):
        try:
            binary(name)
        except MediaError as exc:
            pytest.skip(str(exc))


def footage(seed, seconds=2):
    rng = np.random.default_rng(seed)
    frames = []
    base = rng.integers(10, 175, (6, 10, 3), dtype=np.uint8)
    for index in range(seconds * 24):
        frame = cv2.resize(base, (160, 96), interpolation=cv2.INTER_NEAREST)
        x, y = rng.integers(4, 70, size=2)
        cv2.rectangle(frame, (x, y), (x + 62, y + 20), (240, 240, 240), -1)
        frames.append(frame)
    samples = rng.normal(0, .07, size=seconds * 48000).astype("<f4")
    samples += .12 * np.sin(np.arange(len(samples)) * (seed + 320) * 2 * np.pi / 48000)
    return np.stack(frames), samples


def make_clip(directory, name, chunks, *, bright=False, mute=False):
    frames = np.concatenate([chunk[0] for chunk in chunks])
    if bright:
        frames = np.clip(frames.astype(np.int16) + 5, 0, 255).astype(np.uint8)
    raw = directory / (name + ".bgr")
    raw.write_bytes(frames.tobytes())
    audio = directory / (name + ".f32")
    audio.write_bytes(np.concatenate([chunk[1] for chunk in chunks]).tobytes())
    output = directory / (name + ".mp4")
    args = ["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "160x96", "-r", "24",
            "-i", raw]
    if not mute:
        args += ["-f", "f32le", "-ar", "48000", "-ac", "1", "-i", audio, "-c:a", "aac", "-b:a", "160k"]
    run("ffmpeg", [*args, "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", output], Event())
    raw.unlink()
    audio.unlink()
    return output


def analyze(paths, tmp_path):
    results = []
    for index, path in enumerate(paths):
        directory = tmp_path / f"analysis-{index}"
        directory.mkdir()
        results.append(extract(probe(path, Event()), directory, Event(), print))
    try:
        return confirmed(results, Settings(), Event(), lambda *_: None), results
    except BaseException:
        release(results)
        raise


def test_ordering(tmp_path):
    for name in ("Clip10.mp4", "Clip2.mov", "Clip1.mkv"):
        (tmp_path / name).touch()
    assert [p.stem for p in ordered_paths(tmp_path)] == ["Clip1", "Clip2", "Clip10"]


def test_ordering_mtime(tmp_path):
    import os
    for name, seconds in [("Zoo.mp4", 100), ("Apple.mov", 200)]:
        path = tmp_path / name
        path.touch()
        os.utime(path, (seconds, seconds))
    assert [p.stem for p in ordered_paths(tmp_path)] == ["Zoo", "Apple"]


@pytest.mark.parametrize("bright", [False, True], ids=["exact", "near"])
def test_overlap_and_unique_scenes(tmp_path, binaries, bright):
    a, b, c = [footage(seed) for seed in (1, 2, 3)]
    paths = [make_clip(tmp_path, "clip1", [a, b]), make_clip(tmp_path, "clip2", [b, c], bright=bright)]
    matches, results = analyze(paths, tmp_path)
    try:
        kept, removed = build_plan(results, matches, Settings(), 30)
        assert len(removed) == 1
        interval = removed[0]
        assert (interval.reference_clip, interval.clip) == (0, 1)
        assert interval.reference_start == pytest.approx(2, abs=1/RATE)
        assert interval.start == pytest.approx(0, abs=1/RATE)
        assert interval.end == pytest.approx(2, abs=2/RATE)
        assert interval.end <= 2
        assert sum(segment.frames for segment in kept if segment.clip == 0) == 120
    finally:
        release(results)


def test_internal_duplicate(tmp_path, binaries):
    a, b, c = [footage(seed) for seed in (1, 2, 3)]
    matches, results = analyze([make_clip(tmp_path, "clip1", [a, b, a, c])], tmp_path)
    try:
        kept, removed = build_plan(results, matches, Settings(), 30)
        assert len(removed) == 1
        assert removed[0].start == pytest.approx(4, abs=1/RATE)
        assert removed[0].end == pytest.approx(6, abs=2/RATE)
        assert removed[0].reference_start == 0
        assert len(kept) == 2
    finally:
        release(results)


def test_identical_picture_with_unique_audio_is_preserved(tmp_path, binaries):
    a, b, c = [footage(seed) for seed in (1, 2, 3)]
    paths = [make_clip(tmp_path, "clip1", [a, b]), make_clip(tmp_path, "clip2", [(b[0], c[1]), c])]
    matches, results = analyze(paths, tmp_path)
    try:
        assert not matches
    finally:
        release(results)


def test_no_cut_can_use_previously_removed_evidence(tmp_path):
    from types import SimpleNamespace
    clip = SimpleNamespace(duration=5, audio_end=5)
    inputs = [SimpleNamespace(clip=clip)] * 3
    proposals = [Match(0, 1, 16, 0, 16, 1), Match(1, 2, 0, 0, 16, 1)]
    _, removed = build_plan(inputs, proposals, Settings(), 30)
    assert [row.clip for row in removed] == [1]


def test_mp4_and_audio_sync(tmp_path, binaries):
    import json

    a, b, c = [footage(seed) for seed in (1, 2, 3)]
    paths = [make_clip(tmp_path, "clip1", [a, b]), make_clip(tmp_path, "clip2", [b, c])]
    (tmp_path / "clip3.mp4").write_bytes(b"invalid")
    clips, warnings = scan(tmp_path, Event(), print)
    assert len(clips) == 2 and len(warnings) == 1
    initial = [path.read_bytes() for path in paths]
    result = process(clips, warnings, tmp_path / "out", Settings(), ExportOptions(320, 180), None,
                     Event(), lambda *_: None, print, lambda *_: None)
    report = json.loads((result / "processing-report.json").read_text())
    assert report["removed_count"] == 1
    assert [path.read_bytes() for path in paths] == initial
    merged = result / "merged.mp4"
    media = probe(merged, Event())
    assert (media.width, media.height, media.fps) == (320, 180, 30)
    assert abs(media.audio_end - media.duration) < .04
    metadata = run("ffprobe", ["-v", "error", "-show_entries", "stream=codec_name", "-of", "json", merged], Event())
    assert [stream["codec_name"] for stream in json.loads(metadata)["streams"]] == ["h264", "aac"]

    def decoded(path):
        return np.frombuffer(run("ffmpeg", ["-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "48000",
                                            "-f", "f32le", "-c:a", "pcm_f32le", "-"], Event()), dtype="<f4")
    exported = decoded(merged)
    source = decoded(paths[1])
    # Check a known audio event after the trim, at the corresponding output video timestamp.
    cut = round(report["remove"][0]["end"] * 48000)
    begin = 4 * 48000 + 12000
    assert np.corrcoef(exported[begin:begin+6000], source[cut+12000:cut+18000])[0, 1] > .96


def test_quiet_optional_transition(tmp_path, binaries):
    paths = [make_clip(tmp_path, f"clip{i}", [footage(i)], mute=True) for i in (1, 2)]
    options = ExportOptions(320, 180, transitions=True)
    result = process([probe(path, Event()) for path in paths], [], tmp_path / "out", Settings(), options, None,
                     Event(), lambda *_: None, print, lambda *_: None)
    assert probe(result / "merged.mp4", Event()).duration == pytest.approx(4 - 4/30, abs=.04)

