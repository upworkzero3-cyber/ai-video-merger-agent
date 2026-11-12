from threading import Event

import numpy as np
import pytest
from PIL import Image

from merger.detect import Settings
from merger.export import ExportOptions
from merger.media import probe, run
from merger.pipeline import process
from test_merger import binaries, footage, make_clip


def test_watermark_persists_across_joins(tmp_path, binaries):
    logo = tmp_path / "channel logo.png"
    image = Image.new("RGBA", (48, 24), (0, 0, 0, 0))
    image.paste((0, 255, 0, 255), (4, 4, 44, 20))
    image.save(logo)
    original = logo.read_bytes()
    paths = [make_clip(tmp_path, f"clip{i}", [footage(i)], mute=True) for i in (1, 2)]
    options = ExportOptions(320, 180, transitions=True, watermark_path=str(logo),
                            watermark_corner="top-left", watermark_width=25, watermark_opacity=100)
    result = process([probe(path, Event()) for path in paths], [], tmp_path / "out", Settings(), options, None,
                     Event(), lambda *_: None, print, lambda *_: None)
    assert logo.read_bytes() == original
    merged = result / "merged.mp4"
    for moment in (0.25, 1.93, 3.5):
        data = run("ffmpeg", ["-v", "error", "-ss", str(moment), "-i", merged, "-map", "0:v:0",
                              "-frames:v", "1", "-pix_fmt", "rgb24", "-threads:v", "1",
                              "-f", "rawvideo", "-"], Event())
        frame = np.frombuffer(data, np.uint8).reshape(180, 320, 3)
        rgb = np.median(frame[18:30, 20:55], axis=(0, 1))
        assert rgb[1] > 230 and rgb[0] < 25 and rgb[2] < 25


def test_watermark_default_is_disabled():
    assert not ExportOptions().watermark_path

