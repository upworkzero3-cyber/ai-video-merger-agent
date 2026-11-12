from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from threading import Event


EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mts", ".m2ts", ".wmv", ".mpeg", ".mpg", ".ts"}


class MediaError(RuntimeError):
    pass


class Cancelled(MediaError):
    pass


@dataclass(frozen=True)
class Clip:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    video: int
    audio: tuple[int, ...]
    origin: float
    audio_end: float
    audio_channels: tuple[int, ...] = ()


def natural_key(name: str):
    return tuple((1, int(part)) if part.isdecimal() else (0, part)
                 for part in re.split(r"(\d+)", name.casefold()))


def ordered_paths(folder: Path) -> list[Path]:
    paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS]
    if paths and all(re.search(r"\d", p.stem) for p in paths):
        return sorted(paths, key=lambda p: (natural_key(p.stem), p.stat().st_mtime_ns, p.name.casefold()))
    return sorted(paths, key=lambda p: (p.stat().st_mtime_ns, natural_key(p.stem), p.name.casefold()))


def binary(name: str) -> str:
    root = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
    if not getattr(sys, "frozen", False):
        root = root.parent
    configured = os.environ.get("FFMPEG_BIN")
    directories = ([Path(configured)] if configured else []) + [root / "bin", root]
    for directory in directories:
        candidate = directory / (name + ".exe")
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise MediaError(f"{name}.exe was not found. Install FFmpeg and put its bin directory on PATH or set FFMPEG_BIN.")


def check_cancel(cancel: Event):
    if cancel.is_set():
        raise Cancelled("Processing cancelled.")


def run(name, args, cancel, *, progress=None, stdout=None):
    check_cancel(cancel)
    target = subprocess.PIPE if stdout is None else stdout
    with tempfile.TemporaryFile() as errors:
        with subprocess.Popen(
            [binary(name), *map(str, args)], stdin=subprocess.DEVNULL,
            stdout=target, stderr=errors,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ) as process:
            try:
                while True:
                    check_cancel(cancel)
                    try:
                        data, _ = process.communicate(timeout=0.15)
                        if progress:
                            progress()
                        break
                    except subprocess.TimeoutExpired:
                        if progress:
                            progress()
            except BaseException:
                if process.poll() is None:
                    process.kill()
                process.communicate()
                raise
            if process.returncode:
                errors.seek(0, os.SEEK_END)
                errors.seek(max(0, errors.tell() - 3500))
                detail = errors.read().decode("utf-8", "replace").strip()
                raise MediaError(f"{name} exited with code {process.returncode}. {detail}")
    check_cancel(cancel)
    return data


def number(value, default=0.0):
    try:
        parsed = float(value)
    except (ValueError, TypeError):
        return default
    return parsed if math.isfinite(parsed) else default


def probe(path: Path, cancel: Event) -> Clip:
    raw = run("ffprobe", ["-v", "error", "-show_format", "-show_streams", "-of", "json", path], cancel)
    info = json.loads(raw)
    streams = info.get("streams", [])
    video = next((s for s in streams if s["codec_type"] == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        raise MediaError("No playable video stream.")
    audio = [s for s in streams if s["codec_type"] == "audio"]
    video_start = number(video.get("start_time"), number(info.get("format", {}).get("start_time")))
    origin = min([video_start, *(number(s.get("start_time"), video_start) for s in audio)])
    container_end = number(info.get("format", {}).get("duration")) + number(
        info.get("format", {}).get("start_time"))
    duration = video_start + number(video.get("duration"), container_end - video_start) - origin
    width, height = int(video.get("width", 0)), int(video.get("height", 0))
    rotation = next((number(s.get("rotation")) for s in video.get("side_data_list", [])
                     if "rotation" in s), number(video.get("tags", {}).get("rotate")))
    if round(rotation) % 180 == 90:
        width, height = height, width
    try:
        fps = float(Fraction(video.get("avg_frame_rate", "0")))
    except (ValueError, ZeroDivisionError):
        fps = 0
    if duration <= 0 or min(width, height) <= 0:
        raise MediaError("Video duration or dimensions could not be determined.")
    ends = [number(s.get("start_time"), origin) + number(s.get("duration"),
             container_end - number(s.get("start_time"), origin))
            for s in audio]
    return Clip(path, duration, width, height, fps, video["index"],
                tuple(s["index"] for s in audio), origin, max(ends, default=origin) - origin,
                tuple(max(1, int(s.get("channels", 1))) for s in audio))


def scan(folder, cancel, notify):
    binary("ffmpeg")
    binary("ffprobe")
    clips, warnings = [], []
    for path in ordered_paths(folder):
        try:
            clips.append(probe(path, cancel))
        except Cancelled:
            raise
        except (MediaError, ValueError, KeyError) as exc:
            warning = f"Skipped {path.name}: {exc}"
            warnings.append(warning)
            notify(warning)
    return clips, warnings

