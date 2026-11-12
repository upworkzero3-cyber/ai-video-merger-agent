from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .analysis import AUDIO_RATE
from .media import MediaError, probe, run


@dataclass(frozen=True)
class ExportOptions:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    encoder: str = "libx264"
    transitions: bool = False
    watermark_path: str = ""
    watermark_corner: str = "bottom-right"
    watermark_width: int = 12
    watermark_opacity: int = 85
    prefer_latest_revision: bool = False


LAYOUTS = {1: "mono", 2: "stereo", 3: "2.1", 4: "quad", 5: "5.0", 6: "5.1", 7: "6.1", 8: "7.1"}


def quiet(item, start, duration):
    return all(not samples.size or float(np.max(np.abs(samples))) < .001
               for sound in item.sounds
               for samples in [sound.samples[max(0, round(start * AUDIO_RATE)):
                                             max(0, round((start + duration) * AUDIO_RATE))]])


def add_transitions(analyses, segments, options):
    if not options.transitions:
        return segments
    frames = max(2, round(.13 * options.fps))
    seconds = frames / options.fps
    result = segments[:1]
    for prior, current in zip(segments, segments[1:]):
        first, second = analyses[prior.clip], analyses[current.clip]
        end = prior.start + prior.frames / options.fps
        eligible = (prior.clip != current.clip and current.start == 0
                    and abs(end - first.clip.duration) <= 1 / options.fps
                    and min(prior.frames, current.frames) >= options.fps
                    and quiet(first, end - seconds, seconds) and quiet(second, 0, seconds))
        result.append(replace(current, transition_frames=frames) if eligible else current)
    return result


def graph(analyses, segments, options):
    fps = options.fps
    clip_count = max((len(item.clip.audio) for item in analyses), default=0)
    channel_counts = [max(item.clip.audio_channels[i] for item in analyses
                          if i < len(item.clip.audio_channels)) for i in range(clip_count)] or [2]
    try:
        layouts = [LAYOUTS[count] for count in channel_counts]
    except KeyError as exc:
        raise MediaError("The AAC export supports source tracks with up to eight channels.") from exc

    lines, groups = [], []
    transition = any(segment.transition_frames for segment in segments)
    pixel_format = "yuv444p" if transition else "yuv420p"
    for index, segment in enumerate(segments):
        clip = analyses[segment.clip].clip
        start = segment.start
        duration = segment.frames / fps
        end = start + duration
        v = f"v{index}"
        audio = [f"a{index}_{track}" for track in range(len(layouts))]
        # Normalize to a common timestamp grid before taking exact frame ranges.
        lines.append(
            f"[{segment.clip}:{clip.video}]setpts=PTS-({clip.origin:.9f})/TB,"
            f"fps={fps}:start_time=0:round=up,"
            f"tpad=stop_mode=clone:stop_duration={max(0., end - clip.duration) + 2/fps:.9f},"
            f"trim=start_frame={round(start * fps)}:end_frame={round(start * fps) + segment.frames},"
            f"scale=w='max(2,trunc(min({options.width},{options.height}*dar)/2)*2)':"
            f"h='max(2,trunc(min({options.height},{options.width}/dar)/2)*2)',setsar=1,"
            f"pad={options.width}:{options.height}:(ow-iw)/2:(oh-ih)/2,"
            f"format={pixel_format},settb=1/{fps},setpts=N[{v}]"
        )
        sample_count = round(duration * 48000)
        for track, (layout, label) in enumerate(zip(layouts, audio)):
            common = f"aformat=sample_rates=48000:channel_layouts={layout}"
            if track < len(clip.audio):
                lines.append(
                    f"[{segment.clip}:{clip.audio[track]}]asetpts=PTS-({clip.origin:.9f})/TB,"
                    f"aresample=48000:async=1:first_pts=0,{common},"
                    f"apad=whole_len={round(end * 48000)},"
                    f"atrim=start_sample={round(start * 48000)}:end_sample={round(start * 48000) + sample_count},"
                    f"asetpts=N/SR/TB[{label}]"
                )
            else:
                lines.append(f"anullsrc=r=48000:cl={layout},atrim=end_sample={sample_count}[{label}]")
        groups.append((v, audio))

    if transition:
        video, audio = groups[0]
        length = segments[0].frames
        for index, (segment, (next_v, next_a)) in enumerate(zip(segments[1:], groups[1:]), 1):
            out_v = f"mv{index}"
            out_a = [f"ma{index}_{track}" for track in range(len(layouts))]
            overlap = segment.transition_frames
            if overlap:
                lines.append(f"[{video}][{next_v}]xfade=duration={overlap/fps:.9f}:"
                             f"offset={(length-overlap)/fps:.9f}[{out_v}]")
                for a, b, c in zip(audio, next_a, out_a):
                    lines.append(f"[{a}][{b}]acrossfade=ns={round(overlap/fps*48000)}[{c}]")
            else:
                lines.append(f"[{video}][{next_v}]concat=n=2:v=1:a=0,settb=1/{fps},setpts=N[{out_v}]")
                for a, b, c in zip(audio, next_a, out_a):
                    lines.append(f"[{a}][{b}]concat=n=2:v=0:a=1,asetpts=N/SR/TB[{c}]")
            video, audio = out_v, out_a
            length += segment.frames - overlap
        lines.append(f"[{video}]format=yuv420p[vout]")
        return ";\n".join(lines), ["vout", *audio]

    labels = ["vout", *(f"aout{track}" for track in range(len(layouts)))]
    lines.append("".join(f"[{label}]" for video, audio in groups for label in [video, *audio])
                 + f"concat=n={len(groups)}:v=1:a={len(layouts)}"
                 + "".join(f"[{label}]" for label in labels))
    return ";\n".join(lines), labels


def encode(analyses, segments, options, temporary: Path, destination: Path, cancel, notify):
    if not segments:
        raise MediaError("There are no segments to export.")
    if options.fps not in (24, 25, 30, 50, 60) or min(options.width, options.height) <= 0 or (
        options.width % 2 or options.height % 2
    ):
        raise MediaError("Use an even resolution and a supported output frame rate.")
    if options.encoder not in ("libx264", "h264_nvenc", "h264_qsv", "h264_amf"):
        raise MediaError("Unsupported H.264 encoder.")
    script, labels = graph(analyses, segments, options)
    if options.watermark_path:
        logo = Path(options.watermark_path)
        if not logo.is_file() or logo.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise MediaError("Choose an existing PNG, JPEG, or WebP channel watermark.")
        if (options.watermark_corner not in {"top-left", "top-right", "bottom-left", "bottom-right"}
                or not 4 <= options.watermark_width <= 35 or not 10 <= options.watermark_opacity <= 100):
            raise MediaError("The channel watermark settings are outside the supported range.")
        margin = max(4, round(min(options.width, options.height) * .025))
        width = max(2, options.width * options.watermark_width // 100)
        height = max(2, options.height * options.watermark_width // 100)
        x = f"main_w-overlay_w-{margin}" if options.watermark_corner.endswith("right") else str(margin)
        y = f"main_h-overlay_h-{margin}" if options.watermark_corner.startswith("bottom") else str(margin)
        script += (
            f";\n[{len(analyses)}:v:0]setpts=PTS-STARTPTS,format=rgba,"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"format=rgba,colorchannelmixer=aa={options.watermark_opacity/100:.2f}[logo];\n"
            f"[{labels[0]}][logo]overlay=x={x}:y={y}:eof_action=repeat:format=yuv420[vwater]"
        )
        labels[0] = "vwater"
    script_path = temporary / "filters.txt"
    script_path.write_text(script, encoding="ascii")
    progress_file = temporary / "progress.txt"
    expected = sum(s.frames - s.transition_frames for s in segments) / options.fps
    read_pos = 0
    pending = ""

    def progress():
        nonlocal read_pos, pending
        if not progress_file.exists():
            return
        with progress_file.open("rb") as stream:
            stream.seek(read_pos)
            pending += stream.read().decode("ascii", "replace")
            read_pos = stream.tell()
        lines = pending.split("\n")
        pending = lines.pop()
        for line in lines:
            key, _, value = line.strip().partition("=")
            if key == "out_time_us" and value.lstrip("-").isdigit():
                notify(min(.99, max(0., int(value) / 1_000_000 / expected)))

    args = ["-v", "error", "-xerror", "-nostdin", "-y", "-copyts",
            "-filter_complex_threads", "2", "-progress", progress_file, "-nostats"]
    for item in analyses:
        args.extend(["-threads", "2", "-i", item.clip.path])
    if options.watermark_path:
        args.extend(["-i", options.watermark_path])
    args.extend(["-/filter_complex", script_path])
    for label in labels:
        args.extend(["-map", f"[{label}]"])
    args.extend(["-c:v", options.encoder, "-pix_fmt", "yuv420p"])
    if options.encoder == "libx264":
        args.extend(["-preset", "fast", "-crf", "18", "-threads:v", "4"])
    args.extend(["-r", options.fps, "-fps_mode", "cfr", "-c:a", "aac", "-b:a", "256k",
                 "-movflags", "+faststart", destination])
    run("ffmpeg", args, cancel, progress=progress)
    output = probe(destination, cancel)
    tolerance = max(.09, 2 / options.fps)
    if (abs(output.duration - expected) > tolerance or abs(output.audio_end - expected) > tolerance
            or len(output.audio) != len(labels) - 1
            or (output.width, output.height) != (options.width, options.height)
            or abs(output.fps - options.fps) > .01):
        raise MediaError(f"Export verification failed. Expected {expected:.3f}s, "
                         f"found video {output.duration:.3f}s and audio {output.audio_end:.3f}s.")
    notify(1.)
    return expected

