from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import imagehash
import librosa
import numpy as np
from PIL import Image
from scenedetect import AdaptiveDetector

from .media import Clip, MediaError, check_cancel, run


RATE = 8
AUDIO_RATE = 16000
FRAME_W, FRAME_H = 80, 48
FRAME_BYTES = FRAME_W * FRAME_H * 3


@dataclass
class Sound:
    samples: np.ndarray
    fingerprint: np.ndarray


@dataclass
class Analysis:
    clip: Clip
    hashes: np.ndarray
    tiles: np.ndarray
    scenes: frozenset[int]
    sounds: list[Sound]
    words: tuple[tuple[float, float, str], ...] = ()


def extract(clip: Clip, directory: Path, cancel, log, transcriber=None) -> Analysis:
    frames_path = directory / "frames.rgb"
    vf = (f"setpts=PTS-({clip.origin:.9f})/TB,fps={RATE}:start_time=0:round=up,"
          f"scale={FRAME_W}:{FRAME_H}:flags=area")
    with frames_path.open("wb") as stream:
        run("ffmpeg", ["-v", "error", "-xerror", "-copyts", "-i", clip.path,
                       "-map", f"0:{clip.video}", "-an", "-filter_threads", "1", "-vf", vf,
                       "-pix_fmt", "bgr24", "-c:v", "rawvideo", "-threads", "1", "-f", "rawvideo", "-"],
            cancel, stdout=stream)
    count = min(frames_path.stat().st_size // FRAME_BYTES, math.ceil(clip.duration * RATE))
    if count == 0:
        raise MediaError("No frames decoded for analysis.")
    hashes = np.empty((count, 64), dtype=bool)
    tiles = np.empty((count, 12 * 8 * 3), dtype=np.uint8)
    detector = AdaptiveDetector(min_scene_len=max(2, RATE // 2))
    scenes = set()
    with frames_path.open("rb") as stream:
        for index in range(count):
            check_cancel(cancel)
            frame = np.frombuffer(stream.read(FRAME_BYTES), np.uint8).reshape(FRAME_H, FRAME_W, 3)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hashes[index] = imagehash.phash(Image.fromarray(gray)).hash.ravel()
            tiles[index] = cv2.resize(frame, (12, 8), interpolation=cv2.INTER_AREA).ravel()
            for cut in detector.process_frame(index, frame):
                scenes.add(cut)
    frames_path.unlink()

    sounds = []
    try:
        for position, stream_index in enumerate(clip.audio):
            channels = clip.audio_channels[position]
            sound_path = directory / f"sound-{position}.f32"
            af = f"asetpts=PTS-({clip.origin:.9f})/TB,aresample={AUDIO_RATE}:async=1:first_pts=0"
            with sound_path.open("wb") as stream:
                run("ffmpeg", ["-v", "error", "-xerror", "-copyts", "-i", clip.path,
                               "-map", f"0:{stream_index}", "-vn", "-af", af, "-t", f"{clip.duration:.9f}",
                               "-ac", channels, "-c:a", "pcm_f32le", "-f", "f32le", "-"],
                    cancel, stdout=stream)
            length = sound_path.stat().st_size // (4 * channels)
            samples = (np.memmap(sound_path, dtype="<f4", mode="r", shape=(length, channels))
                       if length else np.empty((0, channels), dtype=np.float32))
            sounds.append(Sound(samples, np.empty(0)))
            sounds[-1].fingerprint = fingerprint_audio(samples, cancel)

        words = ()
        if transcriber is not None and sounds and len(sounds[0].samples):
            log("Transcribing the primary audio track locally.")
            words = transcriber(sounds[0].samples, cancel)
    except BaseException:
        close_sounds(sounds)
        raise
    return Analysis(clip, hashes, tiles, frozenset(scenes), sounds, words)


def fingerprint_audio(samples, cancel):
    hop = AUDIO_RATE // RATE
    count = math.ceil(len(samples) / hop)
    fingerprint = np.zeros((count, 24, samples.shape[1]), dtype=np.float32)
    for start in range(0, count, RATE * 30):
        check_cancel(cancel)
        stop = min(count, start + RATE * 30)
        block = np.asarray(samples[start * hop:stop * hop]).T
        block = np.pad(block, ((0, 0), (0, (stop - start) * hop - block.shape[1])))
        mel = librosa.feature.melspectrogram(y=block, sr=AUDIO_RATE, n_fft=hop, hop_length=hop,
                                            n_mels=24, center=False)
        fingerprint[start:stop] = np.moveaxis(np.log1p(mel), -1, 0).transpose(0, 2, 1)
    return fingerprint


def release(analyses):
    for item in analyses:
        close_sounds(item.sounds)


def close_sounds(sounds):
    for sound in sounds:
        if isinstance(sound.samples, np.memmap):
            sound.samples._mmap.close()


class Transcriber:
    def __init__(self, path):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise MediaError("Install requirements-speech.txt to enable speech analysis.") from exc
        self.model = WhisperModel(str(path), device="cpu", compute_type="int8", local_files_only=True)

    def __call__(self, samples, cancel):
        segments, _ = self.model.transcribe(samples.mean(axis=1), word_timestamps=True, vad_filter=True)
        words = []
        for segment in segments:
            check_cancel(cancel)
            for word in segment.words or ():
                clean = "".join(c for c in word.word.casefold() if c.isalnum())
                if clean:
                    words.append((word.start, word.end, clean))
        return tuple(words)

