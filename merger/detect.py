from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from math import ceil

import librosa
import numpy as np
from scipy.signal import butter, correlate, sosfiltfilt
from scipy.spatial import cKDTree

from .analysis import AUDIO_RATE, RATE, Analysis
from .media import check_cancel


@dataclass(frozen=True)
class Settings:
    minimum_seconds: float = 1.25
    hash_similarity: float = 0.86
    tile_similarity: float = 0.94
    audio_correlation: float = 0.96


@dataclass(frozen=True)
class Match:
    earlier: int
    later: int
    earlier_start: int
    later_start: int
    frames: int
    confidence: float


def offsets(a: Analysis, b: Analysis, same: bool, minimum: int, settings: Settings, cancel):
    """Vote for constant-speed alignments using local hash matches."""
    votes = Counter()
    tree = cKDTree(a.hashes)
    for base in range(0, len(b.hashes), 512):
        check_cancel(cancel)
        distances, indices = tree.query(b.hashes[base:base + 512], k=min(16, len(a.hashes)),
                                        distance_upper_bound=(64 * (1 - settings.hash_similarity)) ** .5 + 1e-8)
        for row, candidates in enumerate(np.atleast_2d(indices).reshape(len(distances), -1)):
            target = base + row
            for source in candidates:
                if source == len(a.hashes) or (same and target - source < minimum):
                    continue
                votes[target - int(source)] += 1
    for offset, count in votes.most_common(192):
        if count >= max(2, minimum // 3):
            yield offset


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2], edges[1::2])


def visual_runs(a, b, offset, same, minimum, settings):
    start = max(0, -offset)
    end = min(len(a.tiles), len(b.tiles) - offset)
    if same:
        # A later interval may never use itself as evidence.
        limit = offset
    else:
        limit = end - start
    left = a.tiles[start:end].astype(np.int16)
    right = b.tiles[start + offset:end + offset].astype(np.int16)
    if len(left) < minimum or limit < minimum:
        return
    pixels = 1 - np.mean(np.abs(left - right), axis=1) / 255
    hashes = 1 - np.mean(a.hashes[start:end] != b.hashes[start + offset:end + offset], axis=1)
    good = (pixels >= settings.tile_similarity) & (hashes >= settings.hash_similarity)
    # Internal scene cuts should line up to within one analysis frame.
    for scene in a.scenes:
        index = scene - start
        if 1 < index < len(good) - 1 and all(scene + offset + d not in b.scenes for d in (-1, 0, 1)):
            good[index] = False
    for scene in b.scenes:
        index = scene - start - offset
        if 1 < index < len(good) - 1 and all(scene - offset + d not in a.scenes for d in (-1, 0, 1)):
            good[index] = False
    for lo, hi in runs(good):
        if hi - lo >= minimum:
            yield start + lo, min(hi - lo, limit), float(np.mean((pixels[lo:hi] + hashes[lo:hi]) / 2))


def activity_matches(a, b, start, target, length):
    x = a.tiles[start:start + length].astype(np.int16)
    y = b.tiles[target:target + length].astype(np.int16)
    dx, dy = np.diff(x, axis=0), np.diff(y, axis=0)
    active = (np.abs(dx).mean(axis=1) > 1.1) | (np.abs(dy).mean(axis=1) > 1.1)
    if np.count_nonzero(active) < max(3, length // 8):
        return False
    dx, dy = dx[active].ravel().astype(float), dy[active].ravel().astype(float)
    norm = np.linalg.norm(dx) * np.linalg.norm(dy)
    return norm > 0 and float(np.dot(dx, dy) / norm) >= 0.72


def waveform(sound, start, stop):
    data = np.zeros((stop - start, sound.samples.shape[1]), dtype=np.float32)
    lo, hi = max(start, 0), min(stop, len(sound.samples))
    if lo < hi:
        data[lo - start:hi - start] = sound.samples[lo:hi]
    return data


@lru_cache(maxsize=1)
def audio_filters():
    hop = AUDIO_RATE // RATE
    mel = librosa.filters.mel(sr=AUDIO_RATE, n_fft=hop, n_mels=24)
    taper = np.hanning(hop + 1)[:-1]
    band = butter(4, (100, 3000), fs=AUDIO_RATE, btype="bandpass", output="sos")
    return mel, taper, band


def aligned_fingerprint(data, mel, taper):
    frames = data.reshape(-1, len(taper)) * taper
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return np.log1p(power @ mel.T).sum(axis=0)


def audio_window(a, b, start, target, length, threshold):
    if len(a.sounds) != len(b.sounds):
        return False
    if not a.sounds:
        return True
    hop = AUDIO_RATE // RATE
    margin = AUDIO_RATE // 16
    mel, taper, band = audio_filters()
    for first, second in zip(a.sounds, b.sounds):
        if first.samples.shape[1] != second.samples.shape[1]:
            return False
        x = waveform(first, start * hop, (start + length) * hop)
        y = waveform(second, target * hop - margin, (target + length) * hop + margin)
        center = y[margin:margin + len(x)]
        filtered_x = sosfiltfilt(band, x, axis=0)
        filtered_y = sosfiltfilt(band, y, axis=0)
        for channel in range(x.shape[1]):
            u, v = x[:, channel], center[:, channel]
            energy = float(np.linalg.norm(u))
            other = float(np.linalg.norm(v))
            silence = .0004 * np.sqrt(len(u))
            if max(energy, other) <= silence:
                continue
            if min(energy, other) <= silence:
                return False

            # AAC re-encoding can shift the waveform and decorrelate high frequencies.
            # Align in the speech band, then verify the full-band spectral fingerprint.
            ref, needle = filtered_y[:, channel], filtered_x[:, channel]
            filtered_energy = np.linalg.norm(needle)
            if filtered_energy < silence:
                ref, needle = y[:, channel], u
                filtered_energy = energy
            dots = correlate(ref, needle, mode="valid", method="fft")
            powers = np.r_[0., np.cumsum(ref.astype(np.float64) ** 2)]
            norms = np.sqrt(np.maximum(powers[len(u):] - powers[:-len(u)], 1e-15)) * filtered_energy
            correlations = dots / norms
            best = int(np.argmax(correlations))
            if correlations[best] < threshold:
                return False
            fp = first.fingerprint[start:start + length, :, channel].sum(axis=0)
            if best == margin:
                fq = second.fingerprint[target:target + length, :, channel].sum(axis=0)
            else:
                fq = aligned_fingerprint(y[best:best + len(u), channel], mel, taper)
            denominator = np.linalg.norm(fp) * np.linalg.norm(fq)
            if not denominator or float(np.dot(fp, fq) / denominator) < .90:
                return False
    return True


def words_match(a, b, start, target, length):
    def spoken(item, position):
        return tuple(word for lo, hi, word in item.words
                     if hi > position / RATE and lo < (position + length) / RATE)
    return spoken(a, start) == spoken(b, target)


def confirmed(analyses, settings, cancel, progress):
    minimum = max(4, ceil(settings.minimum_seconds * RATE))
    matches = []
    for later, b in enumerate(analyses):
        progress(later, len(analyses), b.clip.path.name)
        for earlier in range(later + 1):
            a = analyses[earlier]
            same = earlier == later
            for offset in offsets(a, b, same, minimum, settings, cancel):
                check_cancel(cancel)
                for start, length, score in visual_runs(a, b, offset, same, minimum, settings):
                    target = start + offset
                    if not activity_matches(a, b, start, target, length):
                        continue
                    # Verify small audio windows and keep any region with changed audio.
                    verified = np.zeros(length, dtype=bool)
                    for pos in range(0, length, 4):
                        check_cancel(cancel)
                        size = min(4, length - pos)
                        verified[pos:pos + size] = audio_window(
                            a, b, start + pos, target + pos, size, settings.audio_correlation)
                    for lo, hi in runs(verified):
                        count = int(hi - lo)
                        if count >= minimum and words_match(a, b, start + lo, target + lo, count):
                            # The first decoded frame can be later than its analysis timestamp.
                            # Require one additional matched frame beyond the last removed interval.
                            safe = count - 1
                            if safe >= minimum:
                                matches.append(Match(earlier, later, int(start + lo), int(target + lo),
                                                     safe, round(score, 4)))
    return matches


def cumulative_revisions(analyses, settings, cancel):
    """Confirm a numbered sequence of progressively longer exports with a shared opening."""
    if len(analyses) < 2:
        return []
    records = []
    for index, (earlier, later) in enumerate(zip(analyses, analyses[1:])):
        check_cancel(cancel)
        if (later.clip.duration + 1 / RATE < earlier.clip.duration
                or len(earlier.sounds) != len(later.sounds)):
            return []
        minimum = max(5 * RATE, ceil(.75 * earlier.clip.duration * RATE))
        prefix = next((length for start, length, score in
                       visual_runs(earlier, later, 0, False, minimum, settings)
                       if start == 0 and score >= .975), 0)
        if not prefix or not activity_matches(earlier, later, 0, 0, prefix):
            return []
        anchors = np.linspace(RATE // 2, prefix - 2 * RATE, 9, dtype=int)
        checks = [audio_window(earlier, later, int(pos), int(pos), RATE, settings.audio_correlation)
                  for pos in anchors]
        if sum(checks) < 7:
            return []
        if earlier.words or later.words:
            if not words_match(earlier, later, RATE // 2, RATE // 2, prefix - RATE):
                return []
        records.append({"clip": index, "superseded_by": index + 1,
                        "matching_visual_prefix_seconds": prefix / RATE,
                        "audio_checks_passed": sum(checks), "audio_checks_total": len(checks),
                        "superseded_seconds": max(earlier.clip.duration, earlier.clip.audio_end)})
    return records
