"""Technical QC for recorded audio.

Before anyone listens to a take, most of what makes it unusable is measurable:
wrong sample rate, clipping, a noisy room, a level that will force the trainer to
normalise and drag the noise floor up with it, or three seconds of silence at the
head. This module reads a WAV and reports those, with an explicit pass/fail
against a named spec so a contributor gets "re-record, it clipped" rather than a
wall of numbers.

Only the standard library and NumPy are used — no ``soundfile``, no ``librosa``.
A QC tool that a contributor cannot install is a QC tool that does not get run,
and WAV is what a recording workflow should be producing anyway.

The SNR estimate deserves a caveat, since it is the one number here that is an
estimate rather than a measurement: it splits frames into speech and silence by
energy percentile and compares them. That is reliable enough to catch a bad room
or a noisy preamp, and not reliable enough to quote to two decimal places. It is
reported as a band, and :meth:`AudioQC.problems` phrases it as such.
"""

from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Full scale for 16-bit PCM.
_INT16_MAX = 32768.0


@dataclass(frozen=True)
class RecordingSpec:
    """What a take has to satisfy to be accepted."""

    name: str = "default"
    sample_rate: int = 48000
    min_sample_rate: int = 16000
    channels: int = 1
    min_duration_s: float = 0.4
    max_duration_s: float = 30.0
    #: Target integrated level. -23 dBFS RMS is a conventional speech-corpus
    #: target: loud enough to sit well above dither, quiet enough to leave
    #: headroom for peaks without limiting.
    target_rms_dbfs: float = -23.0
    rms_tolerance_db: float = 6.0
    max_peak_dbfs: float = -1.0
    max_clipped_fraction: float = 0.0001
    min_snr_db: float = 20.0
    max_lead_silence_s: float = 1.0
    max_trail_silence_s: float = 1.5

    @classmethod
    def strict(cls) -> "RecordingSpec":
        """Tighter bounds for material intended to train a voice, not just test one."""
        return cls(
            name="strict",
            min_sample_rate=44100,
            rms_tolerance_db=4.0,
            max_peak_dbfs=-3.0,
            max_clipped_fraction=0.0,
            min_snr_db=30.0,
            max_lead_silence_s=0.5,
            max_trail_silence_s=0.7,
        )


@dataclass
class AudioQC:
    """Measured properties of one recording."""

    path: str
    sample_rate: int
    channels: int
    sample_width_bits: int
    duration_s: float
    peak_dbfs: float
    rms_dbfs: float
    clipped_fraction: float
    snr_db: float
    dc_offset: float
    lead_silence_s: float
    trail_silence_s: float
    spec: RecordingSpec = field(default_factory=RecordingSpec)
    #: Set when the file could not be decoded at all; it then fails QC on
    #: that ground alone and every measurement below is meaningless.
    unreadable: str = ""

    @property
    def snr_band(self) -> str:
        """Coarse, honest bucket for the SNR estimate."""
        if self.snr_db >= 40:
            return "excellent"
        if self.snr_db >= 30:
            return "good"
        if self.snr_db >= 20:
            return "acceptable"
        if self.snr_db >= 10:
            return "noisy"
        return "unusable"

    def findings(self) -> list[tuple[str, str]]:
        """Every spec violation as ``(code, message)``.

        The code is a stable identifier for the *kind* of problem; the message
        carries the measured value and what to do about it. Both are needed:
        aggregating on the message would split ``level -33.0 dBFS is too quiet``
        and ``level -37.8 dBFS is too quiet`` into different buckets, which makes
        a batch report useless for spotting that six of ten takes share one cause.
        """
        s = self.spec
        out: list[tuple[str, str]] = []
        if self.sample_rate < s.min_sample_rate:
            out.append((
                "sample-rate",
                f"sample rate {self.sample_rate} Hz is below the {s.min_sample_rate} Hz minimum",
            ))
        if self.channels != s.channels:
            out.append((
                "channels", f"{self.channels} channels; expected {s.channels} (mono)"
            ))
        if self.duration_s < s.min_duration_s:
            out.append((
                "too-short", f"only {self.duration_s:.2f}s long — likely a truncated take"
            ))
        if self.duration_s > s.max_duration_s:
            out.append((
                "too-long", f"{self.duration_s:.1f}s exceeds the {s.max_duration_s:.0f}s limit"
            ))
        if self.clipped_fraction > s.max_clipped_fraction:
            out.append((
                "clipping",
                f"clipping on {self.clipped_fraction:.3%} of samples — lower the input gain "
                "and re-record",
            ))
        if self.peak_dbfs > s.max_peak_dbfs:
            out.append((
                "headroom",
                f"peak {self.peak_dbfs:.1f} dBFS leaves too little headroom "
                f"(want below {s.max_peak_dbfs:.0f})",
            ))
        if abs(self.rms_dbfs - s.target_rms_dbfs) > s.rms_tolerance_db:
            quiet = self.rms_dbfs < s.target_rms_dbfs
            out.append((
                "level-low" if quiet else "level-high",
                f"level {self.rms_dbfs:.1f} dBFS is too {'quiet' if quiet else 'loud'} "
                f"(target {s.target_rms_dbfs:.0f} +/- {s.rms_tolerance_db:.0f} dB)",
            ))
        if self.snr_db < s.min_snr_db:
            out.append((
                "snr",
                f"signal-to-noise looks {self.snr_band} (~{self.snr_db:.0f} dB) — "
                "check for room noise, fans, or handling",
            ))
        if self.lead_silence_s > s.max_lead_silence_s:
            out.append((
                "lead-silence", f"{self.lead_silence_s:.1f}s of silence before speech starts"
            ))
        if self.trail_silence_s > s.max_trail_silence_s:
            out.append((
                "trail-silence", f"{self.trail_silence_s:.1f}s of silence after speech ends"
            ))
        if abs(self.dc_offset) > 0.01:
            out.append((
                "dc-offset",
                f"DC offset {self.dc_offset:+.3f} — check the interface or apply a highpass",
            ))
        if self.unreadable:
            out.append(("unreadable", self.unreadable))
        return out

    def problems(self) -> list[str]:
        """Human-facing messages for every spec violation."""
        return [msg for _code, msg in self.findings()]

    def problem_codes(self) -> list[str]:
        """Stable identifiers for every spec violation."""
        return [code for code, _msg in self.findings()]

    @property
    def passed(self) -> bool:
        return not self.problems()

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width_bits": self.sample_width_bits,
            "duration_s": round(self.duration_s, 3),
            "peak_dbfs": round(self.peak_dbfs, 2),
            "rms_dbfs": round(self.rms_dbfs, 2),
            "clipped_fraction": round(self.clipped_fraction, 6),
            "snr_db": round(self.snr_db, 1),
            "snr_band": self.snr_band,
            "dc_offset": round(self.dc_offset, 5),
            "lead_silence_s": round(self.lead_silence_s, 3),
            "trail_silence_s": round(self.trail_silence_s, 3),
            "spec": self.spec.name,
            "passed": self.passed,
            "problems": self.problems(),
            "problem_codes": self.problem_codes(),
        }

    def format(self) -> str:  # pragma: no cover - display helper
        status = "PASS" if self.passed else "FAIL"
        head = (
            f"{status}  {Path(self.path).name}\n"
            f"  {self.sample_rate} Hz  {self.channels}ch  {self.sample_width_bits}-bit  "
            f"{self.duration_s:.2f}s\n"
            f"  peak {self.peak_dbfs:6.1f} dBFS   rms {self.rms_dbfs:6.1f} dBFS   "
            f"snr ~{self.snr_db:.0f} dB ({self.snr_band})"
        )
        for p in self.problems():
            head += f"\n  - {p}"
        return head


def _to_float_mono(data: np.ndarray, channels: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mono, per_channel)`` as float in [-1, 1]."""
    if channels > 1:
        per_channel = data.reshape(-1, channels).T
        mono = per_channel.mean(axis=0)
    else:
        per_channel = data.reshape(1, -1)
        mono = data
    return mono, per_channel


def _dbfs(x: float) -> float:
    return 20.0 * math.log10(x) if x > 1e-12 else -120.0


def _frame_rms(x: np.ndarray, frame: int) -> np.ndarray:
    if len(x) < frame:
        return np.array([float(np.sqrt(np.mean(x**2)))]) if len(x) else np.array([0.0])
    n = len(x) // frame
    frames = x[: n * frame].reshape(n, frame)
    return np.sqrt(np.mean(frames**2, axis=1))


def analyze_array(
    samples: np.ndarray,
    sample_rate: int,
    *,
    channels: int = 1,
    sample_width_bits: int = 16,
    path: str = "<array>",
    spec: RecordingSpec | None = None,
    clip_threshold: float = 0.999,
) -> AudioQC:
    """Measure an in-memory float array in [-1, 1]. Used by :func:`analyze_file`."""
    spec = spec or RecordingSpec()
    mono, _ = _to_float_mono(samples, channels)
    if mono.size == 0:
        return AudioQC(path, sample_rate, channels, sample_width_bits, 0.0, -120.0, -120.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, spec)

    duration = len(mono) / float(sample_rate)
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono**2)))
    clipped = float(np.mean(np.abs(mono) >= clip_threshold))
    dc = float(np.mean(mono))

    # 20 ms frames — long enough to be stable, short enough to resolve pauses.
    frame = max(1, int(0.020 * sample_rate))
    rms_frames = _frame_rms(mono, frame)

    # Speech vs noise by energy percentile. The gap between a high and a low
    # percentile is a more robust SNR proxy than max/min, which any single click
    # would dominate.
    if rms_frames.size >= 4:
        speech_level = float(np.percentile(rms_frames, 90))
        noise_level = float(np.percentile(rms_frames, 10))
        snr = _dbfs(speech_level) - _dbfs(max(noise_level, 1e-9))
    else:
        snr = 0.0

    # Silence trimming uses a threshold relative to the speech level, so it
    # adapts to a quiet recording instead of assuming an absolute floor.
    threshold = max(float(np.percentile(rms_frames, 90)) * 0.1, 1e-4)
    voiced = np.nonzero(rms_frames > threshold)[0]
    if voiced.size:
        lead = float(voiced[0] * frame / sample_rate)
        trail = float((len(rms_frames) - 1 - voiced[-1]) * frame / sample_rate)
    else:
        lead, trail = duration, 0.0

    return AudioQC(
        path=path,
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bits=sample_width_bits,
        duration_s=duration,
        peak_dbfs=_dbfs(peak),
        rms_dbfs=_dbfs(rms),
        clipped_fraction=clipped,
        snr_db=snr,
        dc_offset=dc,
        lead_silence_s=lead,
        trail_silence_s=trail,
        spec=spec,
    )


def analyze_file(path: str | Path, spec: RecordingSpec | None = None) -> AudioQC:
    """Read a PCM WAV file and measure it.

    Raises ``ValueError`` with a readable message for formats ``wave`` cannot
    handle (compressed WAV, 24-bit packed, or a non-WAV file with a .wav name) —
    a contributor should be told to export 16-bit PCM, not shown a struct error.
    """
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError, struct.error) as exc:
        # `wave` signals a malformed file three different ways depending on where
        # parsing gives up: wave.Error for a bad chunk, EOFError for a file too
        # short to hold a header, struct.error for a truncated one. All three
        # mean the same thing to whoever recorded it.
        detail = exc or type(exc).__name__
        raise ValueError(
            f"{path.name} is not a readable PCM WAV ({detail}). "
            "Export as uncompressed 16-bit or 32-bit PCM WAV."
        ) from exc

    if width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float64) / _INT16_MAX
    elif width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float64) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    else:
        raise ValueError(
            f"{path.name} uses {width * 8}-bit samples, which this reader does not support. "
            "Export as 16-bit PCM WAV."
        )

    return analyze_array(
        data,
        rate,
        channels=channels,
        sample_width_bits=width * 8,
        path=str(path),
        spec=spec,
    )


def analyze_directory(
    directory: str | Path, spec: RecordingSpec | None = None, pattern: str = "*.wav"
) -> list[AudioQC]:
    """QC every WAV in a directory, sorted by name. Unreadable files are skipped
    with their error attached rather than aborting the batch."""
    results: list[AudioQC] = []
    for p in sorted(Path(directory).glob(pattern)):
        try:
            results.append(analyze_file(p, spec=spec))
        except ValueError as exc:
            results.append(
                AudioQC(
                    str(p), 0, 0, 0, 0.0, -120.0, -120.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    spec or RecordingSpec(), unreadable=str(exc),
                )
            )
    return results


def summarize(results: list[AudioQC]) -> dict:
    """Batch-level summary for a QC report."""
    if not results:
        return {"files": 0, "passed": 0, "failed": 0}
    passed = [r for r in results if r.passed]
    durations = [r.duration_s for r in results]
    problems: dict[str, int] = {}
    for r in results:
        for code in r.problem_codes():
            problems[code] = problems.get(code, 0) + 1
    return {
        "files": len(results),
        "passed": len(passed),
        "failed": len(results) - len(passed),
        "total_duration_s": round(sum(durations), 1),
        "mean_duration_s": round(sum(durations) / len(durations), 2),
        "mean_rms_dbfs": round(sum(r.rms_dbfs for r in results) / len(results), 1),
        "mean_snr_db": round(sum(r.snr_db for r in results) / len(results), 1),
        "common_problems": dict(sorted(problems.items(), key=lambda kv: -kv[1])),
    }
