"""Audio QC: measurement accuracy, spec enforcement, and failure reporting.

Signals are synthesised with known properties so the assertions are about the
measurement rather than about a checked-in binary.
"""

import math
import wave

import numpy as np
import pytest

from kgvoice.bench import audio


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def make_signal(
    *,
    sample_rate=48000,
    duration=2.0,
    rms_dbfs=-23.0,
    gain=1.0,
    noise_dbfs=-80.0,
    lead=0.1,
    trail=0.1,
    seed=0,
):
    """Speech-like signal with a known RMS over its voiced region."""
    rng = np.random.default_rng(seed)
    n = int(sample_rate * duration)
    t = np.arange(n) / sample_rate
    env = (0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)) ** 2
    sig = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate([120, 240, 480, 960]))
    sig = sig * env
    k0, k1 = int(lead * sample_rate), int(trail * sample_rate)
    sig[:k0] = 0.0
    if k1:
        sig[n - k1 :] = 0.0
    voiced = sig[k0 : n - k1] if k1 else sig[k0:]
    sig = sig * (10 ** (rms_dbfs / 20)) / np.sqrt(np.mean(voiced**2))
    sig = sig + rng.normal(0, 10 ** (noise_dbfs / 20), n)
    return np.clip(sig * gain, -1.0, 1.0)


def write_wav(path, samples, sample_rate=48000, channels=1, width=2):
    data = samples
    if channels == 2:
        data = np.repeat(samples, 2)
    scale = 2 ** (width * 8 - 1) - 1
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(sample_rate)
        w.writeframes((data * scale).astype(f"<i{width}").tobytes())
    return path


@pytest.fixture
def clean_wav(tmp_path):
    return write_wav(tmp_path / "kg-0001.wav", make_signal())


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def test_reads_format_fields(clean_wav):
    qc = audio.analyze_file(clean_wav)
    assert qc.sample_rate == 48000
    assert qc.channels == 1
    assert qc.sample_width_bits == 16
    assert qc.duration_s == pytest.approx(2.0, abs=0.01)


def test_rms_is_measured_accurately():
    """A signal built to -23 dBFS over its voiced region should measure near it.

    The whole-file RMS sits slightly below the voiced-region RMS because of the
    silent lead and tail, so the tolerance is one-sided and generous.
    """
    qc = audio.analyze_array(make_signal(rms_dbfs=-23.0), 48000)
    assert -26.0 < qc.rms_dbfs < -22.0


def test_peak_and_clipping_on_a_hot_signal():
    qc = audio.analyze_array(make_signal(gain=20.0), 48000)
    assert qc.peak_dbfs > -0.5
    assert qc.clipped_fraction > 0.0
    assert "clipping" in qc.problem_codes()


def test_clean_signal_reports_no_clipping(clean_wav):
    assert audio.analyze_file(clean_wav).clipped_fraction == 0.0


def test_snr_separates_quiet_and_noisy_rooms():
    quiet = audio.analyze_array(make_signal(noise_dbfs=-90.0), 48000)
    noisy = audio.analyze_array(make_signal(noise_dbfs=-30.0), 48000)
    assert quiet.snr_db > noisy.snr_db + 15
    assert quiet.snr_band in ("good", "excellent")
    assert noisy.snr_band in ("noisy", "unusable")


def test_lead_silence_is_located():
    qc = audio.analyze_array(make_signal(lead=0.8, trail=0.05), 48000)
    assert qc.lead_silence_s == pytest.approx(0.8, abs=0.1)


def test_empty_input_does_not_crash():
    qc = audio.analyze_array(np.array([]), 48000)
    assert qc.duration_s == 0.0
    assert not qc.passed


def test_dc_offset_is_detected():
    qc = audio.analyze_array(make_signal() + 0.05, 48000)
    assert qc.dc_offset == pytest.approx(0.05, abs=0.01)
    assert "dc-offset" in qc.problem_codes()


# --------------------------------------------------------------------------
# Spec enforcement
# --------------------------------------------------------------------------


def test_clean_take_passes_default_spec(clean_wav):
    assert audio.analyze_file(clean_wav).passed


def test_strict_spec_is_at_least_as_demanding(clean_wav):
    default = audio.analyze_file(clean_wav, spec=audio.RecordingSpec())
    strict = audio.analyze_file(clean_wav, spec=audio.RecordingSpec.strict())
    assert set(default.problem_codes()) <= set(strict.problem_codes())


def test_low_sample_rate_fails(tmp_path):
    p = write_wav(tmp_path / "low.wav", make_signal(sample_rate=8000), sample_rate=8000)
    assert "sample-rate" in audio.analyze_file(p).problem_codes()


def test_stereo_fails(tmp_path):
    p = write_wav(tmp_path / "stereo.wav", make_signal(), channels=2)
    assert "channels" in audio.analyze_file(p).problem_codes()


def test_quiet_take_reports_level_low():
    qc = audio.analyze_array(make_signal(rms_dbfs=-50.0), 48000)
    assert "level-low" in qc.problem_codes()
    assert "too quiet" in " ".join(qc.problems())


def test_problem_codes_are_stable_across_measured_values():
    """Two takes quiet by different amounts share one code but differ in message.

    This is what lets a batch report say "six takes were too quiet" instead of
    listing six near-identical strings.
    """
    a = audio.analyze_array(make_signal(rms_dbfs=-40.0), 48000)
    b = audio.analyze_array(make_signal(rms_dbfs=-50.0), 48000)
    assert a.problem_codes() == b.problem_codes() == ["level-low"]
    assert a.problems() != b.problems()


# --------------------------------------------------------------------------
# Batch handling
# --------------------------------------------------------------------------


def test_unreadable_file_is_reported_not_raised(tmp_path):
    (tmp_path / "broken.wav").write_bytes(b"this is not a wav")
    results = audio.analyze_directory(tmp_path)
    assert len(results) == 1
    assert not results[0].passed
    assert "unreadable" in results[0].problem_codes()


def test_analyze_file_raises_on_unreadable(tmp_path):
    p = tmp_path / "broken.wav"
    p.write_bytes(b"nope")
    with pytest.raises(ValueError, match="PCM WAV"):
        audio.analyze_file(p)


def test_directory_batch_summary(tmp_path):
    write_wav(tmp_path / "a.wav", make_signal())
    write_wav(tmp_path / "b.wav", make_signal(rms_dbfs=-50.0))
    write_wav(tmp_path / "c.wav", make_signal(rms_dbfs=-48.0))
    summary = audio.summarize(audio.analyze_directory(tmp_path))
    assert summary["files"] == 3
    assert summary["passed"] == 1
    assert summary["common_problems"]["level-low"] == 2


def test_summarize_empty():
    assert audio.summarize([])["files"] == 0


def test_as_dict_is_json_safe(clean_wav):
    import json

    d = audio.analyze_file(clean_wav).as_dict()
    assert json.loads(json.dumps(d))["passed"] is True
    assert not math.isnan(d["rms_dbfs"])
