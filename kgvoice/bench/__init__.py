"""Recording manifests, audio QC, and entity-weighted WER.

This is the audio half of the toolkit, and the part that turns the phonological
work into something a speech pipeline can consume. The stages run in order::

    select    score corpus sentences for benchmark value, pick a diverse set
    manifest  turn the selection into prompts + pronunciation guidance
      (record)
    audio     measure each take against a spec, pass or fail it
    prosody   annotate what went wrong, with evidence
    wer       score a transcript, weighted toward named entities

* ``select`` — sentence scoring for benchmark value: entity density, numbers and
  dates, acronyms, and mean phonetic difficulty from
  :func:`kgvoice.phon.profile.profile_word`. Selection is greedy over *phoneme
  coverage*, so a set of 100 prompts is diverse rather than 100 variations on one
  hard sentence.
* ``manifest`` — the contract between whoever picks the sentences and whoever
  reads them: prompt text, gold entity spans, and a pronunciation hint for the
  few words per sentence a reader would otherwise get wrong. JSON, round-trips
  exactly, resumable.
* ``audio`` — per-clip acceptance: peak/RMS level, estimated SNR, clipping,
  silence lead/tail, DC offset, sample-rate and bit-depth conformance. Two specs,
  because TTS-grade and ASR-grade are not the same bar
  (:meth:`~kgvoice.bench.audio.RecordingSpec.strict` is the former).
* ``prosody`` — the disfluency/prosody markup scheme, its validator, and
  inter-annotator agreement. Every tag carries the evidence needed to check it.
* ``wer`` — word error rate plus an entity-weighted variant. A transcription that
  gets every function word right and the one place name wrong is not 95% correct
  for any downstream purpose that matters.

Only ``audio`` needs NumPy, and ``manifest`` imports it lazily, so planning and
printing a recording session works on a machine that cannot process audio.

CLI surface::

    kgvoice bench select   --n 200 --out manifest.json
    kgvoice bench prompts  manifest.json --out prompts.txt
    kgvoice bench qc       audio/ --manifest manifest.json --spec strict
    kgvoice bench wer      --manifest manifest.json --hyp transcripts.jsonl
"""

from kgvoice.bench import audio, manifest, prosody, select, wer
from kgvoice.bench.audio import AudioQC, RecordingSpec, analyze_directory, analyze_file
from kgvoice.bench.manifest import Manifest, ManifestItem, attach_recordings, build
from kgvoice.bench.prosody import TAGS, ProsodyAnnotation, ProsodyTag
from kgvoice.bench.select import SentenceScore, coverage_report, rank, score_sentence
from kgvoice.bench.wer import WERResult, align, labels_from_sentence, score, tokenize

__all__ = [
    # submodules
    "audio",
    "manifest",
    "prosody",
    "select",
    "wer",
    # select
    "SentenceScore",
    "score_sentence",
    "rank",
    "coverage_report",
    # manifest
    "Manifest",
    "ManifestItem",
    "build",
    "attach_recordings",
    # audio
    "AudioQC",
    "RecordingSpec",
    "analyze_file",
    "analyze_directory",
    # prosody
    "ProsodyAnnotation",
    "ProsodyTag",
    "TAGS",
    # wer
    "WERResult",
    "score",
    "align",
    "tokenize",
    "labels_from_sentence",
]
