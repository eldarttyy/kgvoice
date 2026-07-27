"""Recording manifests, audio QC, and entity-weighted WER.  *Skeleton only.*

This is the audio half of the toolkit, and the part that turns the phonological
work into something a speech pipeline can consume.

Planned components:

* **manifest** — build a balanced recording prompt list from corpus sentences,
  stratified so that every vowel-harmony class and every entity label is
  represented. Output is a CSV a speaker can read straight down.
* **qc** — per-clip acceptance scoring: peak/RMS level, estimated SNR, clipping,
  silence-lead/tail, sample-rate and bit-depth conformance. Separate accept
  thresholds for TTS-grade and ASR-grade use, because they are not the same bar.
* **wer** — word error rate, plus an entity-weighted variant. A transcription
  that gets every function word right and the one place name wrong is not 95%
  correct for any downstream purpose that matters.
* **prosody** — the disfluency/prosody markup scheme and its validator.

Planned surface::

    kgvoice bench manifest --n 200 --out prompts.csv
    kgvoice bench qc audio/session-01/ --profile tts
    kgvoice bench wer  ref.txt hyp.txt --entity-weighted
"""

__all__: list[str] = []
