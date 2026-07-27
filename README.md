# kgvoice

A Kyrgyz speech-data toolkit: phonology, grapheme-to-phoneme, prosody, and audio QC — built so that every linguistic claim it makes is re-derivable from a corpus rather than asserted.

**Status:** phonology layer implemented and tested; audio and localisation layers are documented skeletons. See [Module status](#module-status).

---

## Why

Kyrgyz speech tooling has a specific, checkable hole in it. The public inventory of Kyrgyz NLP resources ([awesome-kyrgyz-nlp](https://github.com/alexeyev/awesome-kyrgyz-nlp)) lists **no dedicated Kyrgyz phonetics or G2P resource at all**, and no dialect-stratified speech data. The largest Kyrgyz ASR corpus (CSLT, 128h / 163 speakers) is gated behind an email request with the original texts withheld.

Meanwhile the front end that existing systems do use is thin. [Akyl-AI's `tts-mini`](https://github.com/Akyl-AI/tts-mini) — Matcha-TTS, 13h of speech, 7,000 samples, single speaker — offloads phonemisation to `espeak-ng`, and its README states plainly:

> The text preprocessing does not include functionality for processing abbreviations and contractions; however, the built-in phonemizer can transcribe numbers, but to avoid errors, it is better to write numbers in words.

That is a reasonable engineering tradeoff and also a standing invitation: the parts of Kyrgyz that a general-purpose phonemiser gets wrong — vowel harmony in suffixes, uvular/velar dorsal alternation, loanword stress, non-stress-bearing morphemes — are exactly the parts a listener notices.

`kgvoice` writes that layer explicitly, over the one large annotated Kyrgyz corpus that is openly available: [KyrgyzNER](https://github.com/Akyl-AI/KyrgyzNER).

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/kgvoice --help
```

The corpus is downloaded on first use to `~/.cache/kgvoice/` (override with `KGVOICE_DATA`). It is **not** vendored — see [Data and licensing](#data-and-licensing).

## The harmony result

The headline claim is about one gap in the standard description of Kyrgyz rounding harmony.

The low archiphoneme `/A/` (surfacing as `а е о ө`) is usually stated as *"rounds after a back rounded vowel"*, which predicts `*кулдор` for the plural of `кул`. It does not. Rounding spreads to a **low** target only from a trigger that is itself low (`о ө`) or front (`ү`) — the high back rounded `у` does not spread. `кул` → `кулдар`, but `гүл` → `гүлдөр`.

This is the asymmetry Kaun's typology of rounding harmony predicts for a trigger that disagrees with its target in height while offering no front/back perceptual cue. It is stated as a rule in `kgvoice/phon/harmony.py` and re-derived from the corpus by:

```bash
kgvoice phon harmony-audit
```

Over **14,731 word types** from KyrgyzNER/train, across all eight suffix templates, the rule's prediction matches the observed majority in **every cell (0 mismatches)**, at 95.1–98.9% weighted agreement per template. The critical rows, from the dative `GA`:

```
trigger  predicted  observed         n    agree
о        о          о              158   92.4%
у        а          а              143   97.9%     <- high back rounded: no spreading
ө        ө          ө               69  100.0%
ү        ө          ө               77  100.0%     <- high front rounded: spreads
```

Agreement is below 100% partly because the audit uses deliberately shallow morphology — it segments any word ending in a plausible surface form of the template, so `доллар` is counted as a plural of `долл`. That noise lands in the minority column and is reported rather than filtered, which is the point: the audit surfaces its own error mode instead of hiding it.

## Entity-weighted WER

Plain WER treats every token alike, which hides the failure mode that matters: a transcript can score well and still be useless because what it got wrong was the name, the district, and the number. Function words are recoverable from context; named entities are not.

`kgvoice.bench.wer` computes the Levenshtein alignment once and reads three figures off it — overall WER, WER restricted to reference tokens inside an entity, and a per-label breakout. Insertions are charged to the span they fall inside (or the following one at a boundary), so a system cannot hallucinate freely around a name at no cost.

On a KyrgyzNER sentence with one entity token corrupted:

```
WER              8.33%   (1/12)
  entity        12.50%   (1/8)
  non-entity     0.00%
  S/D/I        1/0/0

label              ref    S    D    I      WER
LOCATION             6    0    0    0    0.00%
PERIOD               2    1    0    0   50.00%
```

The same error is 8.33% or 50% depending on which question you asked. Only one of those numbers predicts whether the transcript is usable.

## Module status

| Module | State | Notes |
|---|---|---|
| `kgvoice.corpus` | **working** | KyrgyzNER loader (tab-safe, BIO span decoding), on-demand download |
| `kgvoice.phon.alphabet` | **working** | Vowel/consonant feature inventory |
| `kgvoice.phon.harmony` | **working** | Suffix realisation + corpus audit |
| `kgvoice.phon.syllable` | **working** | Native template + loanword complex onsets |
| `kgvoice.phon.lexicon` | **working** | 71 hand-checked borrowed stems, 57 with source stress |
| `kgvoice.phon.loanword` | **working** | Graded score from orthography + lexicon |
| `kgvoice.phon.stress` | **working** | Final default, non-stress-bearing suffixes, lexicon lookup |
| `kgvoice.phon.g2p` | **working** | Dorsal backing, lateral velarisation, long vowels, iotation |
| `kgvoice.phon.profile` | **working** | Per-token/entity record + recording-difficulty score |
| `kgvoice.bench.wer` | **working** | WER, entity-weighted WER, per-label breakout |
| `kgvoice.bench` (rest) | **skeleton** | `manifest`, `qc`, `prosody` not written |
| `kgvoice.localize` | **skeleton** | Entity-preservation / placeholder-harmony scoring |
| `kgvoice.studio` | **skeleton** | Streamlit annotation UI |

`tests/` covers harmony (20 tests) and WER (21 tests). The other implemented modules are exercised through the CLI but do not yet have their own test files. `bench.wer` is importable as a library but is **not yet wired into the CLI** — `kgvoice bench wer` still exits 2.

## CLI

```bash
kgvoice corpus stats --split train --top 10

kgvoice phon harmony-audit          # re-derive the harmony table from corpus
kgvoice phon paradigm кул гүл       # full nominal paradigm
kgvoice phon check телефон          # flag stem-internal harmony violations (exit 1)
kgvoice phon ipa кол көл --narrow   # grapheme-to-phoneme
kgvoice phon syllabify программа    # прог-рам-ма
kgvoice phon stress барбасың        # бар-БА-сың
kgvoice phon loanword журнал
kgvoice phon profile президент      # full record + recording-difficulty score
```

Worked examples:

```
$ kgvoice phon ipa кол көл жыл саат
кол	qoɫ          # uvular + velarised lateral in a back context
көл	køl          # velar + clear lateral in a front context
жыл	d͡ʒɯɫ
саат	sɑːt         # digraph is one long vowel, not two

$ kgvoice phon profile президент
президент
  ipa         [pre.zi.ˈdent]
  stress      syllable 2 (loanword-lexicon/high)
  loanword    1.00
  signals     complex-onset +0.15, final-cluster +0.05, loanword +0.20
  difficulty  0.40
```

`difficulty` is what makes this useful to a recording pipeline rather than just
interesting: it ranks which of the corpus's entities are worth a human recording
and a human check, from independently observable signals rather than a guess.

## Known limitations

- **Loanword detection under-fires on harmony-conformant borrowings outside the lexicon.** Orthographic signals catch a borrowing when it carries a non-native letter (`компьютер`), a foreign initial cluster (`брошюра`), a borrowed suffix (`мотор`, 0.30), or a harmony violation. A word with *none* of those falls through: `радар` and `канал` both score 0.00. `журнал` is caught only because it is one of the 71 stems entered by hand in `phon/lexicon.py`. Coverage therefore ends where that list ends, and the list is deliberately small — an unverified entry produces confident nonsense, while a missing one produces `unknown` and gets reviewed. Closing this properly needs a real borrowed-stem lexicon, not more heuristics.
- The harmony audit's shallow segmentation is described above and is a known, reported source of minority-column noise.
- Nothing in `bench` exists yet, so there is no audio in this repository and no recorded prosody annotation. The scheme is specified in `kgvoice/bench/__init__.py`; the data is not collected.

## Roadmap

1. **`bench.manifest`** — build a recording prompt list from corpus sentences, stratified so every vowel-harmony class and entity label is represented, ordered by `phon.profile.rank_by_difficulty`.
2. **`bench.qc`** — per-clip acceptance scoring (peak/RMS, estimated SNR, clipping, lead/tail silence, format conformance) with *separate* thresholds for TTS-grade and ASR-grade use.
3. **`bench.prosody`** — the disfluency and prosody markup scheme, plus its validator.
4. **CLI wiring for `bench`** — `kgvoice bench wer ref.txt hyp.txt --entity-weighted`.
5. **`localize`** — entity preservation, `{placeholder}` suffix-harmony collisions, and сен/сиз register consistency across a string catalogue.

## Data and licensing

Code in this repository is MIT (see `pyproject.toml`).

KyrgyzNER is published by Akyl-AI under **CC BY-NC-SA 4.0** — 1,499 news articles from 24.kg (2017–2022), annotated by 59 Kyrgyz linguists. It is downloaded on demand rather than vendored, specifically so its share-alike obligation does not propagate to this code. Attribution is written to the cache directory on download.

Corpus statistics as loaded (train split): 7,033 sentences, 89,248 tokens, 10,881 entity spans across 24 labels, 28.0% of tokens inside an entity.

## Layout

```
kgvoice/
  corpus/     KyrgyzNER loader + downloader
  phon/       alphabet, harmony, syllable, lexicon, loanword, stress, g2p, profile
  localize/   skeleton
  bench/      skeleton
  studio/     skeleton
  cli.py
tests/        harmony test suite
docs/         (empty)
audio/        recordings — gitignored, see audio/README.md
```
