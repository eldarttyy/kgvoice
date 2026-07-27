# kgvoice

[![test](https://github.com/eldarttyy/kgvoice/actions/workflows/test.yml/badge.svg)](https://github.com/eldarttyy/kgvoice/actions/workflows/test.yml)

A Kyrgyz speech-data toolkit: phonology, grapheme-to-phoneme, prosody, and audio QC — built so that every linguistic claim it makes is re-derivable from a corpus rather than asserted.

**Status:** phonology, audio-benchmark, and localisation-audit layers implemented and tested (129 tests); the annotation UI is a documented skeleton. See [Module status](#module-status).

This toolkit is general-purpose and not tied to any one employer. It also
backs a specific portfolio application built on top of it:
[`kyrgyz-speech-eval-suite`](https://github.com/eldarttyy/kyrgyz-speech-eval-suite).

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

## Localisation audit

A translated UI string can read as fluent Kyrgyz and still be wrong in a way a proofreader won't catch: Kyrgyz is agglutinative, so a case suffix glued directly onto a `{placeholder}` is chosen by the *runtime* value's last vowel, not by the template. `{name}ге чакыруу жөнөттүңүз` ("send an invite to {name}") is correct only when `name` ends the way `-ге` expects — wrong for `Айбек` (wants `-ке`), `Нурлан` (`-га`), `Гүл` (`-гө`).

```
$ kgvoice localize audit en.json ky.json
## Suffix collisions (Kyrgyz-specific)

- `invite` — {name}ге read as **dative**
  - target: `{name}ге чакыруу жөнөттүңүз`
  - wrong for: Айбек -> *Айбекге* (want **Айбекке**); Нурлан -> *Нурланге* (want **Нурланга**); ...
  - fix: Select the suffix at runtime — kgvoice.phon.harmony.attach('GA', value) — or restructure the sentence so {name} is not suffixed.
```

The same pass also reports translation coverage, dropped/invented `{placeholder}`s, and сен/сиз register consistency across the whole catalogue. `--format json` for CI; the command exits 1 when anything is found.

## The recording pipeline

The four `bench` modules run in order, and each stage is a CLI command.

```bash
kgvoice bench select --n 100 --out manifest.json   # pick prompts
kgvoice bench prompts manifest.json --out prompts.txt
#   ... record prompts.txt to audio/<utterance_id>.wav ...
kgvoice bench qc audio/ --manifest manifest.json --spec strict
kgvoice bench wer --manifest manifest.json --hyp transcripts.jsonl
```

**Selection** scores every sentence on entity density, numbers and dates, acronyms, and mean phonetic difficulty from `phon.profile`, then picks greedily for *phoneme coverage* so 100 prompts are diverse rather than 100 variants of one hard sentence. Eight prompts already reach 82.6% of the corpus's phoneme inventory.

Selection is gated on readability first, and that gate is not cosmetic. KyrgyzNER is scraped from a news site and retains URLs, Latin transliteration, and HTML entity residue (`laquodyiykanraquo`). Those sentences score *well* on difficulty — they are full of non-alphabetic material — so without an explicit gate they dominate the ranking and the manifest is unreadable. 3.6% of the training split is excluded, with reasons:

```bash
kgvoice bench rejected --split train
```

**Manifests** carry the prompt, the gold entity spans, pronunciation hints for the two or three words per sentence a reader would otherwise get wrong, and the normalised reference tokens — so scoring later needs only the manifest, not a re-derivation of the tokenisation that might have drifted.

```
[kg-0001]
Кытайда 200 миң, Тажикстанда 82 миң 300, Түркияда 2 миң 500, Ооганстанда 2 миң кыргыз жашайт.
  pronunciation:
    Тажикстанда  ->  Та-жик-стан-да  [tɑ.ʒik.stɑn.dɑ]  (unknown stress)
    Ооганстанда  ->  Оо-ган-стан-да  [oː.ʁɑn.stɑn.dɑ]  (unknown stress)
```

**QC** measures level, peak, clipping, estimated SNR, silence lead/tail, DC offset and format against a named spec, and reports each violation with a stable code plus a message aimed at whoever recorded it — `re-record, it clipped`, not a wall of numbers. Two specs, because TTS-grade and ASR-grade are not the same bar.

**Prosody** annotation is built from categories decidable from the audio. "Sounded unnatural" is not a tag; `stress-misplaced` at word 3, syllable 1, expected 2, is. `prosody.agreement` reports exact and word-level inter-annotator agreement separately — quoting only the strict figure makes a scheme look worse than it is, and only the loose one makes it look better.

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
| `kgvoice.bench.select` | **working** | Sentence scoring, readability gate, coverage-greedy selection |
| `kgvoice.bench.manifest` | **working** | Prompts + pronunciation hints + stored references, resumable JSON |
| `kgvoice.bench.audio` | **working** | Level, peak, clipping, SNR, silence, format conformance |
| `kgvoice.bench.prosody` | **working** | Tag schema, validation, inter-annotator agreement |
| `kgvoice.localize` | **working** | Placeholder integrity, suffix/placeholder collision, сен/сиз register audit — composed in `localize.audit` |
| `kgvoice.studio` | **skeleton** | Streamlit annotation UI |

`tests/` covers harmony (20 tests), WER (21 tests), and `localize.audit` (12 tests). The other implemented modules are exercised through the CLI but do not yet have their own test files. Entity preservation (does a named entity survive translation intact?) is the one localisation failure class *not* checked — it needs gold entity spans tied to the source text, which a free-form UI-string catalogue doesn't carry; see the caveat in `kgvoice/localize/audit.py`.

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

kgvoice localize audit en.json ky.json           # markdown report, exit 1 on issues
kgvoice localize audit en.json ky.json --format json
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
- `bench` produces manifests and QC reports; it does not ship any recorded audio or annotation data itself. That data is collected per recording session (see `audio/README.md`) and is gitignored, not vendored.
- `localize` has no entity-preservation check (see [Module status](#module-status)).

## Roadmap

1. ~~`bench.manifest`~~, ~~`bench.qc`~~, ~~`bench.prosody`~~, ~~CLI wiring for `bench`~~, ~~`localize` audit~~ — **done.**
2. `studio` — a Streamlit UI over `bench.manifest` and `bench.prosody` for reviewing takes and recording annotations without hand-editing JSON.
3. Entity-preservation checking in `localize`, once there is a source of gold entity spans for arbitrary UI copy (not just corpus sentences).

## Data and licensing

Code in this repository is MIT (see `pyproject.toml`).

KyrgyzNER is published by Akyl-AI under **CC BY-NC-SA 4.0** — 1,499 news articles from 24.kg (2017–2022), annotated by 59 Kyrgyz linguists. It is downloaded on demand rather than vendored, specifically so its share-alike obligation does not propagate to this code. Attribution is written to the cache directory on download.

Corpus statistics as loaded (train split): 7,033 sentences, 89,248 tokens, 10,881 entity spans across 24 labels, 28.0% of tokens inside an entity.

## Layout

```
kgvoice/
  corpus/     KyrgyzNER loader + downloader
  phon/       alphabet, harmony, syllable, lexicon, loanword, stress, g2p, profile
  localize/   catalog, placeholders + suffix-collision, register, audit
  bench/      select, manifest, audio, prosody, wer
  studio/     skeleton
  cli.py
tests/        harmony test suite
docs/         (empty)
audio/        recordings — gitignored, see audio/README.md
```
