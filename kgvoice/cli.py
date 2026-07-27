"""Command line entry point.

Only the subcommands backed by implemented modules are wired up here:
``corpus`` and ``phon``. ``localize``, ``bench`` and ``studio`` are declared in
the parser so ``--help`` reflects the intended shape of the tool, but they exit
with a clear "not implemented" message rather than a stack trace.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from kgvoice import __version__

_NOT_IMPLEMENTED = {
    "studio": "kgvoice.studio",
}


def _corpus_words(split: str, *, min_len: int = 3) -> list[str]:
    """Unique lowercase word types from a corpus split.

    Types, not tokens: the harmony audit is a statement about the lexicon, and
    counting tokens would let a handful of frequent words dominate a cell.
    """
    from kgvoice.corpus import Corpus, ensure_corpus

    path = ensure_corpus(split)
    corpus = Corpus.from_files(path, name=f"KyrgyzNER/{split}")
    return [
        w
        for w in corpus.vocabulary(lowercase=True)
        if len(w) >= min_len and any(ch.isalpha() for ch in w)
    ]


def cmd_corpus_stats(args: argparse.Namespace) -> int:
    from kgvoice.corpus import Corpus, ensure_corpus

    path = ensure_corpus(args.split)
    corpus = Corpus.from_files(path, name=f"KyrgyzNER/{args.split}")
    print(f"# {corpus.name}  ({path})\n")
    print(corpus.stats().format(top=args.top))
    return 0


def cmd_phon_harmony_audit(args: argparse.Namespace) -> int:
    from kgvoice.phon.harmony import audit_all

    words = _corpus_words(args.split)
    print(f"# harmony audit over {len(words):,} word types from KyrgyzNER/{args.split}\n")
    for result in audit_all(words):
        print(result.format())
        print()
    return 0


def cmd_phon_paradigm(args: argparse.Namespace) -> int:
    from kgvoice.phon.harmony import paradigm

    for stem in args.stems:
        print(stem)
        for name, form in paradigm(stem).items():
            print(f"  {name:<16}{form}")
    return 0


def cmd_phon_check(args: argparse.Namespace) -> int:
    from kgvoice.phon.harmony import check_harmony

    exit_code = 0
    for word in args.words:
        violations = check_harmony(word)
        if violations:
            exit_code = 1
            print(f"{word}: " + "; ".join(str(v) for v in violations))
        else:
            print(f"{word}: harmonic")
    return exit_code


def cmd_phon_ipa(args: argparse.Namespace) -> int:
    from kgvoice.phon import g2p

    for word in args.words:
        print(f"{word}\t{g2p.transcribe(word, narrow=args.narrow)}")
    return 0


def cmd_phon_syllabify(args: argparse.Namespace) -> int:
    from kgvoice.phon import syllable

    for word in args.words:
        print(f"{word}\t{syllable.hyphenate(word)}")
    return 0


def cmd_phon_stress(args: argparse.Namespace) -> int:
    from kgvoice.phon import stress as stress_mod

    for word in args.words:
        print(f"{word}\t{stress_mod.stress(word)}")
    return 0


def cmd_phon_loanword(args: argparse.Namespace) -> int:
    from kgvoice.phon import loanword

    for word in args.words:
        print(f"{word}\t{loanword.analyze(word)}")
    return 0


def cmd_phon_profile(args: argparse.Namespace) -> int:
    from kgvoice.phon.profile import profile_word

    for word in args.words:
        p = profile_word(word)
        signals = ", ".join(f"{k} {v:+.2f}" for k, v in sorted(p.signals.items())) or "-"
        print(
            f"{p.word}\n"
            f"  ipa         {p.ipa}\n"
            f"  stress      syllable {p.stress.index} "
            f"({p.stress.rule}/{p.stress.confidence})\n"
            f"  loanword    {p.loanword.score:.2f}\n"
            f"  signals     {signals}\n"
            f"  difficulty  {p.difficulty:.2f}"
        )
    return 0


def _load_corpus(split: str):
    from kgvoice.corpus import Corpus, ensure_corpus

    return Corpus.from_files(ensure_corpus(split), name=f"KyrgyzNER/{split}")


def cmd_bench_select(args: argparse.Namespace) -> int:
    from kgvoice.bench import manifest as manifest_mod
    from kgvoice.bench import select as select_mod

    corpus = _load_corpus(args.split)
    chosen = select_mod.select(
        corpus.sentences, n=args.n, diversity=args.diversity, min_score=args.min_score
    )
    if not chosen:
        print("no sentences passed the filters", file=sys.stderr)
        return 1

    man = manifest_mod.build(chosen, name=args.name, prefix=args.prefix)
    out = man.save(args.out)

    universe = select_mod.rank(corpus.sentences)
    report = select_mod.coverage_report(chosen, universe)
    print(f"wrote {len(man)} utterances to {out}")
    print(f"  mean score        {report['mean_score']:.3f}")
    print(f"  tokens            {report['tokens']:,}")
    print(
        f"  phoneme coverage  {report['phoneme_coverage']:.1%} "
        f"({report['phonemes_covered']}/{report['phonemes_in_corpus']})"
    )
    labels = report["entity_labels"]
    print(f"  entity labels     {len(labels)} classes, {sum(labels.values())} mentions")
    for label, n in list(labels.items())[:8]:
        print(f"    {label:<16}{n}")
    return 0


def cmd_bench_prompts(args: argparse.Namespace) -> int:
    from kgvoice.bench.manifest import Manifest, write_prompts

    man = Manifest.load(args.manifest)
    out = write_prompts(man, args.out)
    print(f"wrote {len(man)} prompts to {out}")
    return 0


def cmd_bench_rejected(args: argparse.Namespace) -> int:
    """Data-quality report: which corpus sentences cannot be read aloud, and why."""
    from collections import Counter

    from kgvoice.bench.select import rejected

    corpus = _load_corpus(args.split)
    bad = rejected(corpus.sentences)
    total = len(corpus.sentences)
    print(f"{len(bad)}/{total} sentences ({len(bad) / total:.1%}) are unusable as prompts\n")
    for reason, n in Counter(r.reason for r in bad).most_common(args.top):
        print(f"  {n:>5}  {reason}")
    if args.examples:
        print("\nexamples:")
        for r in bad[: args.examples]:
            print(f"  [{r.reason}] {r.text}")
    return 0


def cmd_bench_qc(args: argparse.Namespace) -> int:
    from kgvoice.bench.audio import RecordingSpec, analyze_directory, summarize

    spec = RecordingSpec.strict() if args.spec == "strict" else RecordingSpec()
    results = analyze_directory(args.audio_dir, spec=spec)
    if not results:
        print(f"no .wav files found in {args.audio_dir}", file=sys.stderr)
        return 1

    for r in results:
        print(r.format())
    print()
    for key, value in summarize(results).items():
        print(f"{key:<20}{value}")

    if args.manifest:
        from kgvoice.bench.manifest import Manifest, attach_recordings

        man = attach_recordings(Manifest.load(args.manifest), args.audio_dir, spec_name=args.spec)
        man.save(args.manifest)
        print()
        for key, value in man.progress().items():
            print(f"{key:<20}{value}")

    return 0 if all(r.passed for r in results) else 1


def cmd_bench_wer(args: argparse.Namespace) -> int:
    """Score transcripts against a manifest, weighted toward named entities.

    Hypotheses are JSON Lines with ``utterance_id`` and ``hypothesis`` keys —
    the shape an ASR batch job naturally emits.
    """
    import json

    from kgvoice.bench.manifest import Manifest
    from kgvoice.bench.wer import WERResult, score

    man = Manifest.load(args.manifest)
    hyps: dict[str, str] = {}
    with open(args.hyp, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                hyps[rec["utterance_id"]] = rec.get("hypothesis", "")

    total = WERResult(0, 0, 0, 0, 0)
    scored = missing = 0
    for item in man.items:
        if item.utterance_id not in hyps:
            missing += 1
            continue
        if not item.ref_tokens:
            print(
                f"warning: {item.utterance_id} has no stored reference tokens; "
                "rebuild the manifest with a current kgvoice",
                file=sys.stderr,
            )
            continue
        total = total + score(item.ref_tokens, hyps[item.utterance_id], item.ref_labels)
        scored += 1

    if not scored:
        print("no utterances could be scored", file=sys.stderr)
        return 1
    print(f"# {scored} utterances scored, {missing} missing from hypotheses\n")
    print(total.format())
    return 0


def cmd_localize_audit(args: argparse.Namespace) -> int:
    """Audit a source/target UI-string catalogue pair.

    Exits 1 when the audit finds anything (missing keys, placeholder
    mismatches, suffix collisions, or mixed register) — a CI-friendly signal
    that a catalogue is not ready to ship.
    """
    from kgvoice.localize import audit_files

    report = audit_files(args.source, args.target)
    if args.format == "json":
        import json

        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(report.format())
    return 0 if report.is_clean else 1


def cmd_stub(args: argparse.Namespace) -> int:
    module = _NOT_IMPLEMENTED[args.command]
    print(
        f"'{args.command}' is not implemented yet — {module} is a documented "
        f"skeleton.\nSee the roadmap in README.md.",
        file=sys.stderr,
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kgvoice", description="Kyrgyz speech-data toolkit.")
    parser.add_argument("--version", action="version", version=f"kgvoice {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    corpus = sub.add_parser("corpus", help="KyrgyzNER corpus access")
    corpus_sub = corpus.add_subparsers(dest="subcommand", required=True)
    stats = corpus_sub.add_parser("stats", help="sentence/token/entity counts")
    stats.add_argument("--split", default="train", choices=("train", "test"))
    stats.add_argument("--top", type=int, default=30, help="labels to list")
    stats.set_defaults(func=cmd_corpus_stats)

    phon = sub.add_parser("phon", help="phonology: harmony, inventory")
    phon_sub = phon.add_subparsers(dest="subcommand", required=True)

    audit = phon_sub.add_parser(
        "harmony-audit", help="re-derive the harmony table from the corpus"
    )
    audit.add_argument("--split", default="train", choices=("train", "test"))
    audit.set_defaults(func=cmd_phon_harmony_audit)

    para = phon_sub.add_parser("paradigm", help="realise the nominal paradigm of a stem")
    para.add_argument("stems", nargs="+")
    para.set_defaults(func=cmd_phon_paradigm)

    check = phon_sub.add_parser("check", help="flag stem-internal harmony violations")
    check.add_argument("words", nargs="+")
    check.set_defaults(func=cmd_phon_check)

    ipa_p = phon_sub.add_parser("ipa", help="grapheme-to-phoneme transcription")
    ipa_p.add_argument("words", nargs="+")
    ipa_p.add_argument("--narrow", action="store_true", help="allophonic detail")
    ipa_p.set_defaults(func=cmd_phon_ipa)

    syl = phon_sub.add_parser("syllabify", help="hyphenate into syllables")
    syl.add_argument("words", nargs="+")
    syl.set_defaults(func=cmd_phon_syllabify)

    stress_p = phon_sub.add_parser("stress", help="locate word stress")
    stress_p.add_argument("words", nargs="+")
    stress_p.set_defaults(func=cmd_phon_stress)

    loan = phon_sub.add_parser("loanword", help="orthographic loanword verdict")
    loan.add_argument("words", nargs="+")
    loan.set_defaults(func=cmd_phon_loanword)

    prof = phon_sub.add_parser("profile", help="full pronunciation record + difficulty")
    prof.add_argument("words", nargs="+")
    prof.set_defaults(func=cmd_phon_profile)

    bench = sub.add_parser("bench", help="benchmark selection, audio QC, entity-weighted WER")
    bench_sub = bench.add_subparsers(dest="subcommand", required=True)

    bsel = bench_sub.add_parser("select", help="build a recording manifest from the corpus")
    bsel.add_argument("--split", default="train", choices=("train", "test"))
    bsel.add_argument("--n", type=int, default=100, help="utterances to select")
    bsel.add_argument("--out", default="manifest.json")
    bsel.add_argument("--name", default="session")
    bsel.add_argument("--prefix", default="kg", help="utterance id prefix")
    bsel.add_argument(
        "--diversity",
        type=float,
        default=0.5,
        help="0 = pure score, 1 = pure phoneme coverage",
    )
    bsel.add_argument("--min-score", type=float, default=0.0, dest="min_score")
    bsel.set_defaults(func=cmd_bench_select)

    bpr = bench_sub.add_parser("prompts", help="write a reader-facing prompt sheet")
    bpr.add_argument("manifest")
    bpr.add_argument("--out", default="prompts.txt")
    bpr.set_defaults(func=cmd_bench_prompts)

    brej = bench_sub.add_parser(
        "rejected", help="corpus sentences unusable as prompts, with reasons"
    )
    brej.add_argument("--split", default="train", choices=("train", "test"))
    brej.add_argument("--top", type=int, default=12)
    brej.add_argument("--examples", type=int, default=0)
    brej.set_defaults(func=cmd_bench_rejected)

    bqc = bench_sub.add_parser("qc", help="technical QC over a directory of WAV takes")
    bqc.add_argument("audio_dir")
    bqc.add_argument("--spec", default="default", choices=("default", "strict"))
    bqc.add_argument("--manifest", help="update this manifest with QC results")
    bqc.set_defaults(func=cmd_bench_qc)

    bwer = bench_sub.add_parser("wer", help="entity-weighted WER against a manifest")
    bwer.add_argument("--manifest", required=True)
    bwer.add_argument("--hyp", required=True, help="JSONL of {utterance_id, hypothesis}")
    bwer.set_defaults(func=cmd_bench_wer)

    localize = sub.add_parser("localize", help="EN/KY catalogue audit")
    localize_sub = localize.add_subparsers(dest="subcommand", required=True)
    laudit = localize_sub.add_parser(
        "audit", help="placeholder integrity, suffix collisions, register consistency"
    )
    laudit.add_argument("source", help="source-language catalogue (JSON)")
    laudit.add_argument("target", help="target-language catalogue (JSON)")
    laudit.add_argument("--format", default="md", choices=("md", "json"))
    laudit.set_defaults(func=cmd_localize_audit)

    for name in _NOT_IMPLEMENTED:
        stub = sub.add_parser(name, help="(not implemented yet)")
        stub.add_argument("args", nargs="*")
        stub.set_defaults(func=cmd_stub)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as exc:  # network/corpus failures carry actionable text
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
