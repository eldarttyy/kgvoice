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
    "localize": "kgvoice.localize",
    "bench": "kgvoice.bench",
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
