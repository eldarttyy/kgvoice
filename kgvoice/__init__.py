"""kgvoice — a Kyrgyz speech-data toolkit.

Four modules over one shared corpus core:

* ``kgvoice.corpus``   — KyrgyzNER loader, entity spans, corpus statistics.
* ``kgvoice.phon``     — grapheme-to-phoneme, IPA, syllabification, stress, vowel harmony.
* ``kgvoice.localize`` — EN/KY entity-preservation evaluation for translated text.
* ``kgvoice.bench``    — recording manifests, audio QC, entity-weighted WER, prosody annotation.

The design goal throughout is that every linguistic claim the toolkit makes is
checkable against the corpus rather than asserted. See ``kgvoice.phon.harmony``
and the ``kgvoice phon harmony-audit`` command.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
