"""EN/KY localisation evaluation.  *Not yet implemented — skeleton only.*

The problem this module is meant to address: machine and human translations of
UI strings into Kyrgyz fail in ways a BLEU-style metric does not see. The three
failure classes worth scoring separately are

1. **Entity corruption.** A named entity that must survive translation intact
   gets declined, transliterated inconsistently, or dropped. :mod:`kgvoice.corpus`
   supplies gold entity spans to check against.
2. **Suffix/placeholder collision.** Kyrgyz is agglutinative, so a suffix on a
   ``{placeholder}`` cannot be hardcoded — its vowel is chosen by the *runtime*
   value's last vowel. :func:`kgvoice.phon.harmony.harmonize` is the check.
3. **Register drift.** сен/сиз consistency across a string catalogue.

Planned surface::

    kgvoice localize audit  en.json ky.json
    kgvoice localize report en.json ky.json --format md
"""

__all__: list[str] = []
