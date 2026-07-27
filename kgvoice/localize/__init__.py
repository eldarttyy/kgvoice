"""EN/KY localisation evaluation.

Machine and human translations of UI strings into Kyrgyz fail in ways a
BLEU-style metric does not see. Three failure classes are checked here:

1. **Placeholder integrity** — a dropped or invented ``{placeholder}``.
   :mod:`kgvoice.localize.placeholders`.
2. **Suffix/placeholder collision.** Kyrgyz is agglutinative, so a case suffix
   glued directly onto a ``{placeholder}`` is correct for one shape of runtime
   value and wrong for the rest. Same module, built on
   :mod:`kgvoice.phon.harmony`.
3. **Register drift.** сен/сиз consistency across a string catalogue.
   :mod:`kgvoice.localize.register`.

:func:`kgvoice.localize.audit.audit` runs all three over a catalogue pair and
returns one report.

**Not implemented:** entity preservation — does a named entity survive
translation intact? That needs gold entity spans tied to the source text,
which a free-form UI-string catalogue does not carry. See the caveat in
:mod:`kgvoice.localize.audit`.
"""

from kgvoice.localize.audit import LocalizationAudit, PlaceholderIssue, SuffixIssue, audit, audit_files
from kgvoice.localize.catalog import Catalog, CatalogPair
from kgvoice.localize.register import RegisterReport, RegisterVerdict, audit_catalog, detect, register_of

__all__ = [
    "audit",
    "audit_files",
    "LocalizationAudit",
    "PlaceholderIssue",
    "SuffixIssue",
    "Catalog",
    "CatalogPair",
    "detect",
    "register_of",
    "audit_catalog",
    "RegisterVerdict",
    "RegisterReport",
]
