"""Identifier splitting — the one thing FTS5's tokenizer does not do for code.

`chunks_fts` uses FTS5's default ``unicode61`` tokenizer, which treats ``_`` as a
separator (Unicode class Pc) but knows nothing about capitalization. Measured against
a real FTS5 table, indexing ``default_lm_config(cfg: ExecutorStep)`` gives:

    default             HIT      executor            miss
    lm                  HIT      step                miss
    config              HIT      executorstep        HIT
    default_lm_config   HIT      httpserver          HIT

So snake_case is already handled on both the index and the query side, and the *only*
gap is camelCase: a query for "executor step" can never reach ``ExecutorStep``, and a
query for ``ExecutorStep`` can never reach a snake_case ``executor_step``.

Closing it takes an expansion on **both** sides, which is why this module is duplicated
(deliberately, ~40 lines) from marinmirror's ``marinmirror/identifiers.py``:

* index side — marinmirror's ``expand()`` fills ``chunks.syms``, a third FTS column
* query side — ``expand_query()`` here expands query tokens the same way

The two copies MUST agree, and the client is the copy that cannot be fixed by a rebuild:
mumwelt *downloads* corpus-index.db rather than shipping with it, so a client can be
older or newer than the index it is querying. A divergence shows up as a silent recall
drop rather than an error. ``expand()`` is kept here even though only the server calls
it, so the two files stay byte-comparable.
"""
from __future__ import annotations

import re

# Split an identifier into words. The alternation order matters:
#   [A-Z]+(?=[A-Z][a-z])  — an acronym run that ends where a CapWord starts: HTTPServer → HTTP
#   [A-Z]?[a-z0-9]+       — an ordinary CapWord or lowercase word: Executor, config, v2
#   [A-Z]+                — a trailing all-caps run: parseURL → URL
#   \d+                   — a bare number run
# Anything not matched (``_``, ``.``, punctuation) is skipped, so this handles
# snake_case, camelCase, PascalCase, SCREAMING_SNAKE and dotted paths uniformly.
_WORDS = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+|\d+")

# An identifier worth expanding: contains a camel hump. Pure snake_case and single
# words are already covered by unicode61, so expanding them would only bloat the index.
_CAMEL = re.compile(r"[a-z0-9][A-Z]|[A-Z]{2,}[a-z]")

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def split_identifier(name: str) -> list[str]:
    """``"ExecutorStep"`` → ``["executor", "step"]``; ``"default_lm_config"`` → 3 words.

    Returns lowercase words of length >= 2. One-character fragments are dropped: they
    are almost always noise (the ``x`` in ``xShape``) and FTS5 would match them against
    a huge portion of the corpus.
    """
    return [w.lower() for w in _WORDS.findall(name or "") if len(w) >= 2]


def is_camel(name: str) -> bool:
    """Whether ``name`` has a camel hump, i.e. whether unicode61 will under-tokenize it."""
    return bool(_CAMEL.search(name or ""))


def expand(text: str, *, cap: int = 4000) -> str:
    """Build the ``syms`` FTS payload for ``text``: the split words of its camelCase ids.

    Only camel identifiers contribute (see ``is_camel``) — snake_case and plain words are
    already reachable, so re-emitting them would inflate every document's FTS length and
    depress bm25 scores across the board for no recall gain.

    Deduped and order-preserving, so the payload stays stable across rebuilds (it feeds a
    content hash upstream). ``cap`` bounds a pathological generated file; the words are
    emitted most-informative-first only in the weak sense of source order, so truncation
    loses tail vocabulary rather than the symbol's own name, which appears early.
    """
    out: list[str] = []
    seen: set[str] = set()
    for ident in _IDENT.findall(text or ""):
        if not is_camel(ident):
            continue
        for word in split_identifier(ident):
            if word not in seen:
                seen.add(word)
                out.append(word)
    joined = " ".join(out)
    return joined[:cap].rsplit(" ", 1)[0] if len(joined) > cap else joined


def expand_query(query: str) -> list[str]:
    """Query-side counterpart: the FTS terms a user's query should match on.

    Returns the literal tokens *plus* the split words of any camel token, so
    ``"ExecutorStep"`` reaches both ``executorstep`` (in ``text``) and ``executor step``
    (in ``syms``). The literal token is always kept and always first — an exact
    identifier match must never be diluted by its own fragments.
    """
    out: list[str] = []
    seen: set[str] = set()
    for tok in _IDENT.findall(query or ""):
        for term in (tok, *(split_identifier(tok) if is_camel(tok) else ())):
            low = term.lower()
            if len(low) >= 2 and low not in seen:
                seen.add(low)
                out.append(term)
    return out
