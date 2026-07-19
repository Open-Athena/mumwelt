"""Expand a single hit into its surrounding context, using only the cached corpus.

A lone Discord message or GitHub comment is often a fragment. Because the corpus holds
*every* message/comment as its own chunk, we can reconstruct context from the corpus
itself — no extra calls to marinmirror:

  - discord  → the ±N-message window in the same channel, by timestamp
  - github   → the issue/PR plus its comment thread
  - else     → the chunk itself (narrative section, W&B run — use ``mum run`` for full config)
"""
from __future__ import annotations

import sqlite3

from . import config

WINDOW = 12


# Columns never worth returning to a caller: the embedding is a BLOB (not JSON-
# serializable), and `syms` is a derived FTS payload — the split words of text's own
# camelCase identifiers, so echoing it back is pure redundancy in every `--json` result.
_HIDDEN_COLS = {"syms"}


def _row(r) -> dict:
    """A chunk row as a plain dict, minus the embedding BLOB and derived search columns."""
    return {k: v for k, v in dict(r).items()
            if k not in _HIDDEN_COLS and not isinstance(v, (bytes, bytearray))}


def _has_parts(con) -> bool:
    """Whether this index carries the part/n_parts columns.

    A client can be newer than its cached corpus-index.db (the index is downloaded,
    not shipped with the code), so every part-aware query degrades gracefully.
    """
    return any(r[1] == "part" for r in con.execute("PRAGMA table_info(chunks)"))


def base_ref(ref: str | None) -> str:
    """Strip the part suffix from a chunk ref: ``"5596:p3"`` → ``"5596"``.

    A long document is split into parts whose refs are ``<id>:p<i>`` (the embed cache
    is keyed on ref, so parts need distinct ones). Anything that reasons about the
    *document* — thread expansion, dedupe keys — must work from the base.
    """
    return (ref or "").split(":p")[0]


def _find(con, target: str):
    """Resolve a target (canonical url, or a bare ref id) to a chunk row.

    Falls back to the first part of a split document, so ``mum context 5596`` still
    resolves when #5596 is stored as ``5596:p0``, ``5596:p1``, ….
    """
    order = "ORDER BY part " if _has_parts(con) else ""
    row = con.execute(f"SELECT * FROM chunks WHERE url = ? {order}LIMIT 1",
                      (target,)).fetchone()
    if row:
        return row
    row = con.execute("SELECT * FROM chunks WHERE ref = ? LIMIT 1", (target,)).fetchone()
    if row:
        return row
    return con.execute(
        f"SELECT * FROM chunks WHERE ref LIKE ? {order}LIMIT 1",
        (f"{target}:p%",)).fetchone()


def show(target: str, window: int = WINDOW) -> dict | None:
    """Return ``{focal, context: [chunk, …], kind}`` or None if the target isn't found."""
    if not config.CORPUS.exists():
        return None
    con = sqlite3.connect(config.CORPUS)
    con.row_factory = sqlite3.Row
    try:
        focal = _find(con, target)
        if not focal:
            return None
        src, kind = focal["source"], focal["kind"]
        ctx: dict[int, sqlite3.Row] = {focal["id"]: focal}

        if src == "discord":
            chan, ts = focal["parent"], focal["date"]
            for r in con.execute(
                "SELECT * FROM chunks WHERE source='discord' AND parent=? AND date<=? "
                "ORDER BY date DESC LIMIT ?", (chan, ts, window + 1)):
                ctx[r["id"]] = r
            for r in con.execute(
                "SELECT * FROM chunks WHERE source='discord' AND parent=? AND date>? "
                "ORDER BY date ASC LIMIT ?", (chan, ts, window)):
                ctx[r["id"]] = r
        elif src == "github":
            # the enclosing issue/PR number (the chunk's own ref, or its parent),
            # with any part suffix stripped so every part of the body and of each
            # comment comes back as one thread
            issue = base_ref(focal["parent"] or focal["ref"])
            for r in con.execute(
                "SELECT * FROM chunks WHERE source='github' "
                "AND (ref=? OR ref LIKE ? OR parent=?) ORDER BY date",
                (issue, f"{issue}:p%", issue)):
                ctx[r["id"]] = r

        # date, then part — parts of one body share a date and must stay in order
        has_parts = _has_parts(con)
        rows = sorted(ctx.values(),
                      key=lambda r: ((r["date"] or ""), r["part"] if has_parts else 0))
        return {"focal": _row(focal), "kind": src, "context": [_row(r) for r in rows]}
    finally:
        con.close()
