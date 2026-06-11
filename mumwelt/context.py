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


def _find(con, target: str):
    """Resolve a target (canonical url, or a bare ref id) to a chunk row."""
    row = con.execute("SELECT * FROM chunks WHERE url = ? LIMIT 1", (target,)).fetchone()
    if row:
        return row
    return con.execute("SELECT * FROM chunks WHERE ref = ? LIMIT 1", (target,)).fetchone()


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
            # the enclosing issue/PR number (the chunk's own ref, or its parent)
            issue = focal["parent"] or focal["ref"]
            for r in con.execute(
                "SELECT * FROM chunks WHERE source='github' AND (ref=? OR parent=?) "
                "ORDER BY date", (issue, issue)):
                ctx[r["id"]] = r

        rows = sorted(ctx.values(), key=lambda r: (r["date"] or ""))
        return {"focal": dict(focal), "kind": src, "context": [dict(r) for r in rows]}
    finally:
        con.close()
