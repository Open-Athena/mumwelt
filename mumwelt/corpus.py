"""Local hybrid search over the cached ``corpus-index.db``.

Fuses FTS5 keyword search with semantic search (cosine over the stored 384-d float32
embeddings) via reciprocal-rank fusion — the same recipe the index was built for. FTS
wins on identifiers (run names, #5596, a login); vectors win on natural-language intent;
the corpus mixes both, so we fuse. Every hit carries a canonical ``url``.
"""
from __future__ import annotations

import os
import re
import sqlite3

from . import config
from .identifiers import expand_query

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def exists() -> bool:
    return config.CORPUS.exists()


def meta() -> dict:
    con = sqlite3.connect(config.CORPUS)
    try:
        return dict(con.execute("SELECT key,value FROM meta"))
    finally:
        con.close()


def _literal_tokens(q: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]{2,}", q)


def _fts_query(q: str) -> str:
    """The precise MATCH expression: the query's literal tokens, as before."""
    return " OR ".join(f'"{t}"' for t in _literal_tokens(q))


def _fts_query_expanded(q: str) -> str:
    """The high-recall MATCH expression: literal tokens plus split camelCase words.

    FTS5's ``unicode61`` tokenizer treats ``_`` as a separator, so snake_case already
    matches both whole and in parts. It does nothing with capitalization, so
    ``ExecutorStep`` indexes as the single token ``executorstep`` — a query for "executor
    step" could never reach it, and a query for ``ExecutorStep`` could never reach a
    snake_case ``executor_step``. The corpus side puts the same expansion in the ``syms``
    column; the two halves only work together.

    This expression is deliberately NOT used on its own — see ``_fts``.
    """
    toks = expand_query(q) or _literal_tokens(q)
    return " OR ".join(f'"{t}"' for t in toks)


def _filters(source, kind, since, until) -> tuple[str, list]:
    where, params = ["1=1"], []
    if source:
        where.append("source IN (%s)" % ",".join("?" * len(source)))
        params += source
    if kind:
        where.append("kind IN (%s)" % ",".join("?" * len(kind)))
        params += kind
    if since:
        where.append("substr(date,1,10) >= ?")
        params.append(since)
    if until:
        where.append("substr(date,1,10) <= ?")
        params.append(until)
    return " AND ".join(where), params


def _fts_cols(con) -> int:
    """Number of columns in ``chunks_fts`` (2 pre-``syms``, 3 after).

    mumwelt downloads corpus-index.db rather than shipping it, so a new client routinely
    queries an older index. bm25() rejects a weight list longer than the table's column
    count, so the weights have to be sized to whatever index is actually on disk.
    """
    try:
        row = con.execute("SELECT * FROM chunks_fts LIMIT 0")
        return len([d[0] for d in row.description])
    except sqlite3.OperationalError:
        return 2


def _fts_rank(con, fq, flt, params, limit) -> list[int]:
    """Run one MATCH expression, ranked by bm25 with per-column weights.

    ``text`` and ``title`` keep bm25's default weight of 1.0, so every pre-existing
    source ranks exactly as it did before ``syms`` existed — this change is additive by
    construction. Only the new column is weighted, and it is weighted *down*: a document
    that matched only because identifier splitting reached it is a weaker hit than one
    whose real text matched, and must never outrank one.

    NB (measured, deliberately NOT applied): raising the *title* weight sharply improves
    code lookups — for "ServeSpec" it moves the defining chunk from off the first page to
    rank 1, and for "LoRAConfig" from rank 5 to rank 2 — because a code chunk's title is
    its qualified symbol name. But it re-ranks github/discord/wandb too, which is exactly
    what the evals/ harness exists to measure. Run it before touching these.
    """
    weights = {3: "chunks_fts, 1.0, 1.0, 0.2", 2: "chunks_fts, 1.0, 1.0"}[_fts_cols(con)]
    sql = ("SELECT c.id FROM chunks_fts f JOIN chunks c ON c.id=f.rowid "
           f"WHERE chunks_fts MATCH ? AND {flt} ORDER BY bm25({weights}) LIMIT ?")
    try:
        return [r[0] for r in con.execute(sql, [fq, *params, limit])]
    except sqlite3.OperationalError:
        return []


def _fts(con, q, flt, params, limit) -> list[int]:
    """Keyword leg: exact hits first, identifier-expanded hits strictly below them.

    Flat-OR-ing the literal token together with its split words destroys precision —
    measured on the code corpus, a query for ``ExecutorStep`` ranked six unrelated
    ``_override_tracker``/``StepSpec`` chunks above the actual ``ExecutorStep`` class,
    because a document matching the common fragments "executor" and "step" many times
    outscores one matching the exact identifier once.

    So the two are run as separate tiers and concatenated rather than unioned. The caller
    (``_rrf``) scores by *rank position*, so appending expansion-only hits after the exact
    ones can add recall but can never demote an exact hit. Precision is therefore
    identical to the pre-expansion behavior by construction, not by tuning.
    """
    exact = _fts_query(q)
    if not exact:
        return []
    out = _fts_rank(con, exact, flt, params, limit)
    expanded = _fts_query_expanded(q)
    if expanded != exact and len(out) < limit:
        seen = set(out)
        out += [i for i in _fts_rank(con, expanded, flt, params, limit)
                if i not in seen][:limit - len(out)]
    return out


_MODEL = None


def _model():
    """Load the query embedder once and reuse it (≈1-2 s to construct)."""
    global _MODEL
    if _MODEL is None:
        from fastembed import TextEmbedding
        _MODEL = TextEmbedding(config.EMBED_MODEL)
    return _MODEL


def _query_vec(texts):
    """Embed one or more query texts → a single unit query vector.

    With one text this is the plain query embedding. With several (HyDE: the literal
    query plus N hypothetical answer docs), each is L2-normalized then mean-pooled, so
    every doc weighs equally regardless of length and per-doc noise partially cancels.
    """
    import numpy as np

    vs = []
    for v in _model().embed(list(texts)):
        v = np.asarray(v, dtype=np.float32)
        vs.append(v / (np.linalg.norm(v) + 1e-9))
    qv = np.mean(vs, axis=0)
    return qv / (np.linalg.norm(qv) + 1e-9)


def _vec(con, qv, flt, params, limit) -> list[int]:
    import numpy as np

    rows = con.execute(
        f"SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL AND {flt}",
        params).fetchall()
    if not rows:
        return []
    ids = np.array([r[0] for r in rows])
    mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)

    norms = np.linalg.norm(mat, axis=1) + 1e-9
    sims = (mat @ qv) / norms
    top = np.argpartition(-sims, min(limit, len(sims) - 1))[:limit]
    top = top[np.argsort(-sims[top])]
    return [int(ids[i]) for i in top]


def _rrf(*rankings) -> tuple[list[int], dict]:
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            score[cid] = score.get(cid, 0.0) + 1.0 / (config.RRF_K + rank + 1)
    return sorted(score, key=lambda c: -score[c]), score


def search(query: str, k: int = 10, source=None, kind=None, since=None, until=None,
           fts_only: bool = False, vec_only: bool = False, vec_text=None) -> list[dict]:
    """Return up to ``k`` fused hits as dicts (source, kind, ref, url, title, snippet, …).

    ``vec_text`` (HyDE): one or more hypothetical-answer docs to drive the *vector* leg
    instead of the literal query; their embeddings are mean-pooled (N-doc averaging). The
    FTS leg always uses the literal ``query`` so exact identifiers (#5596, run names) are
    never diluted. ``None`` → classic behavior (vector leg embeds the query itself).
    """
    con = sqlite3.connect(config.CORPUS)
    try:
        flt, params = _filters(source, kind, since, until)
        pool = max(k * 4, 40)
        fts = [] if vec_only else _fts(con, query, flt, params, pool)
        vec = []
        if not fts_only:
            vec = _vec(con, _query_vec(vec_text or [query]), flt, params, pool)
        order, score = _rrf(vec, fts)
        fset, vset = set(fts), set(vec)
        out = []
        # A long document is indexed as several parts that share one url. Collapse to
        # one hit per url, keeping the best-scoring part — `order` is already sorted by
        # score, so the first sighting wins. Max, not sum: summing would rank a document
        # highly just for being long enough to split, which is exactly the flooding this
        # is meant to prevent. Dedupe before truncating to k, or a single chatty issue
        # can occupy most of the result slots.
        seen: set[str] = set()
        for cid in order:
            if len(out) >= k:
                break
            (s, kd, ref, parent, title, author, date, url, text) = con.execute(
                "SELECT source,kind,ref,parent,title,author,date,url,text "
                "FROM chunks WHERE id=?", (cid,)).fetchone()
            if url in seen:
                continue
            seen.add(url)
            snippet = re.sub(r"\s+", " ", (text or "")).strip()[:240]
            via = "+".join(m for m, hit in (("vec", cid in vset), ("fts", cid in fset)) if hit)
            out.append(dict(
                source=s, kind=kd, ref=ref, parent=parent, title=title, author=author,
                date=(date or "")[:10], url=url, snippet=snippet,
                score=round(score[cid], 4), via=via))
        return out
    finally:
        con.close()
