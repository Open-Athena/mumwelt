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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def exists() -> bool:
    return config.CORPUS.exists()


def meta() -> dict:
    con = sqlite3.connect(config.CORPUS)
    try:
        return dict(con.execute("SELECT key,value FROM meta"))
    finally:
        con.close()


def _fts_query(q: str) -> str:
    toks = re.findall(r"[A-Za-z0-9_]{2,}", q)
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


def _fts(con, q, flt, params, limit) -> list[int]:
    fq = _fts_query(q)
    if not fq:
        return []
    sql = ("SELECT c.id FROM chunks_fts f JOIN chunks c ON c.id=f.rowid "
           f"WHERE chunks_fts MATCH ? AND {flt} ORDER BY bm25(chunks_fts) LIMIT ?")
    try:
        return [r[0] for r in con.execute(sql, [fq, *params, limit])]
    except sqlite3.OperationalError:
        return []


_MODEL = None


def _model():
    """Load the query embedder once and reuse it (≈1-2 s to construct)."""
    global _MODEL
    if _MODEL is None:
        from fastembed import TextEmbedding
        _MODEL = TextEmbedding(config.EMBED_MODEL)
    return _MODEL


def _vec(con, q, flt, params, limit) -> list[int]:
    import numpy as np

    rows = con.execute(
        f"SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL AND {flt}",
        params).fetchall()
    if not rows:
        return []
    ids = np.array([r[0] for r in rows])
    mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)

    qv = np.asarray(next(iter(_model().embed([q]))), dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1) + 1e-9
    qv = qv / (np.linalg.norm(qv) + 1e-9)
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
           fts_only: bool = False, vec_only: bool = False) -> list[dict]:
    """Return up to ``k`` fused hits as dicts (source, kind, ref, url, title, snippet, …)."""
    con = sqlite3.connect(config.CORPUS)
    try:
        flt, params = _filters(source, kind, since, until)
        pool = max(k * 4, 40)
        fts = [] if vec_only else _fts(con, query, flt, params, pool)
        vec = [] if fts_only else _vec(con, query, flt, params, pool)
        order, score = _rrf(vec, fts)
        fset, vset = set(fts), set(vec)
        out = []
        for cid in order[:k]:
            (s, kd, ref, parent, title, author, date, url, text) = con.execute(
                "SELECT source,kind,ref,parent,title,author,date,url,text "
                "FROM chunks WHERE id=?", (cid,)).fetchone()
            snippet = re.sub(r"\s+", " ", (text or "")).strip()[:240]
            via = "+".join(m for m, hit in (("vec", cid in vset), ("fts", cid in fset)) if hit)
            out.append(dict(
                source=s, kind=kd, ref=ref, parent=parent, title=title, author=author,
                date=(date or "")[:10], url=url, snippet=snippet,
                score=round(score[cid], 4), via=via))
        return out
    finally:
        con.close()
