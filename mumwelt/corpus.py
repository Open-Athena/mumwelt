"""Local hybrid search over the cached ``corpus-index.db``.

Fuses FTS5 keyword search with semantic search (cosine over the stored float32
embeddings) via reciprocal-rank fusion — the same recipe the index was built for. FTS
wins on identifiers (run names, #5596, a login); vectors win on natural-language intent;
the corpus mixes both, so we fuse. Every hit carries a canonical ``url``.

The corpus may carry more than one vector space (prose vs code, different encoders and
different widths — see ``meta.spaces``). Each space is scored separately and RRF fuses
the rankings; because RRF scores by rank position, cosine scales that are not comparable
across models combine without calibration.

**The index is authoritative about which model produced its vectors, not this client.**
Every model name is read from ``meta.spaces`` at query time. Hardcoding it here is how
you get a client that embeds with one model and scores against another's vectors — which
does not fail, it just silently returns plausible nonsense.
"""
from __future__ import annotations

import functools
import json
import os
import re
import sqlite3
import sys

from . import config
from .identifiers import expand_query

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_SPACE = "prose"


def exists() -> bool:
    return config.CORPUS.exists()


def meta() -> dict:
    con = sqlite3.connect(config.CORPUS)
    try:
        return dict(con.execute("SELECT key,value FROM meta"))
    finally:
        con.close()


def spaces(m: dict | None = None) -> dict[str, dict]:
    """``name -> {model, dim, sources}`` for the local corpus, read off the artifact.

    Falls back to the legacy single-space ``meta.model``/``meta.dim``, and finally to the
    client default, so an older index (or one built before spaces existed) still works.
    """
    m = meta() if m is None else m
    raw = m.get("spaces")
    if raw:
        try:
            parsed = json.loads(raw)
            if parsed:
                return parsed
        except ValueError:
            pass
    return {DEFAULT_SPACE: {"model": m.get("model") or config.EMBED_MODEL,
                            "dim": int(m.get("dim") or config.EMBED_DIM),
                            "sources": []}}


@functools.lru_cache(maxsize=1)
def _supported_models() -> frozenset[str] | None:
    """Model names the installed fastembed can load — ``None`` if fastembed is absent.

    A cheap pre-flight: it lists supported models rather than constructing one, so no
    weights are downloaded. That lets a search command flag a missing encoder *before*
    running a query that would otherwise fall back to keyword-only for that space with
    nothing in the results to say the semantic leg never ran.
    """
    try:
        from fastembed import TextEmbedding
    except Exception:                            # noqa: BLE001 — any import failure ⇒ absent
        return None
    try:
        return frozenset(m["model"] for m in TextEmbedding.list_supported_models())
    except Exception:                            # noqa: BLE001 — advisory only
        return frozenset()


def install_hint() -> str:
    """The exact command that fixes a missing encoder, tailored to what is wrong."""
    if _supported_models() is None:
        return "pip install -U fastembed   # fastembed is not installed"
    return "pip install -U fastembed mumwelt   # upgrade fastembed to one that ships this model"


def unavailable_spaces(source=None, exclude_source=None) -> list[tuple[str, str]]:
    """``(space, model)`` pairs the corpus needs but this client cannot encode for.

    Honors the same source filter the search itself uses, so it reports only the spaces a
    given query would actually touch — an unfiltered prose search never flags the code
    encoder, and ``--source code`` never flags the prose one. Empty when everything loads.
    """
    supported = _supported_models()
    out: list[tuple[str, str]] = []
    for name, spec in sorted(spaces().items()):
        srcs = spec.get("sources") or []
        if source and srcs and set(srcs).isdisjoint(source):
            continue
        if exclude_source and srcs and set(srcs).issubset(exclude_source):
            continue
        model = spec.get("model") or config.EMBED_MODEL
        if supported is None or model not in supported:
            out.append((name, model))
    return out


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


def _filters(source, kind, since, until, exclude_source=None,
             exclude_kind=None) -> tuple[str, list]:
    where, params = ["1=1"], []
    if source:
        where.append("source IN (%s)" % ",".join("?" * len(source)))
        params += source
    if exclude_source:
        where.append("source NOT IN (%s)" % ",".join("?" * len(exclude_source)))
        params += list(exclude_source)
    if kind:
        where.append("kind IN (%s)" % ",".join("?" * len(kind)))
        params += kind
    if exclude_kind:
        where.append("kind NOT IN (%s)" % ",".join("?" * len(exclude_kind)))
        params += list(exclude_kind)
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


_MODELS: dict[str, object] = {}
_MODEL_WARNED: set[str] = set()


def _model(name: str):
    """Load a query embedder by model name, once each. ``None`` if unavailable.

    Returns None rather than raising so an index carrying a space this client cannot
    embed for degrades to "that space contributes nothing" instead of taking the whole
    search down — the FTS leg and every other space still answer. The warning is emitted
    once per model so a scripted caller is not spammed.
    """
    if name in _MODELS:
        return _MODELS[name]
    try:
        from fastembed import TextEmbedding
        _MODELS[name] = TextEmbedding(name)
    except Exception as e:                       # noqa: BLE001 — any load failure
        if name not in _MODEL_WARNED:
            _MODEL_WARNED.add(name)
            print(f"  ⚠ corpus was built with embedding model {name!r}, which this "
                  f"client cannot load ({type(e).__name__}: {e}). That vector space is "
                  f"being skipped — results fall back to keyword search for it.\n"
                  f"    Fix: {install_hint()}", file=sys.stderr)
        _MODELS[name] = None
    return _MODELS[name]


def warm(source=None, exclude_source=None) -> None:
    """Preload every encoder this corpus needs, before a threaded fan-out.

    Left lazy, N worker threads would each construct the same encoder concurrently —
    wasteful at ~1.3GB for the code model, and fastembed's constructor is not something
    to race. Honors the same source filter as the search itself, so a code-only fan-out
    never pays for the prose encoder.
    """
    for spec in spaces().values():
        srcs = spec.get("sources") or []
        if source and srcs and set(srcs).isdisjoint(source):
            continue
        if exclude_source and srcs and set(srcs).issubset(exclude_source):
            continue
        _model(spec.get("model") or config.EMBED_MODEL)


def _query_vec(texts, model_name: str):
    """Embed one or more query texts with ``model_name`` → a single unit query vector.

    With one text this is the plain query embedding. With several (HyDE: the literal
    query plus N hypothetical answer docs), each is L2-normalized then mean-pooled, so
    every doc weighs equally regardless of length and per-doc noise partially cancels.
    """
    import numpy as np

    model = _model(model_name)
    if model is None:
        return None
    vs = []
    for v in model.embed(list(texts)):
        v = np.asarray(v, dtype=np.float32)
        vs.append(v / (np.linalg.norm(v) + 1e-9))
    qv = np.mean(vs, axis=0)
    return qv / (np.linalg.norm(qv) + 1e-9)


def _has_space_col(con) -> bool:
    """Whether this index tags rows with ``embed_space`` (older ones do not)."""
    try:
        con.execute("SELECT embed_space FROM chunks LIMIT 0")
        return True
    except sqlite3.OperationalError:
        return False


def code_sources() -> list[str]:
    """Sources that live in the code vector space, per the corpus's own metadata."""
    sp = spaces().get("code") or {}
    return list(sp.get("sources") or ["code"])


def _vec(con, texts, flt, params, limit, source=None, exclude_source=None) -> list[list[int]]:
    """One cosine ranking per vector space — never one matrix over all of them.

    Vector width is per space, so BLOBs from different spaces cannot be stacked: mixing
    384-d and 768-d rows either throws on reshape or, if the counts happen to divide
    evenly, silently produces a garbage matrix. Filtering by ``embed_space`` first is
    what makes that impossible rather than merely unlikely.
    """
    import numpy as np

    all_spaces = spaces()
    tagged = _has_space_col(con)
    out: list[list[int]] = []
    for name, spec in sorted(all_spaces.items()):
        # Skip a space the source filter has already excluded — that is what keeps
        # `--source code` from loading the prose encoder and vice versa.
        srcs = spec.get("sources") or []
        if source and srcs and set(srcs).isdisjoint(source):
            continue
        # A space whose every source is excluded contributes nothing — skip it before
        # loading its encoder, which is the difference between a prose-only query paying
        # for the 1.3GB code model and not.
        if exclude_source and srcs and set(srcs).issubset(exclude_source):
            continue
        if tagged:
            where, p = f"embed_space = ? AND {flt}", [name, *params]
        else:
            where, p = flt, list(params)
        rows = con.execute(
            f"SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL AND {where}",
            p).fetchall()
        if not rows:
            continue
        qv = _query_vec(texts, spec.get("model") or config.EMBED_MODEL)
        if qv is None:
            continue
        ids = np.array([r[0] for r in rows])
        mat = np.frombuffer(b"".join(r[1] for r in rows),
                            dtype=np.float32).reshape(len(rows), -1)
        if mat.shape[1] != qv.shape[0]:
            # The index and the model it names disagree on width. Skipping keeps the
            # other spaces usable; the message names both sides so it is diagnosable.
            print(f"  ⚠ space {name!r}: index vectors are {mat.shape[1]}-d but "
                  f"{spec.get('model')!r} produced {qv.shape[0]}-d — skipping this space.",
                  file=sys.stderr)
            continue
        norms = np.linalg.norm(mat, axis=1) + 1e-9
        sims = (mat @ qv) / norms
        top = np.argpartition(-sims, min(limit, len(sims) - 1))[:limit]
        top = top[np.argsort(-sims[top])]
        out.append([int(ids[i]) for i in top])
    return out


def _rrf(*rankings) -> tuple[list[int], dict]:
    """Reciprocal-rank fusion over the FTS ranking plus one ranking per vector space.

    NB a chunk belongs to exactly one space, so nothing is double-counted — but each
    space contributes its own rank-1, so the top of every space scores 1/(K+1) where a
    single vector leg previously had one rank-1 overall. That structurally lifts the
    smaller space. Left unweighted deliberately, matching the server: the correction is a
    measured per-space weight and there are no code-specific golds to measure against yet.
    """
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            score[cid] = score.get(cid, 0.0) + 1.0 / (config.RRF_K + rank + 1)
    return sorted(score, key=lambda c: -score[c]), score


def search(query: str, k: int = 10, source=None, kind=None, since=None, until=None,
           fts_only: bool = False, vec_only: bool = False, vec_text=None,
           with_code: bool = False, exclude_kind=None) -> list[dict]:
    """Return up to ``k`` fused hits as dicts (source, kind, ref, url, title, snippet, …).

    **Code is excluded unless you ask for it.** With no explicit ``source``, the code
    source does not compete for these ``k`` slots; use ``search_code`` (or
    ``search_split``) for the code lane. This is not a preference — fusing them is
    zero-sum, and measured on the 9-question harness, code in the fused ranking cost
    1.9pp of gold recall at both @10 and @20 while adding nothing at @50. It also cannot
    *gain* anything on that metric, since the metric counts only issue/PR citations, so
    the honest reading is: mixing costs prose and cannot be credited for code. Separating
    them lets an agent judge code on its own terms, which is the one thing a cosine
    threshold cannot do.

    ``source=["code"]`` (explicit) makes code primary and gets the full ``k`` — that is
    the code lane, and it is unaffected by any of the above. ``with_code=True`` opts a
    caller back into the old fused behavior.

    ``vec_text`` (HyDE): one or more hypothetical-answer docs to drive the *vector* leg
    instead of the literal query; their embeddings are mean-pooled (N-doc averaging). The
    FTS leg always uses the literal ``query`` so exact identifiers (#5596, run names) are
    never diluted. ``None`` → classic behavior (vector leg embeds the query itself).
    """
    excl = None if (source or with_code) else code_sources()
    con = sqlite3.connect(config.CORPUS)
    try:
        flt, params = _filters(source, kind, since, until, exclude_source=excl,
                               exclude_kind=exclude_kind)
        pool = max(k * 4, 40)
        fts = [] if vec_only else _fts(con, query, flt, params, pool)
        vecs: list[list[int]] = []
        if not fts_only:
            vecs = _vec(con, vec_text or [query], flt, params, pool, source,
                        exclude_source=excl)
        order, score = _rrf(*vecs, fts)
        fset = set(fts)
        vset = set().union(*vecs) if vecs else set()
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


BRANCH_KIND = "branch-symbol"


def search_code(query: str, k: int = 25, branches: bool | None = None, **kw) -> list[dict]:
    """The code lane: search only the code source, with the full ``k`` budget.

    Separate from ``search`` by design. Code answers a different question than prose does
    ("how is this implemented" vs "what happened / why"), and the two rank in different
    vector spaces whose cosine scales are not comparable — which is exactly why fusing
    them has to score by rank position and therefore hands the code space a guaranteed
    top slot on every query, relevant or not. Kept apart, code costs prose nothing and an
    agent can read these hits and decide for itself whether any are relevant. That
    judgment is the part no similarity threshold can make.

    ``branches`` splits main from in-flight work, which is a *truth-status* distinction,
    not a relevance one: a symbol on `main` is how the system works, a branch symbol is
    something someone is trying that may never land. Answering "how does X work" from
    branch code is not a worse answer, it is a wrong one.

      ``False`` → main only (authoritative: how it works today)
      ``True``  → branch symbols only (speculative: what is in flight)
      ``None``  → both, undifferentiated (legacy; prefer an explicit choice)

    Branch symbols exist only when they *differ* from the merge base, so the corpus
    selects for actively-edited code — which is exactly what people ask about. Measured,
    they are 39% of code chunks but took 48-84% of an undifferentiated top-25, and 21/25
    on "how do we compute MFU". Splitting the budget is what stops in-flight work from
    crowding out the code that actually ships.
    """
    kw.pop("source", None)
    if branches is True:
        kw["kind"] = [BRANCH_KIND]
    elif branches is False:
        kw["exclude_kind"] = [BRANCH_KIND]
    return search(query, k=k, source=code_sources(), **kw)


def search_split(query: str, k: int = 10, code_k: int = 25, branch_k: int = 10,
                 **kw) -> dict:
    """All three lanes: ``{"hits": prose, "code": main, "branches": in-flight}``.

    Three lanes because there are three questions with three truth statuses — what
    happened (the record), how does it work (main, authoritative), what is in flight
    (branches, speculative). None of them compete for another's slots, so none can crowd
    another out, and a caller cannot accidentally cite unmerged code as current behavior.

    The prose list is byte-for-byte what ``search`` returned before the code space
    existed, so adopting this can never regress an existing prose result. The code lists
    are additive — they are meant to be *offered* to a reasoning caller, which should feel
    free to conclude that none of it is relevant. "Nothing in code" is a real answer.
    """
    return {"hits": search(query, k=k, **kw),
            "code": search_code(query, k=code_k, branches=False, **kw),
            "branches": search_code(query, k=branch_k, branches=True, **kw)}
