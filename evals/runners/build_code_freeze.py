#!/usr/bin/env python3
"""Mint an eval freeze = an existing freeze + the code source + the `syms` column.

Builds the arms used to A/B the code source (see evals/retrieval-recall.json):

    2026-07-18-syms   baseline + syms column/3-col FTS, NO code rows  (isolates the client)
    2026-07-18-code   the above + 43,960 code chunks                  (isolates dilution)

The syms-only arm is the control that makes the result readable: it came out identical to
the baseline on all 9 questions at every k, which is what pins the observed movement on the
code chunks rather than on the FTS schema change.

The point is attribution: the prose half of the corpus is byte-identical to the baseline
freeze, so any movement in retrieval_recall.py is caused by *this change* and nothing else
(no corpus drift; existing rows' vectors are copied, never recomputed).

Embeddings are cached to disk keyed by content hash, so a re-run after a schema or
scripting fix costs seconds instead of the ~16 minutes the initial 44k-chunk embed takes.

    python build_code_freeze.py <out-name> [--base 2026-07-18-chunked]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sqlite3
import sys
import time

MUM = pathlib.Path.home() / "workspace" / "mumwelt"
MIRROR = pathlib.Path.home() / "workspace" / "marinmirror"
sys.path.insert(0, str(MIRROR))


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _cache_conn(path: pathlib.Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE IF NOT EXISTS vec(hash TEXT PRIMARY KEY, blob BLOB)")
    return con


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--base", default="2026-07-18-chunked")
    a = ap.parse_args()

    scratch = pathlib.Path(os.environ["SCRATCH"])
    src = MUM / "evals" / "corpus" / a.base
    out = MUM / "evals" / "corpus" / a.out
    out.mkdir(parents=True, exist_ok=True)
    db = out / "corpus-index.db"

    print(f"[1/6] copying baseline {a.base} -> {db}")
    shutil.copy2(src / "corpus-index.db", db)
    shutil.copy2(src / "corpus-manifest.json", out / "corpus-manifest.json")
    if not (out / "summaries").exists() and (src / "summaries").exists():
        shutil.copytree(src / "summaries", out / "summaries")

    from marinmirror import config
    config.CODE_REPO_DIR = pathlib.Path.home() / "workspace" / "marin"
    config.CODE_FILEMETA = scratch / "filemeta.json"
    config.STATE_FILE = scratch / "state.json"
    from marinmirror.identifiers import expand
    from marinmirror.pipeline import embed as embed_stage
    from marinmirror.sources.code import CodeSource

    print("[2/6] generating code chunks")
    t0 = time.time()
    chunks = list(CodeSource().chunks())
    print(f"      {len(chunks)} chunks in {time.time()-t0:.0f}s")

    print("[3/6] embedding (cached by content hash)")
    t0 = time.time()
    cache = _cache_conn(scratch / "embed-cache.db")
    for c in chunks:
        c.hash = embed_stage.content_hash(c)
    have = dict(cache.execute("SELECT hash, blob FROM vec"))
    missing = [c for c in chunks if c.hash not in have]
    print(f"      {len(have)} cached, {len(missing)} to embed")
    if missing:
        embedded = embed_stage.embed(missing, prev_index_path=db)
        cache.executemany("INSERT OR REPLACE INTO vec(hash,blob) VALUES(?,?)",
                          [(h, blob) for _, h, blob in embedded])
        cache.commit()
        have = dict(cache.execute("SELECT hash, blob FROM vec"))
    print(f"      embed stage done in {time.time()-t0:.0f}s")

    con = sqlite3.connect(db)
    cols = [r[1] for r in con.execute("PRAGMA table_info(chunks)")]
    if "syms" not in cols:
        print("[4/6] adding syms column")
        con.execute("ALTER TABLE chunks ADD COLUMN syms TEXT")
        cols.append("syms")

    # Insert only the columns this freeze's schema actually has: older freezes predate
    # part/n_parts, and hardcoding the newest column set silently breaks against them.
    field = {"source": lambda c: c.source, "kind": lambda c: c.kind, "ref": lambda c: c.ref,
             "parent": lambda c: c.parent, "title": lambda c: c.title,
             "author": lambda c: c.author, "date": lambda c: c.date, "url": lambda c: c.url,
             "text": lambda c: c.text, "hash": lambda c: c.hash,
             "embedding": lambda c: have.get(c.hash), "part": lambda c: c.part,
             "n_parts": lambda c: c.n_parts,
             "syms": lambda c: expand(f"{c.text}\n{c.title}")}
    use = [c for c in cols if c in field]
    print(f"[5/6] inserting {len(chunks)} code rows into columns: {', '.join(use)}")
    ins = f"INSERT INTO chunks({','.join(use)}) VALUES({','.join('?' * len(use))})"
    con.executemany(ins, [[field[k](c) for k in use] for c in chunks])
    con.commit()

    print("[6/6] rebuilding FTS over (text, title, syms)")
    con.execute("DROP TABLE IF EXISTS chunks_fts")
    con.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5("
                "text, title, syms, content='chunks', content_rowid='id')")
    con.execute("INSERT INTO chunks_fts(rowid,text,title,syms) "
                "SELECT id,text,title,syms FROM chunks")
    con.commit()
    n, ne = con.execute("SELECT count(*), count(embedding) FROM chunks").fetchone()
    per = dict(con.execute("SELECT source,count(*) FROM chunks GROUP BY 1"))
    con.execute("PRAGMA optimize")
    con.close()

    freeze = json.loads((src / "FREEZE.json").read_text())
    freeze.update({"date": a.out, "chunk_count": n, "sha256_corpus": sha256(db),
                   "per_source": per, "derived_from": a.base,
                   "note": "baseline freeze + code source + syms column"})
    (out / "FREEZE.json").write_text(json.dumps(freeze, indent=2))
    print(f"\ndone: {n} chunks ({ne} embedded) -> {out}")
    print("per source:", per)


if __name__ == "__main__":
    main()
