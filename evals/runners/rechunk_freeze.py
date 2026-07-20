#!/usr/bin/env python3
"""Rebuild a frozen corpus with sub-chunked GitHub bodies, holding everything else fixed.

Why not just run the marinmirror pipeline? Because the local raw stores have drifted
from the freeze (31 issues / 26 PRs / 137 comments present in the freeze are missing
locally, and labels/milestones have been edited since). Rebuilding from raw would
change *two* variables at once — the document set and the chunking — and the eval
comparison would mean nothing.

So this re-chunks the freeze *from itself*:

  * github rows      → reconstructed into (head, body, terms), re-chunked through
                       marinmirror.chunking, re-embedded
  * everything else  → copied verbatim, embedding blob and all (zero re-embed, so
                       discord/wandb are provably byte-identical to the source freeze)
  * a github part whose embed_text hashes to the source freeze's hash reuses that
    vector — which doubles as an assertion that unsplit documents are unchanged

Result: the only difference between the two indexes is how long GitHub bodies are
split and embedded. Usage:

    python3 evals/runners/rechunk_freeze.py <src-freeze-dir> <dst-freeze-dir>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.home() / "workspace" / "marinmirror"))

from marinmirror import config                                    # noqa: E402
from marinmirror.chunking import chunk_document                   # noqa: E402
from marinmirror.pipeline.assemble import SCHEMA_DDL              # noqa: E402
from marinmirror.sources.util import clean                        # noqa: E402

import hashlib                                                    # noqa: E402

BATCH = 256          # 18-core/137GB box, not the 2-core deploy VM


def content_hash(embed_text: str) -> str:
    return hashlib.sha1((embed_text or "").encode("utf-8")).hexdigest()


def split_head(text: str) -> tuple[str, str]:
    """``text`` for an issue/PR is ``"{title}[\\n{meta_line}]\\n\\n{body}"``."""
    parts = (text or "").split("\n\n", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (parts[0], "")


def terms_from_head(head: str) -> str:
    """Invert ``_meta``'s display_line back into embed_terms.

    display_line = ``"Labels: a b  Milestone: m  Type: t"`` (two-space separated),
    embed_terms = ``"a b. m. t"``. Values are clean()'d, so they never contain a
    double space and the split is unambiguous.
    """
    lines = head.split("\n", 1)
    if len(lines) < 2:
        return ""
    vals = []
    for field in lines[1].split("  "):
        if ": " in field:
            vals.append(field.split(": ", 1)[1])
    return ". ".join(v for v in vals if v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    a = ap.parse_args()

    src_db = a.src / "corpus-index.db"
    if not src_db.exists():
        print(f"no corpus-index.db under {a.src}", file=sys.stderr)
        return 1
    a.dst.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(src_db)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM chunks").fetchall()
    src_meta = dict(con.execute("SELECT key,value FROM meta"))
    con.close()
    print(f"source: {len(rows)} chunks  (chunk_size={config.CHUNK_SIZE} "
          f"overlap={config.CHUNK_OVERLAP})")

    passthrough, to_embed, reused = [], [], []
    for r in rows:
        if r["source"] != "github":
            passthrough.append((dict(r), r["hash"], r["embedding"], 0, 1))
            continue
        title = r["title"] or ""
        if r["kind"] == "comment":
            head, body, terms, sep, etitle = "", (r["text"] or ""), "", ": ", title
            # comment chunk titles are "comment on #N — {parent title}"; embed used
            # "Re {parent title}: {body}"
            etitle = "Re " + (title.split(" — ", 1)[1] if " — " in title else "")
        else:
            head, body = split_head(r["text"] or "")
            terms, sep, etitle = terms_from_head(head), ". ", title

        pairs = chunk_document(body, title=etitle, terms=terms, sep=sep)
        for i, (part, embed_text) in enumerate(pairs):
            d = dict(r)
            d["ref"] = r["ref"] if len(pairs) == 1 else f"{r['ref']}:p{i}"
            d["text"] = (f"{head}\n\n{part}".strip()
                         if head and i == 0 else part)
            h = content_hash(embed_text)
            rec = [d, h, None, i, len(pairs)]
            if h == r["hash"] and r["embedding"] is not None:
                rec[2] = r["embedding"]          # provably unchanged → reuse vector
                reused.append(rec)
            else:
                to_embed.append((rec, embed_text))

    gh_total = len(reused) + len(to_embed)
    print(f"github: {gh_total} chunks  ({len(reused)} vector-reused, "
          f"{len(to_embed)} to embed)   non-github passthrough: {len(passthrough)}")

    if to_embed:
        import numpy as np
        from fastembed import TextEmbedding
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        # Only github (a prose source) is re-embedded here, so the prose space's model is
        # the right one — read off the source freeze rather than assumed, so a freeze
        # built with a different encoder re-embeds consistently with its own vectors.
        prose_model = ((json.loads(src_meta.get("spaces") or "{}")
                        .get(config.PROSE_SPACE) or {}).get("model")
                       or src_meta.get("model") or config.EMBED_MODEL)
        model = TextEmbedding(prose_model, threads=os.cpu_count())
        t0 = time.time()
        texts = [t for _, t in to_embed]
        done = 0
        for rec, vec in zip((r for r, _ in to_embed),
                            model.embed(texts, batch_size=BATCH)):
            rec[2] = np.asarray(vec, dtype=np.float32).tobytes()
            done += 1
            if done % 5000 == 0:
                el = time.time() - t0
                print(f"  embedded {done}/{len(texts)}  {done/el:.0f}/s  "
                      f"eta {(len(texts)-done)/(done/el)/60:.1f}m", flush=True)
        print(f"  embedded {done} in {(time.time()-t0)/60:.1f}m")

    out = a.dst / "corpus-index.db"
    tmp = Path(str(out) + ".tmp")
    if tmp.exists():
        tmp.unlink()
    w = sqlite3.connect(tmp)
    w.executescript(SCHEMA_DDL)
    ins = ("INSERT INTO chunks(source,kind,ref,parent,title,author,date,url,text,hash,"
           "embedding,embed_space,part,n_parts) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)")
    per_source: dict[str, int] = {}
    n = embedded_n = 0
    # Deterministic row order so a rebuild is byte-comparable run to run.
    all_rows = sorted(passthrough + reused + [r for r, _ in to_embed],
                      key=lambda t: (t[0]["source"], t[0]["kind"], t[0]["ref"], t[3]))
    for d, h, blob, part, n_parts in all_rows:
        # Carry the source's vector space through. A freeze taken before spaces existed
        # has no such column, and every embedded row in one belongs to the single legacy
        # prose space — leaving it NULL would make the vector leg match nothing.
        space = (d.get("embed_space") or config.PROSE_SPACE) if blob is not None else None
        w.execute(ins, (d["source"], d["kind"], d["ref"], d["parent"], d["title"],
                        d["author"], d["date"], d["url"], d["text"], h, blob, space,
                        part, n_parts))
        n += 1
        embedded_n += blob is not None
        per_source[d["source"]] = per_source.get(d["source"], 0) + 1
    w.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5("
              "text, title, content='chunks', content_rowid='id')")
    w.execute("INSERT INTO chunks_fts(rowid,text,title) SELECT id,text,title FROM chunks")
    meta = dict(src_meta)
    meta.update({
        "chunks": str(n), "embedded": str(embedded_n),
        "per_source": json.dumps(per_source),
        "chunk_size": str(config.CHUNK_SIZE),
        "chunk_overlap": str(config.CHUNK_OVERLAP),
        "rechunked_from": str(a.src.name),
        "built_at_epoch": str(int(time.time())),
    })
    w.execute("DELETE FROM meta")
    w.executemany("INSERT INTO meta(key,value) VALUES(?,?)", list(meta.items()))
    w.commit()
    w.execute("PRAGMA optimize")
    w.close()
    os.replace(tmp, out)

    for extra in ("corpus-manifest.json",):
        if (a.src / extra).exists():
            shutil.copy2(a.src / extra, a.dst / extra)
    if (a.src / "summaries").exists() and not (a.dst / "summaries").exists():
        shutil.copytree(a.src / "summaries", a.dst / "summaries")

    print(f"\nwrote {out}  ({out.stat().st_size/1e6:.0f} MB)")
    print("  " + ", ".join(f"{k} {v}" for k, v in sorted(per_source.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
