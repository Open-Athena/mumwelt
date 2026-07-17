#!/usr/bin/env python3
"""Mint a content-addressed corpus freeze under evals/corpus/<date>/ (DESIGN.md §3).

Copies corpus-index.db + corpus-manifest.json + summaries/ out of a live MARIN_CACHE
and writes FREEZE.json (sha256s, chunk/thread/source counts, summary metadata) so the
harness can pin to it and refuse to run on a mismatched DB.

    python3 evals/runners/build_freeze.py 2026-07-16 [--src ~/.cache/marin]
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, sqlite3, time, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--src", default=os.path.expanduser("~/.cache/marin"))
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    dst = ROOT / "corpus" / args.date
    dst.mkdir(parents=True, exist_ok=True)

    for name in ("corpus-index.db", "corpus-manifest.json"):
        shutil.copy2(src / name, dst / name)
    if (dst / "summaries").exists():
        shutil.rmtree(dst / "summaries")
    shutil.copytree(src / "summaries", dst / "summaries")

    db, man = dst / "corpus-index.db", dst / "corpus-manifest.json"
    manifest = json.load(open(man))

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        per_source = dict(con.execute("SELECT source, count(*) FROM chunks GROUP BY source"))
        chunk_count = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
        parent_threads = con.execute(
            "SELECT count(DISTINCT parent) FROM chunks WHERE source='github'").fetchone()[0]
    finally:
        con.close()

    sdir = dst / "summaries"
    weeks = sorted(p.name for p in sdir.glob("summary-*.html"))
    stext = "".join(p.read_text(errors="ignore") for p in sdir.rglob("*.html"))
    issue_refs = len(set(re.findall(r"/(?:issues|pull)/(\d{2,6})", stext))
                     | set(re.findall(r"#(\d{3,5})\b", stext)))

    freeze = {
        "date": args.date,
        "source": str(src).replace(os.path.expanduser("~"), "~"),
        "built_at_epoch": manifest.get("built_at_epoch"),
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                      time.gmtime(manifest.get("built_at_epoch", time.time()))),
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embed_model": manifest.get("embed_model"),
        "embed_dim": manifest.get("embed_dim"),
        "chunk_count": chunk_count,
        "parent_threads": parent_threads,
        "per_source": per_source,
        "sha256": {"corpus-index.db": sha256(db), "corpus-manifest.json": sha256(man)},
        "summaries": {
            "weeks": len(weeks),
            "latest": weeks[-1] if weeks else None,
            "issue_refs": issue_refs,
        },
        "note": "frozen corpus = corpus-index.db chunks UNION issue #s referenced by these summaries",
    }
    json.dump(freeze, open(dst / "FREEZE.json", "w"), indent=2)
    print(json.dumps(freeze, indent=2))
    print(f"\nfroze {chunk_count} chunks -> {dst}")


if __name__ == "__main__":
    main()
