#!/usr/bin/env python3
"""Deterministic retrieval metric: can the index surface the gold citations at all?

Q (score.py) measures retrieval *and* synthesis through one LLM, over 9 questions —
too noisy to attribute a change to the index. This measures the index alone: run each
question's text through hybrid search and ask what fraction of its gold must-have
issue/PR citations appear in the top-k documents. No LLM, fully repeatable, and it
moves only if retrieval moves.

Ceiling caveat: a gold citation that the *answer* found via thread expansion or a
weekly summary may legitimately never appear in a top-k search hit, so absolute recall
here is a lower bound. The comparison between two indexes on identical queries is the
meaningful number, not the absolute level.

    python3 evals/runners/retrieval_recall.py 2026-07-16 2026-07-18-chunked
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
KS = (10, 20, 50)


def issues_from(hits) -> list[str]:
    """Issue/PR numbers in retrieved-rank order, deduped, keeping first occurrence."""
    out, seen = [], set()
    for h in hits:
        m = re.search(r"/(?:issues|pull)/(\d+)", h.get("url") or "")
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            out.append(m.group(1))
    return out


def _reload_corpus(freeze: str):
    """Re-import mumwelt against a different MARIN_CACHE.

    Every `mumwelt.*` module must go, including the package itself: `mumwelt.corpus`
    does `from . import config`, which resolves via the *attribute* on the cached
    parent package. Dropping only the submodules leaves that attribute bound to the
    old config and both arms silently read the same database.
    """
    os.environ["MARIN_CACHE"] = str(ROOT / "evals" / "corpus" / freeze)
    sys.path.insert(0, str(ROOT))
    for name in [m for m in sys.modules if m == "mumwelt" or m.startswith("mumwelt.")]:
        del sys.modules[name]
    corpus = importlib.import_module("mumwelt.corpus")
    config = importlib.import_module("mumwelt.config")
    assert config.CORPUS.parent.name == freeze, f"wrong corpus: {config.CORPUS}"
    return corpus


def queries_for(qid: str, question: str) -> list[str]:
    """The question plus the sub-queries the archived baseline actually issued.

    A single broad query ("explain how training on GPUs is going") is not what the
    skill does — it decomposes and fans out. The archived run recorded its sub-queries
    in the provenance footer, so replaying that exact set against both indexes is both
    more representative and still a controlled comparison (identical queries, identical
    gold, only the index differs).
    """
    out = [question]
    p = ROOT / "evals/candidates/opus48-marin-research" / f"{qid}.md"
    if p.exists():
        m = re.search(r"\*Sub-queries:\s*(.+?)\*", p.read_text(), re.S)
        if m:
            out += [s.strip().strip('"') for s in m.group(1).split("·") if s.strip()]
    return out


def run_freeze(freeze: str, questions: list[dict], k: int) -> dict[str, list[str]]:
    """Search every question against one freeze; returns qid -> ranked issue numbers.

    Sub-query hits are pooled by best rank across the fan-out, mirroring how the CLI's
    search-multi merges them.
    """
    corpus = _reload_corpus(freeze)
    out = {}
    for q in questions:
        best: dict[str, int] = {}
        for query in queries_for(q["id"], q["question"]):
            for rank, num in enumerate(issues_from(corpus.search(query, k=k))):
                if num not in best or rank < best[num]:
                    best[num] = rank
        out[q["id"]] = [n for n, _ in sorted(best.items(), key=lambda kv: kv[1])]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("freezes", nargs="+")
    a = ap.parse_args()

    questions = [json.load(open(p)) for p in sorted((ROOT / "evals/questions").glob("*.json"))]
    results = {f: run_freeze(f, questions, max(KS)) for f in a.freezes}

    print(f"{'question':12} {'must':>5}", end="")
    for f in a.freezes:
        print(f"  |{f[:18]:>18}", end="")
    print("\n" + " " * 12 + f"{'':>5}", end="")
    for _ in a.freezes:
        print("  |" + "".join(f"{'@'+str(k):>6}" for k in KS), end="")
    print()

    totals = {f: {k: [0, 0] for k in KS} for f in a.freezes}
    for q in questions:
        must = {str(c).lstrip("#") for c in q.get("citations", {}).get("must_have", [])}
        print(f"{q['id']:12} {len(must):5}", end="")
        for f in a.freezes:
            ranked = results[f][q["id"]]
            print("  |", end="")
            for k in KS:
                hit = len(must & set(ranked[:k]))
                totals[f][k][0] += hit
                totals[f][k][1] += len(must)
                print(f"{(hit/len(must)*100 if must else 0):5.0f}%", end="")
        print()

    print(f"\n{'MICRO MEAN':12} {'':5}", end="")
    for f in a.freezes:
        print("  |", end="")
        for k in KS:
            hit, tot = totals[f][k]
            print(f"{(100*hit/tot if tot else 0):5.1f}%", end="")
    print()
    out = ROOT / "evals" / "retrieval-recall.json"
    out.write_text(json.dumps(
        {"freezes": a.freezes, "ks": list(KS),
         "per_question": {f: results[f] for f in a.freezes},
         "micro": {f: {str(k): totals[f][k] for k in KS} for f in a.freezes}}, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
