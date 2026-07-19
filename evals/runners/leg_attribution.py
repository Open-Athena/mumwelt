#!/usr/bin/env python3
"""Which retrieval leg actually finds the gold citations — FTS, vector, or both?

`retrieval_recall.py` measures the fused result. That hides the thing we actually
need to know: if the vector leg contributes almost nothing, then fixing its coverage
(sub-chunking) cannot move the fused result no matter how much text it unlocks.

For each question we run the identical query set three ways — FTS only, vector only,
and the normal RRF fusion — and measure gold must-have citation recall for each. Then
we attribute each *found* gold citation to the leg(s) that surfaced it.

    python3 evals/runners/leg_attribution.py 2026-07-16 2026-07-18-chunked
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
K = 20


def issues_from(hits) -> list[str]:
    out, seen = [], set()
    for h in hits:
        m = re.search(r"/(?:issues|pull)/(\d+)", h.get("url") or "")
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            out.append(m.group(1))
    return out


def _reload(freeze: str):
    os.environ["MARIN_CACHE"] = str(ROOT / "evals" / "corpus" / freeze)
    sys.path.insert(0, str(ROOT))
    for name in [m for m in sys.modules if m == "mumwelt" or m.startswith("mumwelt.")]:
        del sys.modules[name]
    corpus = importlib.import_module("mumwelt.corpus")
    cfg = importlib.import_module("mumwelt.config")
    assert cfg.CORPUS.parent.name == freeze
    return corpus


def queries_for(qid: str, question: str) -> list[str]:
    out = [question]
    p = ROOT / "evals/candidates/opus48-marin-research" / f"{qid}.md"
    if p.exists():
        m = re.search(r"\*Sub-queries:\s*(.+?)\*", p.read_text(), re.S)
        if m:
            out += [s.strip().strip('"') for s in m.group(1).split("·") if s.strip()]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("freezes", nargs="+")
    a = ap.parse_args()
    questions = [json.load(open(p))
                 for p in sorted((ROOT / "evals/questions").glob("*.json"))]

    report = {}
    for freeze in a.freezes:
        corpus = _reload(freeze)
        tot = {"fts": 0, "vec": 0, "both": 0, "hybrid": 0, "need": 0,
               "fts_only": 0, "vec_only": 0}
        per_q = {}
        for q in questions:
            must = {str(c).lstrip("#") for c in q["citations"]["must_have"]}
            f_hits, v_hits, h_hits = set(), set(), set()
            for query in queries_for(q["id"], q["question"]):
                f_hits |= set(issues_from(corpus.search(query, k=K, fts_only=True)))
                v_hits |= set(issues_from(corpus.search(query, k=K, vec_only=True)))
                h_hits |= set(issues_from(corpus.search(query, k=K)))
            f, v, h = must & f_hits, must & v_hits, must & h_hits
            tot["need"] += len(must)
            tot["fts"] += len(f)
            tot["vec"] += len(v)
            tot["both"] += len(f & v)
            tot["fts_only"] += len(f - v)
            tot["vec_only"] += len(v - f)
            tot["hybrid"] += len(h)
            per_q[q["id"]] = {"must": len(must), "fts": len(f), "vec": len(v),
                              "fts_only": len(f - v), "vec_only": len(v - f),
                              "hybrid": len(h)}
        report[freeze] = {"totals": tot, "per_question": per_q}

    n = report[a.freezes[0]]["totals"]["need"]
    print(f"gold must-have citations: {n}   (recall@{K}, sub-queries pooled)\n")
    hdr = f"{'freeze':22}{'FTS':>8}{'VEC':>8}{'both':>8}{'FTS-only':>10}{'VEC-only':>10}{'fused':>8}"
    print(hdr)
    print("-" * len(hdr))
    for f in a.freezes:
        t = report[f]["totals"]
        pct = lambda x: f"{100*x/t['need']:.0f}%"
        print(f"{f:22}{pct(t['fts']):>8}{pct(t['vec']):>8}{pct(t['both']):>8}"
              f"{pct(t['fts_only']):>10}{pct(t['vec_only']):>10}{pct(t['hybrid']):>8}")

    print(f"\nper-question VEC-only finds (citations no keyword query surfaced):")
    for f in a.freezes:
        pq = report[f]["per_question"]
        print(f"  {f:22}" + "  ".join(f"{k}:{v['vec_only']}" for k, v in pq.items()))

    out = ROOT / "evals" / "leg-attribution.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
