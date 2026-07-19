#!/usr/bin/env python3
"""Score several candidate runs against the golds and print a side-by-side table.

    python3 evals/runners/compare_runs.py \
        archived=candidates/opus48-marin-research \
        base=candidates/opus48-base-rerun \
        chunked=candidates/opus48-chunked

Each argument is ``label=path`` relative to ``evals/``. Every run is scored in
``frozen`` mode against the same gold records, so the only thing that varies is the
answer text. Also reports a paired per-question delta against the FIRST run listed,
because with 9 questions a mean difference is easy to over-read.
"""
from __future__ import annotations

import json
import pathlib
import re
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runners"))
import score as scorer  # noqa: E402

ORDER = ["gpu", "h100-67b", "muon", "july", "april", "ablations", "classifier",
         "benchmarks", "inference"]


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    runs = []
    for a in args:
        label, _, rel = a.partition("=")
        runs.append((label, ROOT / rel))

    freeze = scorer.freeze_issue_set(*scorer.freeze_paths(ROOT))
    print(f"# freeze={scorer.FREEZE_NAME}  ({len(freeze):,} issue/PR numbers in-corpus)\n")

    rows: dict[str, dict[str, dict]] = {}
    for qid in ORDER:
        qp = ROOT / "questions" / f"{qid}.json"
        if not qp.exists():
            continue
        q = json.load(open(qp))
        rows[qid] = {}
        for label, d in runs:
            p = d / f"{qid}.md"
            if not p.exists():
                continue
            body = re.sub(r"<!--.*?-->", "", p.read_text(), flags=re.S)
            rows[qid][label] = scorer.score(q, body, freeze, mode="frozen")

    labels = [l for l, _ in runs]
    w = max(len(l) for l in labels) + 8
    print(f"{'question':12}" + "".join(f"{l:>{w}}" for l in labels))
    print(f"{'':12}" + "".join(f"{'Q  cite/fact':>{w}}" for _ in labels))
    for qid in ORDER:
        if qid not in rows:
            continue
        print(f"{qid:12}", end="")
        for l in labels:
            r = rows[qid].get(l)
            if not r:
                print(f"{'—':>{w}}", end="")
                continue
            cell = (f"{r['Q']:5.1f} {r['Q_components']['cite_recall_MH']:.2f}/"
                    f"{r['facts']['cov_MH']:.2f}")
            gate = "" if r["Q"] > 0 or not any(
                g.get("fail") for g in r["gates"].values()) else "!"
            print(f"{cell+gate:>{w}}", end="")
        print()

    print(f"\n{'MEAN':12}", end="")
    means = {}
    for l in labels:
        qs = [rows[q][l]["Q"] for q in rows if l in rows[q]]
        means[l] = statistics.mean(qs) if qs else float("nan")
        print(f"{means[l]:>{w}.1f}", end="")
    print()

    if len(labels) >= 2:
        ref = labels[0]
        print(f"\nPaired deltas vs '{ref}' (per question):")
        for l in labels[1:]:
            ds = [rows[q][l]["Q"] - rows[q][ref]["Q"]
                  for q in rows if l in rows[q] and ref in rows[q]]
            if not ds:
                continue
            wins = sum(d > 0 for d in ds)
            sd = statistics.stdev(ds) if len(ds) > 1 else 0.0
            print(f"  {l:>12}: mean {statistics.mean(ds):+6.1f}  median "
                  f"{statistics.median(ds):+6.1f}  sd {sd:5.1f}  "
                  f"better on {wins}/{len(ds)}  [{', '.join(f'{d:+.0f}' for d in ds)}]")

    out = ROOT / "compare-runs.json"
    out.write_text(json.dumps(
        {"freeze": scorer.FREEZE_NAME, "means": means,
         "per_question": {q: {l: {k: v for k, v in r.items() if k != "cited_all"}
                              for l, r in rows[q].items()} for q in rows}},
        indent=2, default=str) + "\n")
    print(f"\nwrote {out.relative_to(ROOT.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
