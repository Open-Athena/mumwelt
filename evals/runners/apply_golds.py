#!/usr/bin/env python3
"""Fold generated proposed golds into the question records so score.py can run.

Consumes:
  - evals/golds-proposed/<id>.md            (the gold prose, written by the synth agent)
  - evals/golds-proposed/_targets.json      ({id: {must_have_citations, nice_citations,
                                              forbidden_citations, key_facts:[{match,note}]}})
Writes each questions/<id>.json with: gold_prose, citations (must/nice/forbidden), facts.must_have,
and a minimal trap_checks (superseded->forbid_cite). All PROVISIONAL, pending human gold review.

Validates every must-have citation exists in the frozen corpus (flags out-of-freeze ones).

    python3 evals/runners/apply_golds.py
"""
from __future__ import annotations
import json, pathlib, sys, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
QDIR = ROOT / "questions"
GDIR = ROOT / "golds-proposed"
CORPUS = os.environ.get("MARIN_EVAL_FREEZE", "2026-07-16")
sys.path.insert(0, str(ROOT / "runners"))
import score as scorer  # noqa: E402


def norm(c):
    return str(c).lstrip("#").strip()


def safe_rx(pat):
    """Ensure the fact 'match' compiles; if the agent emitted an invalid regex,
    fall back to a literal (escaped) match so score.py never throws."""
    try:
        re.compile(pat)
        return pat
    except re.error:
        return re.escape(pat)


def main() -> None:
    targets = json.load(open(GDIR / "_targets.json"))
    freeze = scorer.freeze_issue_set(ROOT / f"corpus/{CORPUS}/corpus-index.db",
                                     ROOT / f"corpus/{CORPUS}/summaries")
    for qid, t in targets.items():
        qp = QDIR / f"{qid}.json"
        q = json.load(open(qp))
        gold_md = GDIR / f"{qid}.md"
        q["gold_prose"] = gold_md.read_text().strip() if gold_md.exists() else ""

        prose = q["gold_prose"]
        gold_cited = scorer.cited_issues(prose)  # what the gold actually cites

        must = [norm(c) for c in t.get("must_have_citations", [])]
        nice = [norm(c) for c in t.get("nice_citations", [])]
        forb = [norm(c) for c in t.get("forbidden_citations", [])]

        # RECONCILE so the gold scores 100 against its own targets (it is the reference):
        #  - a must-have the gold doesn't cite is a gold-prose gap, not a candidate target
        #  - a "forbidden" the gold DOES cite can't be forbidden (it self-penalizes)
        #  - a fact whose regex doesn't match the gold prose is a bad regex
        # Dropped items are recorded under review_* for the human gold pass.
        must_keep = [c for c in must if c in gold_cited]
        must_drop = [c for c in must if c not in gold_cited]
        forb_keep = [c for c in forb if c not in gold_cited]
        forb_drop = [c for c in forb if c in gold_cited]
        oof = [c for c in must_keep if c not in freeze]  # should be empty (gold is corpus-grounded)

        facts_in = t.get("key_facts", []) or []
        facts_keep, facts_drop = [], []
        for f in facts_in:
            m = f.get("match")
            if not m:
                continue
            rx = safe_rx(m)
            (facts_keep if re.search(rx, prose, re.I) else facts_drop).append((rx, f.get("note", "")))

        q["citations"] = {
            "must_have": [f"#{c}" for c in must_keep],
            "nice_to_have": [f"#{c}" for c in nice],
            "forbidden": [f"#{c}" for c in forb_keep],
            "recency_critical": [],
        }
        q["facts"] = {
            "must_have": [{"id": str(i + 1), "match": rx, "note": note}
                          for i, (rx, note) in enumerate(facts_keep)],
            "nice_to_have": [],
        }
        q["trap_checks"] = {"superseded": {"forbid_cite": forb_keep}} if forb_keep else {}
        q["traps"] = ({"superseded": [f"#{c} is wrong/superseded/off-platform here" for c in forb_keep]}
                      if forb_keep else {})
        q["gold_provisional"] = True
        # review breadcrumbs for the human gold pass
        review = {}
        if must_drop:
            review["must_have_not_cited_by_gold"] = must_drop
        if forb_drop:
            review["forbidden_but_gold_cites_them"] = forb_drop
        if facts_drop:
            review["facts_gold_fails"] = [{"match": rx, "note": n} for rx, n in facts_drop]
        if oof:
            review["must_have_out_of_freeze"] = oof
        q["review_notes"] = review

        json.dump(q, open(qp, "w"), indent=2)
        drp = f"  drop: must-{len(must_drop)} forb-{len(forb_drop)} fact-{len(facts_drop)}" if review else ""
        oofp = f"  ⚠OOF:{oof}" if oof else ""
        print(f"{qid:11} must={len(must_keep):2} nice={len(nice):2} forb={len(forb_keep):2} "
              f"facts={len(facts_keep):2}{drp}{oofp}")


if __name__ == "__main__":
    main()
