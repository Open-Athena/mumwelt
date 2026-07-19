#!/usr/bin/env python3
"""Deterministic scorer for Marin skill evals (DESIGN.md §4).

No LLM. Given a question record and a candidate answer's text, compute the
quality number Q (0-100), the hard-gate status, and every facet sub-metric.
Reused by the harness (score.py stage) and by the /tmp viewer generator.

    from score import freeze_issue_set, score
    valid = freeze_issue_set("evals/corpus/2026-07-16/corpus-index.db")
    result = score(question_dict, answer_text, valid, mode="frozen")
"""
from __future__ import annotations
import os, re, sqlite3, pathlib

# Q weights (DESIGN §4.2). Nice-to-haves are deliberately NOT in Q.
W_CITE, W_FACT, W_TRAP = 0.35, 0.35, 0.30
FORBIDDEN_PENALTY = 10  # points per forbidden citation

# Phrases that signal an explicit "not found" abstention.
_ABSTAIN = re.compile(
    r"\b(not found|no (?:evidence|information|record|mention)|"
    r"could(?:n't| not) (?:find|locate)|nothing (?:in|found)|"
    r"i (?:don'?t|do not) (?:have|find)|no relevant)\b", re.I)


def cited_issues(text: str) -> set[str]:
    """Every issue/PR reference in the answer text — both #NNNN and github URLs."""
    nums = set(re.findall(r"#(\d{3,5})\b", text))
    nums |= set(re.findall(r"/(?:issues|pull)/(\d{2,6})", text))
    return nums


#: Which frozen corpus the harness scores against. Override to compare index builds:
#:     MARIN_EVAL_FREEZE=2026-07-18-chunked python3 evals/runners/score.py
#: `evals/runners/mum-frozen` reads the same variable, so retrieval and the
#: hallucination gate can never drift onto different corpora.
FREEZE_NAME = os.environ.get("MARIN_EVAL_FREEZE", "2026-07-16")


def freeze_dir(root: pathlib.Path | None = None) -> pathlib.Path:
    """Absolute path of the active freeze directory."""
    root = root or pathlib.Path(__file__).resolve().parents[1]
    return root / "corpus" / FREEZE_NAME


def freeze_paths(root: pathlib.Path | None = None) -> tuple[pathlib.Path, pathlib.Path]:
    """``(corpus-index.db, summaries/)`` for the active freeze."""
    d = freeze_dir(root)
    return d / "corpus-index.db", d / "summaries"


def freeze_issue_set(db_path: str | pathlib.Path,
                     summaries_dir: str | pathlib.Path | None = None) -> set[str]:
    """Issue/PR numbers present anywhere in the frozen corpus.

    The frozen corpus is the indexed chunks in ``corpus-index.db`` UNION the issue
    numbers referenced by the frozen weekly summaries — a candidate that legitimately
    read a summary can cite an issue whose primary thread isn't separately indexed, and
    that must not count as a hallucination.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        urls = [r[0] for r in con.execute(
            "SELECT url FROM chunks WHERE source='github' AND url IS NOT NULL")]
        # An issue can also be *referenced* from inside an indexed non-GitHub chunk —
        # e.g. a Discord message reading "GH issue added! .../marin/issues/7164".
        # A candidate that retrieved that message and cited the issue did real
        # retrieval, so gating it as a hallucination is a false positive. Only full
        # URLs are harvested here: a bare "#7164" in chat is too ambiguous to trust.
        refs = [r[0] for r in con.execute(
            "SELECT text FROM chunks WHERE source!='github' AND text IS NOT NULL")]
    finally:
        con.close()
    out: set[str] = set()
    for u in urls:
        m = re.search(r"/(?:issues|pull)/(\d+)", u)
        if m:
            out.add(m.group(1))
    for t in refs:
        out |= set(re.findall(r"/(?:issues|pull)/(\d{2,6})", t))
    if summaries_dir:
        sdir = pathlib.Path(summaries_dir)
        if sdir.exists():
            txt = "".join(p.read_text(errors="ignore") for p in sdir.rglob("*.html"))
            out |= set(re.findall(r"/(?:issues|pull)/(\d{2,6})", txt))
            out |= set(re.findall(r"#(\d{3,5})\b", txt))
    return out


def _norm(cites):
    return {str(c).lstrip("#") for c in cites}


def _recall(needed, cited, equiv=None):
    """Recall over `needed`, where a needed cite is satisfied by itself OR any of its
    accepted equivalents (question['citation_equivalents'][n]) being in `cited`."""
    needed = _norm(needed)
    if not needed:
        return 1.0, [], []
    equiv = {str(k).lstrip("#"): _norm(v) for k, v in (equiv or {}).items()}
    hit, miss = [], []
    for n in needed:
        if n in cited or (equiv.get(n) and (equiv[n] & cited)):
            hit.append(n)
        else:
            miss.append(n)
    hit.sort(key=int); miss.sort(key=int)
    return len(hit) / len(needed), hit, miss


def _fact_hits(facts, text):
    rows = []
    for f in facts:
        ok = re.search(f["match"], text, re.I) is not None
        rows.append({"id": f["id"], "match": f["match"],
                     "note": f.get("note", ""), "pass": ok})
    frac = (sum(r["pass"] for r in rows) / len(rows)) if rows else 1.0
    return frac, rows


def _trap_hits(question, text, cited):
    checks = question.get("trap_checks", {})
    descs = question.get("traps", {})
    rows = []
    for name, chk in checks.items():
        reasons = []
        ok = True
        for rx in chk.get("require", []):
            if not re.search(rx, text, re.I):
                ok = False; reasons.append(f"missing /{rx}/")
        for rx in chk.get("deny", []):
            if re.search(rx, text, re.I):
                ok = False; reasons.append(f"present /{rx}/ (should be absent)")
        for c in _norm(chk.get("require_cite", [])):
            if c not in cited:
                ok = False; reasons.append(f"not citing #{c}")
        for c in _norm(chk.get("forbid_cite", [])):
            if c in cited:
                ok = False; reasons.append(f"cites forbidden #{c}")
        d = descs.get(name)
        d = "; ".join(d) if isinstance(d, list) else (d or "")
        rows.append({"name": name, "desc": d, "pass": ok, "why": "; ".join(reasons)})
    frac = (sum(r["pass"] for r in rows) / len(rows)) if rows else 1.0
    return frac, rows


def score(question: dict, answer_text: str,
          freeze: set[str] | None = None, mode: str = "frozen") -> dict:
    """Score one candidate answer against one question record.

    mode:
      frozen    - candidate must work off the frozen corpus; any cited #N not in
                  the freeze is a hallucination hard-gate (Q=0).
      live      - candidate hit the live repo; out-of-freeze cites are allowed
                  (flagged as provenance, never gated).
      reference - the gold itself; gates never fire.
    """
    cited = cited_issues(answer_text)
    cites = question.get("citations", {})

    equiv = question.get("citation_equivalents", {})
    cite_recall, cite_hit, cite_miss = _recall(cites.get("must_have", []), cited, equiv)
    nice_recall, nice_hit, _ = _recall(cites.get("nice_to_have", []), cited)
    rec_recall, rec_hit, rec_miss = _recall(cites.get("recency_critical", []), cited, equiv)
    forbidden = sorted(_norm(cites.get("forbidden", [])) & cited, key=int)

    fact_cov, fact_rows = _fact_hits(question.get("facts", {}).get("must_have", []), answer_text)
    nice_fact_cov, nice_fact_rows = _fact_hits(question.get("facts", {}).get("nice_to_have", []), answer_text)
    trap_pass, trap_rows = _trap_hits(question, answer_text, cited)
    # Responsiveness floor: an answer that recalls no must-have citation AND matches no
    # must-have fact is non-responsive — it must not bank trap points purely by omitting a
    # forbidden issue. (DESIGN §9 anti-gaming; judges flagged the 30% trap windfall.)
    trap_zeroed = (cite_recall == 0 and fact_cov == 0 and trap_pass > 0)
    if trap_zeroed:
        trap_pass = 0.0

    # provenance: which cited issues live inside the freeze
    if freeze is not None:
        in_freeze = sorted(cited & freeze, key=int)
        out_freeze = sorted(cited - freeze, key=int)
    else:
        in_freeze, out_freeze = sorted(cited, key=int), []

    # hard gates
    answerable = question.get("answerable", True)
    abstained = bool(_ABSTAIN.search(answer_text)) and len(answer_text) < 600
    gates = {
        "hallucinated_citation": {
            "fail": mode == "frozen" and bool(out_freeze),
            "detail": out_freeze,
            "active": mode == "frozen",
        },
        "abstention": {
            # answerable Q must not abstain; unanswerable Q must abstain
            "fail": (abstained if answerable else not abstained),
            "detail": "abstained" if abstained else "answered",
            "active": True,
        },
    }
    gate_failed = any(g["fail"] for g in gates.values())

    raw = 100 * (W_CITE * cite_recall + W_FACT * fact_cov + W_TRAP * trap_pass)
    q = raw - FORBIDDEN_PENALTY * len(forbidden)
    q = max(0.0, min(100.0, q))
    if gate_failed:
        q = 0.0

    return {
        "mode": mode,
        "Q": round(q, 1),
        "Q_components": {
            "cite_recall_MH": round(cite_recall, 3),
            "fact_cov_MH": round(fact_cov, 3),
            "trap_pass": round(trap_pass, 3),
            "forbidden_penalty": FORBIDDEN_PENALTY * len(forbidden),
            "raw_before_penalty": round(raw, 1),
        },
        "gates": gates,
        "gate_failed": gate_failed,
        "trap_zeroed_no_coverage": trap_zeroed,
        "citations": {
            "must_have": {"recall": round(cite_recall, 3), "hit": cite_hit, "miss": cite_miss},
            "nice_to_have": {"recall": round(nice_recall, 3), "hit": nice_hit},
            "recency_critical": {"recall": round(rec_recall, 3), "hit": rec_hit, "miss": rec_miss},
            "forbidden_cited": forbidden,
            "provenance": {"in_freeze": in_freeze, "out_of_freeze": out_freeze},
        },
        "facts": {"must_have": fact_rows, "nice_to_have": nice_fact_rows,
                  "cov_MH": round(fact_cov, 3), "cov_nice": round(nice_fact_cov, 3)},
        "traps": trap_rows,
        "cited_all": sorted(cited, key=int),
    }


if __name__ == "__main__":
    import json, sys
    ROOT = pathlib.Path(__file__).resolve().parents[1]
    valid = freeze_issue_set(*freeze_paths(ROOT))
    for qid in ["gpu", "july", "april", "muon"]:
        q = json.load(open(ROOT / f"questions/{qid}.json"))
        gold = score(q, q["gold_prose"], valid, mode="reference")
        bp = ROOT / f"baselines/opus48-repo/{qid}.md"
        line = f"{qid:6} gold Q={gold['Q']:5.1f}"
        if bp.exists():
            body = re.sub(r"<!--.*?-->", "", bp.read_text(), flags=re.S)
            b = score(q, body, valid, mode="live")
            line += (f"   baseline Q={b['Q']:5.1f}  "
                     f"cite={b['Q_components']['cite_recall_MH']:.2f} "
                     f"fact={b['Q_components']['fact_cov_MH']:.2f} "
                     f"trap={b['Q_components']['trap_pass']:.2f} "
                     f"out-of-freeze={len(b['citations']['provenance']['out_of_freeze'])}")
        print(line)
