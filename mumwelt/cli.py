"""``mum`` — the Marin context CLI.

    mum status                     freshness of the local corpus + summaries vs upstream
    mum refresh [--force]          pull the latest corpus (marinmirror) + weekly summaries (mws.oa.dev)
    mum search "<q>" [...]         hybrid keyword+semantic search; cited hits
    mum search-multi "q1" "q2" ... run several queries concurrently, merged+deduped
    mum show <url|ref>             expand a hit to its context (discord window / github thread)
    mum run <project>/<run>        a W&B run's metadata + final summary numbers
    mum summaries [list|show|refresh|links]   the weekly overview narratives
    mum skills [list|print|install]           the agent skills (Claude + portable)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from . import client, config, context, corpus, summaries


def _csv(v):
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


def _age_h(epoch) -> float:
    return (time.time() - int(epoch or 0)) / 3600


# ---- status / refresh -------------------------------------------------------

def cmd_status(a):
    if corpus.exists():
        m = corpus.meta()
        ah = _age_h(m.get("built_at_epoch"))
        stale = " (STALE — run `mum refresh`)" if ah > config.STALE_HOURS else ""
        print(f"corpus:    {m.get('chunks', '?')} chunks, built {ah:.0f}h ago{stale}")
    else:
        print("corpus:    not downloaded — run `mum refresh`")
    periods = summaries.list_periods()
    print(f"summaries: {len(periods)} weeks" + (f", latest {periods[0]}" if periods else " — run `mum refresh`"))
    try:
        sm = client.manifest()
        print(f"server:    built {_age_h(sm['built_at_epoch']):.0f}h ago, "
              f"{sm['corpus_index']['bytes'] // 1048576} MB")
    except (client.AuthError, client.ClientError) as e:
        print(f"server:    {e}", file=sys.stderr)


def cmd_refresh(a):
    # 1. corpus (manifest-gated)
    try:
        sm = client.manifest()
        want = sm["corpus_index"]["sha256"]
        have = None
        if config.CORPUS_MANIFEST.exists():
            try:
                have = json.loads(config.CORPUS_MANIFEST.read_text())["corpus_index"]["sha256"]
            except Exception:
                have = None
        if a.force or not corpus.exists() or want != have:
            client.download_corpus(expected_sha=want)
            config.CORPUS_MANIFEST.write_text(json.dumps(sm))
            print(f"corpus:    updated → {corpus.meta().get('chunks', '?')} chunks")
        else:
            print("corpus:    already current")
    except (client.AuthError, client.ClientError) as e:
        print(f"corpus:    SKIPPED — {e}", file=sys.stderr)
    # 2. weekly summaries (public, index-diff)
    try:
        summaries.refresh(force=a.force)
    except Exception as e:
        print(f"summaries: SKIPPED — {e}", file=sys.stderr)


# ---- search -----------------------------------------------------------------

def _require_corpus():
    if not corpus.exists():
        sys.exit("no local corpus — run `mum refresh` first.")


def _print_hits(hits):
    if not hits:
        print("(no results)")
        return
    for i, r in enumerate(hits, 1):
        who = f" · {r['author']}" if r.get("author") else ""
        print(f"{i}. [{r['source']}/{r['kind']}] {r['title']}  ({r['via']} · {r['date']}{who})")
        print(f"   {r['url']}")
        print(f"   {r['snippet']}\n")


def cmd_search(a):
    _require_corpus()
    hits = corpus.search(a.query, k=a.k, source=_csv(a.source), kind=_csv(a.kind),
                         since=a.since, until=a.until, fts_only=a.fts_only)
    print(json.dumps(hits, indent=2)) if a.json else _print_hits(hits)


def cmd_search_multi(a):
    _require_corpus()
    corpus._model() if not a.fts_only else None   # warm the model once before fan-out
    kw = dict(k=a.k, source=_csv(a.source), kind=_csv(a.kind), fts_only=a.fts_only)
    with ThreadPoolExecutor(max_workers=min(8, len(a.queries))) as ex:
        results = list(ex.map(lambda q: corpus.search(q, **kw), a.queries))
    # merge: dedupe by url, keep best score, record which sub-queries surfaced it
    merged: dict[str, dict] = {}
    for q, hits in zip(a.queries, results):
        for h in hits:
            cur = merged.get(h["url"])
            if cur is None:
                h = {**h, "queries": [q]}
                merged[h["url"]] = h
            else:
                cur["queries"].append(q)
                cur["score"] = max(cur["score"], h["score"])
    out = sorted(merged.values(), key=lambda h: -h["score"])[:a.total]
    print(json.dumps(out, indent=2)) if a.json else _print_hits(out)


def cmd_show(a):
    _require_corpus()
    res = context.show(a.target, window=a.window)
    if not res:
        sys.exit(f"not found in corpus: {a.target}")
    if a.json:
        print(json.dumps(res, indent=2))
        return
    foc = res["focal"]["url"]
    print(f"# {res['kind']} context around {foc}\n")
    for c in res["context"]:
        mark = "▶" if c["url"] == foc else " "
        when = (c["date"] or "")[:16].replace("T", " ")
        who = c.get("author") or ""
        body = (c["text"] or "").strip().replace("\n", " ")
        print(f"{mark} [{when}] {who}: {body[:300]}")


def cmd_run(a):
    target = a.run
    import re as _re
    m = _re.search(r"/([^/]+)/runs/([^/?#]+)", target)
    if m:
        project, run = m.group(1), m.group(2)
    elif "/" in target:
        project, run = target.split("/", 1)
    else:
        sys.exit("usage: mum run <project>/<run>  (or a wandb.ai run URL)")
    try:
        d = client.wandb_config(project, run)
    except (client.AuthError, client.ClientError) as e:
        sys.exit(str(e))
    if a.json:
        print(json.dumps(d, indent=2))
        return
    summ = d.get("summary") or {}
    print(f"{d.get('name')}  [{d.get('project')}]  state={d.get('state')}")
    print(f"  url: {d.get('url')}")
    finals = {k: summ[k] for k in (
        "train/loss", "loss", "eval/bpb", "eval/loss", "effective_estimated_mfu",
        "throughput/total_tokens", "parameter_count", "_step", "_runtime")
        if isinstance(summ.get(k), (int, float))}
    if finals:
        print("  final numbers:")
        for k, v in finals.items():
            print(f"    {k} = {v}")


# ---- summaries --------------------------------------------------------------

def cmd_summaries(a):
    sub = a.sub or "list"
    if sub == "refresh":
        summaries.refresh(force=a.force)
    elif sub == "list":
        periods = summaries.list_periods()
        if not periods:
            print("no summaries — run `mum summaries refresh` (or `mum refresh`)")
        for p in periods:
            print(f"  {p}  {config.MWS_URL}/summaries/summary-{p}.html")
    elif sub == "show":
        period = _resolve_period(a.period)
        txt = summaries.read_text(period) if period else None
        if not txt:
            sys.exit("summary not found — `mum summaries list`")
        print(txt)
    elif sub == "links":
        period = _resolve_period(a.period)
        for u in summaries.links(period) if period else []:
            print(u)


def _resolve_period(period):
    periods = summaries.list_periods()
    if not periods:
        return None
    if not period or period == "latest":
        return periods[0]
    return next((p for p in periods if p.startswith(period) or period in p), None)


# ---- skills -----------------------------------------------------------------

def cmd_skills(a):
    from . import skills_pkg
    skills_pkg.dispatch(a)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mum", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status").set_defaults(fn=cmd_status)

    p = sub.add_parser("refresh"); p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_refresh)

    def add_search_flags(p):
        p.add_argument("-k", type=int, default=10)
        p.add_argument("--source"); p.add_argument("--kind")
        p.add_argument("--since"); p.add_argument("--until")
        p.add_argument("--fts-only", action="store_true")
        p.add_argument("--json", action="store_true")

    p = sub.add_parser("search"); p.add_argument("query"); add_search_flags(p)
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("search-multi"); p.add_argument("queries", nargs="+")
    p.add_argument("-k", type=int, default=8); p.add_argument("--total", type=int, default=20)
    p.add_argument("--source"); p.add_argument("--kind")
    p.add_argument("--fts-only", action="store_true"); p.add_argument("--json", action="store_true")
    p.add_argument("--since"); p.add_argument("--until")
    p.set_defaults(fn=cmd_search_multi)

    p = sub.add_parser("show"); p.add_argument("target")
    p.add_argument("--window", type=int, default=context.WINDOW)
    p.add_argument("--json", action="store_true"); p.set_defaults(fn=cmd_show)

    p = sub.add_parser("run"); p.add_argument("run"); p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("summaries")
    p.add_argument("sub", nargs="?", choices=["list", "show", "refresh", "links"])
    p.add_argument("period", nargs="?"); p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_summaries)

    p = sub.add_parser("skills")
    p.add_argument("sub", nargs="?", choices=["list", "print", "install"], default="list")
    p.add_argument("dest", nargs="?")
    p.set_defaults(fn=cmd_skills)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
