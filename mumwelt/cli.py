"""``mum`` — the Marin context CLI.

    mum status                     freshness of the local corpus + summaries vs upstream
    mum refresh [--force]          pull the latest corpus (marinmirror) + weekly summaries (mws.oa.dev)
    mum search "<q>" [...]         hybrid keyword+semantic search; cited hits
    mum search-multi "q1" "q2" ... run several queries concurrently, merged+deduped
    mum show <url|ref>             expand a hit to its context (discord window / github thread)
    mum run <project>/<run>        a W&B run's metadata + final summary numbers
    mum summaries [list|show|refresh|links]   the weekly overview narratives
    mum publish [file] --title "…"            render Markdown → LaTeX-styled HTML in a secret gist + share link
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
        if ah <= config.STALE_HOURS:
            note = ""                                   # fresh — use as-is
        elif ah <= config.MAX_AGE_HOURS:
            note = f" (STALE — built {ah / 24:.1f}d ago; refresh optional, ASK before pulling)"
        else:
            note = (f" (STALE — built {ah / 24:.1f}d ago, past {config.MAX_AGE_HOURS / 24:.0f}d max; "
                    "run `mum refresh`)")
        print(f"corpus:    {m.get('chunks', '?')} chunks, built {ah:.0f}h ago{note}")
        sp = corpus.spaces(m)
        print("  vectors:  " + ", ".join(
            f"{n} {s.get('dim')}d ({s.get('model')})" for n, s in sorted(sp.items())))
    else:
        print("corpus:    not downloaded — run `mum refresh`")
    periods = summaries.list_periods()
    print(f"summaries: {len(periods)} weeks" + (f", latest {periods[0]}" if periods else " — run `mum refresh`"))
    try:
        sm = client.manifest()
        print(f"server:    built {_age_h(sm['built_at_epoch']):.0f}h ago, "
              f"{sm['corpus_index']['bytes'] // 1048576} MB")
        _print_source_health(sm.get("sources") or {})
        _check_embed_spaces(sm)
    except (client.AuthError, client.ClientError) as e:
        print(f"server:    {e}", file=sys.stderr)


def _check_embed_spaces(sm: dict) -> None:
    """Warn if the server's index needs an encoder this client cannot load.

    Model identity is read off the corpus at query time (``corpus.spaces``), so a
    server-side model change can never silently mis-score here — but it CAN leave this
    client unable to score a space at all, which shows up as quietly worse recall. Saying
    so at refresh/status time surfaces it while the user is already thinking about the
    corpus, rather than mid-question.
    """
    advertised = sm.get("embed_spaces") or {}
    if not advertised:
        return
    try:
        from fastembed import TextEmbedding
        supported = {m["model"] for m in TextEmbedding.list_supported_models()}
    except Exception:                            # noqa: BLE001 — advisory check only
        return
    missing = sorted((n, (s or {}).get("model")) for n, s in advertised.items()
                     if (s or {}).get("model") and s["model"] not in supported)
    if not missing:
        return
    print("  ⚠ this corpus uses embedding model(s) your fastembed cannot load:",
          file=sys.stderr)
    for name, model in missing:
        print(f"      space {name}: {model}", file=sys.stderr)
    print("    Those spaces are skipped — keyword search still works, semantic search "
          "for them does not. Fix: pip install -U fastembed mumwelt", file=sys.stderr)


def _print_source_health(sources: dict) -> None:
    """Surface any source whose last sync failed, and how long it has been failing.

    Nothing on the client ever read per-source state, so a dead source was invisible
    here: the GitHub pull broke on 2026-07-05 when marin-explorer switched to OAuth, and
    for two weeks `mum status` reported a healthy corpus while every issue/PR answer came
    from a frozen snapshot. The corpus can be fresh and a source can still be dead — those
    are different facts and status has to show both.
    """
    import time as _time

    broken = [(name, s) for name, s in sorted(sources.items()) if (s or {}).get("error")]
    if not broken:
        return
    print(f"  ⚠ {len(broken)} source(s) FAILING — corpus is serving stale data for these:",
          file=sys.stderr)
    for name, s in broken:
        last_ok = s.get("last_ok_at")
        age = (f", last OK {(_time.time() - last_ok) / 86400:.1f}d ago"
               if last_ok else ", no successful sync on record")
        print(f"      {name}: {str(s.get('error'))[:80]}{age}", file=sys.stderr)


def cmd_refresh(a):
    # 1. corpus (manifest-gated)
    try:
        sm = client.manifest()
        _check_embed_spaces(sm)
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
    src = _csv(a.source)
    kw = dict(kind=_csv(a.kind), since=a.since, until=a.until, fts_only=a.fts_only,
              vec_text=getattr(a, "vec_text", None))
    hits = corpus.search(a.query, k=a.k, source=src, **kw)

    # The code lanes run alongside every unfiltered search rather than competing with it.
    # An explicit --source already says what the caller wants, so don't second-guess it.
    code = branches = []
    lanes = src is None and not a.no_code
    if lanes:
        code = corpus.search_code(a.query, k=a.code_k, branches=False, **kw)
        branches = corpus.search_code(a.query, k=a.branch_k, branches=True, **kw)

    if a.json:
        print(json.dumps({"hits": hits, "code": code, "branches": branches}
                         if lanes else hits, indent=2))
        return
    _print_hits(hits)
    if lanes:
        # Two lanes, not one: main is how it works, a branch is what someone is trying.
        # Cite accordingly — a branch symbol is not evidence of current behavior.
        print(f"--- code · main ({len(code)}) "
              f"{'—' if code else '— nothing relevant found'} ---\n")
        if code:
            _print_hits(code)
        print(f"--- code · in-flight branches ({len(branches)}) "
              f"{'—' if branches else '— nothing relevant found'} ---\n")
        if branches:
            _print_hits(branches)


def cmd_search_multi(a):
    _require_corpus()
    if not a.fts_only:
        # Warm only the encoders this fan-out will actually use: an unfiltered
        # search-multi is prose-only now, so it must not pay to load the code model.
        _src = _csv(a.source)
        corpus.warm(source=_src,
                    exclude_source=None if _src else corpus.code_sources())
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


def cmd_context(a):
    """One-shot bulk context: search (one or more queries, optional HyDE) → expand the top
    hits to FULL threads → optionally follow one hop of #refs → dedupe → pack. All in a single
    process, so 'bring 100k in' costs sqlite reads (~ms each) not N cold subprocesses."""
    _require_corpus()
    import re as _re
    vt = getattr(a, "vec_text", None)
    merged = {}
    for q in a.query:
        for h in corpus.search(q, k=a.k, source=_csv(a.source), kind=_csv(a.kind),
                               since=a.since, until=a.until, vec_text=vt):
            cur = merged.get(h["url"])
            if cur is None or h["score"] > cur["score"]:
                merged[h["url"]] = h
    ranked = sorted(merged.values(), key=lambda h: -h["score"])

    shown, blocks, stats = set(), [], {"chars": 0}

    def add(target):
        res = context.show(target, window=a.window)
        if not res:
            return []
        foc = res["focal"]
        # base_ref so parts of one split body collapse to a single thread block
        key = (foc["source"], context.base_ref(foc.get("parent") or foc.get("ref")))
        if key in shown:
            return []
        shown.add(key)
        lines = [f"=== {foc.get('title') or foc['source']}  ({foc['url']}) ==="]
        refs = []
        ctx_chunks = res["context"][-a.tail:] if getattr(a, "tail", 0) else res["context"]
        for c in ctx_chunks:
            t = (c.get("text") or "").strip()
            if not t:
                continue
            lines.append(f"[{(c.get('date') or '')[:10]} {c.get('author') or ''}] {t}")
            refs += _re.findall(r"#(\d{3,6})\b", t)
        block = "\n".join(lines)[: a.per_thread_chars]
        blocks.append(block)
        stats["chars"] += len(block)
        return refs

    followups = []
    for h in ranked[: a.expand]:
        if stats["chars"] >= a.max_chars:
            break
        followups += add(h["url"])
    if a.follow_links:
        for r in dict.fromkeys(followups):          # unique, order-preserving
            if stats["chars"] >= a.max_chars:
                break
            add(r)                                    # context.show accepts a bare ref

    out = "\n\n".join(blocks)[: a.max_chars]
    if getattr(a, "toc", False):
        idx = "\n".join(b.split("\n", 1)[0] for b in blocks)
        out = f"THREAD INDEX ({len(blocks)} threads):\n{idx}\n\n=====\n\n" + out
    if a.json:
        print(json.dumps({"queries": a.query, "threads": len(blocks),
                          "chars": len(out), "approx_tokens": len(out) // 4,
                          "context": out}, indent=2))
    else:
        print(out)
    print(f"--- mum context: {len(blocks)} threads, {len(out)} chars "
          f"(~{len(out)//4} tokens) ---", file=sys.stderr)


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


# ---- publish ----------------------------------------------------------------

def cmd_publish(a):
    from . import publish
    publish.cmd_publish(a)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mum", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status").set_defaults(fn=cmd_status)

    p = sub.add_parser("refresh"); p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_refresh)

    # k=50: gold-citation recall on the eval harness runs 32% @10 -> 45% @20 -> 58% @50,
    # and raising k is close to free because the vector leg already scores every chunk
    # regardless — a larger k only lengthens the ranked list it returns.
    K_DEFAULT = 50

    def add_search_flags(p):
        p.add_argument("-k", type=int, default=K_DEFAULT)
        p.add_argument("--source"); p.add_argument("--kind")
        p.add_argument("--since"); p.add_argument("--until")
        p.add_argument("--fts-only", action="store_true")
        p.add_argument("--vec-text", action="append", metavar="DOC", dest="vec_text",
                       help="HyDE: hypothetical-answer doc to drive the vector leg; "
                            "repeat for N-doc averaging. FTS leg still uses the literal query.")
        p.add_argument("--json", action="store_true")
        # The code lane is separate, not fused: code never competes for the prose slots,
        # so asking for it costs those results nothing.
        p.add_argument("--code-k", type=int, default=25,
                       help="hits in the code·main lane (default 25)")
        p.add_argument("--branch-k", type=int, default=10,
                       help="hits in the code·in-flight-branches lane (default 10)")
        p.add_argument("--no-code", action="store_true",
                       help="skip both code lanes entirely")

    p = sub.add_parser("search"); p.add_argument("query"); add_search_flags(p)
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("search-multi"); p.add_argument("queries", nargs="+")
    # --total is the cap on the MERGED set, so it has to scale with k or the fan-out
    # retrieves 50 per query and then throws away all but 20 of the union.
    p.add_argument("-k", type=int, default=K_DEFAULT, help="candidates per query")
    p.add_argument("--total", type=int, default=60, help="cap on the merged, deduped set")
    p.add_argument("--source"); p.add_argument("--kind")
    p.add_argument("--fts-only", action="store_true"); p.add_argument("--json", action="store_true")
    p.add_argument("--since"); p.add_argument("--until")
    p.set_defaults(fn=cmd_search_multi)

    p = sub.add_parser("show"); p.add_argument("target")
    p.add_argument("--window", type=int, default=context.WINDOW)
    p.add_argument("--json", action="store_true"); p.set_defaults(fn=cmd_show)

    p = sub.add_parser("context", help="one-shot bulk: search + expand top hits to full "
                                       "threads + follow #refs; pack a big cited context blob")
    p.add_argument("query", nargs="+")
    p.add_argument("-k", type=int, default=20, help="candidates per query")
    p.add_argument("--expand", type=int, default=12, help="how many top hits to expand to full threads")
    p.add_argument("--follow-links", type=int, default=0, help="hops of #ref chasing (0 or 1)")
    p.add_argument("--per-thread-chars", type=int, default=12000, help="cap per thread (breadth)")
    p.add_argument("--tail", type=int, default=0, help="keep only the last N chunks per thread (close-out)")
    p.add_argument("--toc", action="store_true", help="prepend a thread index (attention anchor)")
    p.add_argument("--max-chars", type=int, default=400000)
    p.add_argument("--window", type=int, default=context.WINDOW)
    p.add_argument("--source"); p.add_argument("--kind")
    p.add_argument("--since"); p.add_argument("--until")
    p.add_argument("--vec-text", action="append", dest="vec_text")
    p.add_argument("--json", action="store_true"); p.set_defaults(fn=cmd_context)

    p = sub.add_parser("run"); p.add_argument("run"); p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("summaries")
    p.add_argument("sub", nargs="?", choices=["list", "show", "refresh", "links"])
    p.add_argument("period", nargs="?"); p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_summaries)

    p = sub.add_parser("publish", help="render Markdown research → LaTeX-styled HTML in a "
                                       "secret gist; print a shareable htmlpreview link")
    p.add_argument("file", nargs="?", help="Markdown file (default: read stdin)")
    p.add_argument("--title", help="document title (default: leading # H1, else 'Research report')")
    p.add_argument("--author", help="author line (default: your gh login)")
    p.add_argument("--description", help="gist description (default: the title)")
    p.add_argument("--filename", help="HTML filename in the gist (default: slug of title)")
    p.add_argument("--public", action="store_true", help="public gist (default: secret/unlisted)")
    p.add_argument("--no-date", action="store_true", help="omit today's date from the header")
    p.add_argument("--open", action="store_true", help="open the preview link in a browser")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_publish)

    p = sub.add_parser("skills")
    p.add_argument("sub", nargs="?", choices=["list", "print", "install"], default="list")
    p.add_argument("dest", nargs="?")
    p.set_defaults(fn=cmd_skills)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
