"""Local store of the published weekly summaries from mws.oa.dev (public, no auth).

The weekly summaries are the overview/orientation layer — a distilled, human-written
narrative of each week plus a dense set of outbound links (PRs, issues, runs, people)
that complement chunk-level search. mws.oa.dev has no machine index, but its landing
page lists every ``summaries/summary-<period>.html``, so discovery = fetch ``/``, parse
the links, download new/changed pages.
"""
from __future__ import annotations

import re
import sys
import urllib.error
import urllib.request

from . import config

_LINK_RE = re.compile(r'href="(summaries/summary-(\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2})\.html)"')


def _fetch(path: str) -> bytes:
    req = urllib.request.Request(config.MWS_URL + path, headers={"User-Agent": "mumwelt"})
    with urllib.request.urlopen(req, timeout=60, context=config.ssl_context()) as r:
        return r.read()


def list_periods() -> list[str]:
    """Local summary periods (newest first), e.g. ``2026-06-01_2026-06-07``."""
    if not config.SUMMARIES.exists():
        return []
    return sorted((p.stem.replace("summary-", "")
                   for p in config.SUMMARIES.glob("summary-*.html")), reverse=True)


def path_for(period: str):
    return config.SUMMARIES / f"summary-{period}.html"


def refresh(force: bool = False, progress: bool = True) -> dict:
    """Download new/changed summary pages from mws.oa.dev. Returns counts."""
    config.SUMMARIES.mkdir(parents=True, exist_ok=True)
    index = _fetch("/").decode("utf-8", "replace")
    config.SUMMARIES_INDEX.write_text(index, encoding="utf-8")
    pages = {m.group(2): m.group(1) for m in _LINK_RE.finditer(index)}
    new = changed = 0
    for period, rel in sorted(pages.items()):
        dest = path_for(period)
        try:
            body = _fetch("/" + rel)
        except urllib.error.HTTPError as e:
            print(f"  ({period}: HTTP {e.code})", file=sys.stderr)
            continue
        if not dest.exists():
            new += 1
        elif force or dest.read_bytes() != body:
            changed += 1
        elif not force:
            continue
        dest.write_bytes(body)
    if progress:
        print(f"  summaries: {len(pages)} published, +{new} new, ~{changed} changed",
              file=sys.stderr)
    return {"published": len(pages), "new": new, "changed": changed}


def read_text(period: str) -> str | None:
    """Return a summary as plain text (HTML stripped), or None if absent."""
    p = path_for(period)
    if not p.exists():
        return None
    html = p.read_text(encoding="utf-8", errors="replace")
    html = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"&[a-z]+;", " ", text)
    return re.sub(r"[ \t]*\n\s*\n+", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def links(period: str) -> list[str]:
    """Outbound links from a summary (the connective tissue to chase)."""
    p = path_for(period)
    if not p.exists():
        return []
    html = p.read_text(encoding="utf-8", errors="replace")
    seen, out = set(), []
    for m in re.finditer(r'href="(https?://[^"]+)"', html):
        u = m.group(1)
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out
