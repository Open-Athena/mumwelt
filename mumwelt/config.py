"""Endpoints, cache paths, and token resolution.

Two upstreams:
  - **marinmirror** (auth-gated): the search corpus + W&B run config. Bearer token = any
    GitHub token whose owner is an Open-Athena member (``read:org``).
  - **mws.oa.dev** (public): the published weekly summaries — no token.
"""
from __future__ import annotations

import os
import shutil
import ssl
import subprocess
from pathlib import Path

# --- upstreams ---------------------------------------------------------------
MARINMIRROR_URL = os.environ.get("MARINMIRROR_URL", "https://marinmirror.exe.xyz").rstrip("/")
MWS_URL = os.environ.get("MARIN_MWS_URL", "https://mws.oa.dev").rstrip("/")

# --- local cache -------------------------------------------------------------
CACHE = Path(os.environ.get("MARIN_CACHE",
                            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "marin"))
CORPUS = CACHE / "corpus-index.db"          # the downloaded search index
CORPUS_MANIFEST = CACHE / "corpus-manifest.json"   # last-seen server manifest (refresh gate)
SUMMARIES = CACHE / "summaries"             # weekly-summary HTML pages
SUMMARIES_INDEX = SUMMARIES / "index.html"  # the mws.oa.dev landing page (link source)

# --- search ------------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
RRF_K = 60                                  # reciprocal-rank-fusion constant
STALE_HOURS = float(os.environ.get("MARIN_STALE_HOURS", "24"))   # past this: a refresh is worthwhile
# Past this age the corpus is too old to trust — refresh without asking. Within it (but past
# STALE_HOURS) a refresh is optional: ask before pulling rather than auto-pulling.
MAX_AGE_HOURS = float(os.environ.get("MARIN_MAX_AGE_DAYS", "7")) * 24


def ssl_context() -> ssl.SSLContext:
    """Trust the platform CA store, falling back to certifi where it loaded nothing."""
    ctx = ssl.create_default_context()
    if not ctx.get_ca_certs():
        try:
            import certifi
            ctx.load_verify_locations(certifi.where())
        except ImportError:
            pass
    return ctx


def token() -> str | None:
    """Resolve a marinmirror bearer token: env → gh CLI → token file.

    Order: ``MARINMIRROR_TOKEN`` / ``MARIN_TOKEN`` env, then ``gh auth token``, then
    ``~/.config/marin/token``. Returns None if none found.
    """
    for var in ("MARINMIRROR_TOKEN", "MARIN_TOKEN"):
        v = os.environ.get(var)
        if v and v.strip():
            return v.strip()
    if shutil.which("gh"):
        try:
            r = subprocess.run(["gh", "auth", "token"], capture_output=True,
                               text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except Exception:
            pass
    f = Path.home() / ".config" / "marin" / "token"
    if f.exists():
        t = f.read_text().strip()
        if t:
            return t
    return None
