"""HTTP client for marinmirror (auth-gated): manifest, corpus download, W&B config."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import config


class AuthError(RuntimeError):
    """Authorization failed. ``status`` separates the two cases a user can act on:

    ``None``    — no token was found locally, so nothing was sent. The fix is local.
    401 / 403   — a token was sent and the server refused it. The fix is upstream, and
                  ``detail`` carries the server's own words for why.
    """

    def __init__(self, msg: str, *, status: int | None = None, detail: str | None = None):
        super().__init__(msg)
        self.status = status
        self.detail = detail


class ClientError(RuntimeError):
    pass


def _headers() -> dict:
    tok = config.token()
    if not tok:
        raise AuthError(
            "no marinmirror token found. Set MARINMIRROR_TOKEN, run `gh auth login`, or "
            "write ~/.config/marin/token.")
    return {"Authorization": f"Bearer {tok}", "User-Agent": "mumwelt"}


def _error_detail(e: urllib.error.HTTPError) -> str | None:
    """The server's own explanation of a failure, if it sent one.

    marinmirror answers with FastAPI-shaped JSON (``{"detail": ...}``), and its message is
    strictly better than anything inferable here: it knows *why* a token was refused,
    where this client only knows that it was. Relaying it also means the server can change
    the authorization story — new states, new wording — and users hear about it without a
    client release. Best-effort by construction: a body that is missing, truncated, or not
    JSON must never mask the HTTP error already being reported.
    """
    try:
        raw = e.read().decode("utf-8", "replace").strip()
    except Exception:                              # noqa: BLE001 — advisory read only
        return None
    if not raw:
        return None
    try:
        body = json.loads(raw)
    except ValueError:
        return raw[:300]
    if isinstance(body, dict):
        for key in ("detail", "message", "error"):
            v = body.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()[:300]
            if v is not None:
                return json.dumps(v)[:300]
    return raw[:300]


def _open(path: str, headers: dict | None = None, timeout: int = 30):
    req = urllib.request.Request(config.MARINMIRROR_URL + path,
                                 headers={**_headers(), **(headers or {})})
    try:
        return urllib.request.urlopen(req, timeout=timeout, context=config.ssl_context())
    except urllib.error.HTTPError as e:
        detail = _error_detail(e)
        suffix = f" — {detail}" if detail else ""
        if e.code in (401, 403):
            raise AuthError(f"{e.code}: not authorized{suffix}",
                            status=e.code, detail=detail) from e
        raise ClientError(f"{e.code} {e.reason} for {path}{suffix}") from e
    except urllib.error.URLError as e:
        raise ClientError(f"cannot reach {config.MARINMIRROR_URL}: {e.reason}") from e


def get_json(path: str) -> dict:
    with _open(path) as r:
        return json.load(r)


def manifest() -> dict:
    return get_json("/manifest.json")


def wandb_config(project: str, run: str) -> dict:
    p, r = urllib.parse.quote(project, safe=""), urllib.parse.quote(run, safe="")
    return get_json(f"/wandb/{p}/{r}/config")


def download_corpus(expected_sha: str | None = None, progress: bool = True) -> None:
    """Stream /corpus-index.db to a temp file, verify sha256, atomic-swap into place."""
    dest = config.CORPUS
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".db.tmp")
    h = hashlib.sha256()
    with _open("/corpus-index.db", timeout=600) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            blk = r.read(1 << 20)
            if not blk:
                break
            f.write(blk)
            h.update(blk)
            done += len(blk)
            if progress and total and done % (16 << 20) < (1 << 20):
                print(f"\r  downloading corpus {done // 1048576}/{total // 1048576} MB",
                      end="", file=sys.stderr, flush=True)
    if progress:
        print(file=sys.stderr)
    if expected_sha and h.hexdigest() != expected_sha:
        tmp.unlink(missing_ok=True)
        raise ClientError("sha256 mismatch after download — try again")
    os.replace(tmp, dest)
