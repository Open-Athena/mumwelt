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
    pass


class ClientError(RuntimeError):
    pass


def _headers() -> dict:
    tok = config.token()
    if not tok:
        raise AuthError(
            "no marinmirror token found. Set MARINMIRROR_TOKEN, run `gh auth login`, or "
            "write ~/.config/marin/token. Needs a GitHub token of an Open-Athena member "
            "(read:org scope).")
    return {"Authorization": f"Bearer {tok}", "User-Agent": "mumwelt"}


def _open(path: str, headers: dict | None = None, timeout: int = 30):
    req = urllib.request.Request(config.MARINMIRROR_URL + path,
                                 headers={**_headers(), **(headers or {})})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise AuthError(f"{e.code}: not authorized — Open-Athena membership required "
                            f"(token owner must be a member, read:org)") from e
        raise ClientError(f"{e.code} {e.reason} for {path}") from e
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
