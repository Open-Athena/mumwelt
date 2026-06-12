"""Publish a research report (Markdown in → LaTeX-styled HTML) to a private GitHub gist.

The flow agents care about: hand `mum publish` the Markdown you already wrote (every claim
cited with a URL), and it (1) renders it to a clean, LaTeX-looking HTML document — serif
typeset body via latex.css, math via MathJax — (2) creates a **secret** gist via the `gh`
CLI, and (3) prints a one-click **htmlpreview.github.io** link that renders that HTML in a
browser, for easy sharing.

No third-party Python deps: Markdown→HTML uses ``pandoc`` if it's on PATH (GFM, best
fidelity) and otherwise a small built-in converter covering the subset research reports
use (headings, lists, blockquotes, code, links, bold/italic, rules, math passthrough).

Auth/transport is ``gh`` (GitHub CLI) — already how this toolkit gets its token. A "secret"
gist is unlisted, **not** truly private: anyone with the link (or the htmlpreview link) can
read it, so don't publish secrets.
"""
from __future__ import annotations

import datetime
import html as _html
import json
import re
import shutil
import subprocess
import sys
import webbrowser

# CDNs the rendered page pulls in at view time (so htmlpreview shows real typesetting).
LATEX_CSS = "https://cdn.jsdelivr.net/npm/latex.css/style.min.css"
MATHJAX = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
HTMLPREVIEW = "https://htmlpreview.github.io/?"


# ---- Markdown → HTML body ---------------------------------------------------

def _md_to_body(md: str) -> str:
    """Markdown → HTML fragment. Prefer pandoc; fall back to the built-in converter."""
    if shutil.which("pandoc"):
        try:
            r = subprocess.run(
                ["pandoc", "--from=gfm", "--to=html5", "--mathjax"],
                input=md, capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                return r.stdout
            print(f"publish: pandoc failed ({r.stderr.strip()[:120]}); using built-in "
                  "converter", file=sys.stderr)
        except Exception as e:  # pragma: no cover - pandoc present but unhappy
            print(f"publish: pandoc error ({e}); using built-in converter", file=sys.stderr)
    return _fallback_md(md)


_INLINE_CODE = re.compile(r"`([^`]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC = re.compile(r"(?<![\*\w])[*_]([^*_\n]+)[*_](?![\*\w])")
_BARE_URL = re.compile(r"(?<![\"(>=])\bhttps?://[^\s<>)\]]+")


def _inline(text: str) -> str:
    """Escape HTML, then apply inline Markdown. Code spans/links are protected first."""
    stash: list[str] = []

    def keep(htmlfrag: str) -> str:
        stash.append(htmlfrag)
        return f"\x00{len(stash) - 1}\x00"

    text = _INLINE_CODE.sub(lambda m: keep(f"<code>{_html.escape(m.group(1))}</code>"), text)
    text = _LINK.sub(
        lambda m: keep(f'<a href="{_html.escape(m.group(2), quote=True)}">'
                       f"{_html.escape(m.group(1))}</a>"), text)
    text = _html.escape(text)
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    text = _ITALIC.sub(r"<em>\1</em>", text)
    text = _BARE_URL.sub(lambda m: f'<a href="{m.group(0)}">{m.group(0)}</a>', text)
    return re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)


def _fallback_md(md: str) -> str:
    """Dependency-free converter for the Markdown subset research reports use."""
    out: list[str] = []
    lines = md.replace("\r\n", "\n").split("\n")
    i, n = 0, len(lines)
    list_stack: list[str] = []  # "ul" / "ol"

    def close_lists(level: int = 0):
        while len(list_stack) > level:
            out.append(f"</{list_stack.pop()}>")

    while i < n:
        line = lines[i]
        # fenced code block
        if line.strip().startswith("```"):
            close_lists()
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(_html.escape(lines[i]))
                i += 1
            i += 1
            out.append("<pre><code>" + "\n".join(buf) + "</code></pre>")
            continue
        # display math $$ ... $$ (passed through untouched for MathJax)
        if line.strip() == "$$":
            close_lists()
            i += 1
            buf = []
            while i < n and lines[i].strip() != "$$":
                buf.append(lines[i])
                i += 1
            i += 1
            out.append("$$" + "\n".join(buf) + "$$")
            continue
        stripped = line.strip()
        if not stripped:
            close_lists()
            i += 1
            continue
        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", stripped):
            close_lists()
            out.append("<hr>")
            i += 1
            continue
        h = re.match(r"^(#{1,6})\s+(.*)$", line)
        if h:
            close_lists()
            lvl = len(h.group(1))
            out.append(f"<h{lvl}>{_inline(h.group(2).strip())}</h{lvl}>")
            i += 1
            continue
        if stripped.startswith(">"):
            close_lists()
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].lstrip())
                i += 1
            out.append(f"<blockquote>{_inline(' '.join(buf))}</blockquote>")
            continue
        m_ul = re.match(r"^[-*+]\s+(.*)$", stripped)
        m_ol = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if m_ul or m_ol:
            want = "ul" if m_ul else "ol"
            if not list_stack or list_stack[-1] != want:
                close_lists()
                list_stack.append(want)
                out.append(f"<{want}>")
            out.append(f"<li>{_inline((m_ul or m_ol).group(1))}</li>")
            i += 1
            continue
        # paragraph (gather until blank line / block start)
        close_lists()
        buf = [stripped]
        i += 1
        while i < n and lines[i].strip() and not re.match(
                r"^(#{1,6}\s|>|[-*+]\s|\d+[.)]\s|```|\$\$|-{3,}$|\*{3,}$|_{3,}$)",
                lines[i].strip()):
            buf.append(lines[i].strip())
            i += 1
        out.append(f"<p>{_inline(' '.join(buf))}</p>")
    close_lists()
    return "\n".join(out)


# ---- provenance footer ------------------------------------------------------

# Sentinel a writeup uses to mark the start of its provenance trailer (data freshness +
# query trace, per the marin-research skill). Everything after it renders as a muted
# <footer>. It's an HTML comment, so it's invisible if the Markdown is shown raw, too.
PROVENANCE_MARKER = "<!--provenance-->"
_TRAILING_RULE = re.compile(r"(?:\n[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*)+\s*$")


def _split_provenance(md: str) -> tuple[str, str]:
    """Split off the provenance trailer at the sentinel. Returns (body_md, footer_md).

    The body's trailing thematic break (the ``---`` the writeup puts before the trailer)
    is dropped — the footer supplies its own rule via CSS, so we don't double it up.
    """
    idx = md.find(PROVENANCE_MARKER)
    if idx == -1:
        return md, ""
    body = _TRAILING_RULE.sub("\n", md[:idx])
    return body, md[idx + len(PROVENANCE_MARKER):]


# ---- HTML document ----------------------------------------------------------

# Muted, small-type trailer set off by a hairline rule. Links/blockquote inherit the
# grey so the whole block reads as a footnote, not body copy.
PROVENANCE_CSS = """\
footer.provenance { margin-top: 3em; padding-top: 0.8em; border-top: 1px solid #ddd;
  font-size: 0.8em; line-height: 1.5; color: #888; }
footer.provenance a { color: #888; }
footer.provenance p { margin: 0.3em 0; }
footer.provenance blockquote { margin: 0; padding: 0; border: none; color: inherit; }"""


def render_html(md: str, title: str, author: str, date: str) -> str:
    body_md, prov_md = _split_provenance(md)
    body = _md_to_body(body_md)
    footer = ""
    if prov_md.strip():
        footer = f'\n<footer class="provenance">\n{_md_to_body(prov_md)}\n</footer>'
    author_line = f'<p class="author">{_html.escape(author)}'
    author_line += f"<br>{_html.escape(date)}</p>" if date else "</p>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)}</title>
<link rel="stylesheet" href="{LATEX_CSS}">
<style>
{PROVENANCE_CSS}
</style>
<script>
window.MathJax = {{ tex: {{
  inlineMath: [['$','$'], ['\\\\(','\\\\)']],
  displayMath: [['$$','$$'], ['\\\\[','\\\\]']]
}} }};
</script>
<script id="MathJax-script" async src="{MATHJAX}"></script>
</head>
<body class="libertinus">
<header>
<h1>{_html.escape(title)}</h1>
{author_line}
</header>
{body}{footer}
</body>
</html>
"""


# ---- gist -------------------------------------------------------------------

def _gh_login() -> str:
    try:
        r = subprocess.run(["gh", "api", "user", "-q", ".login"],
                           capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return ""


def create_gist(filename: str, content: str, description: str, public: bool) -> dict:
    """Create a gist via `gh api` and return the parsed response (has files[].raw_url)."""
    if not shutil.which("gh"):
        sys.exit("publish: `gh` (GitHub CLI) not found — install it and `gh auth login`.")
    payload = json.dumps({
        "description": description,
        "public": public,
        "files": {filename: {"content": content}},
    })
    r = subprocess.run(["gh", "api", "-X", "POST", "/gists", "--input", "-"],
                       input=payload, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        sys.exit(f"publish: gist creation failed — {r.stderr.strip()[:300]}")
    return json.loads(r.stdout)


# ---- command ----------------------------------------------------------------

def _slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return s[:60] or "research"


def cmd_publish(a) -> None:
    raw = sys.stdin.read() if (not a.file or a.file == "-") else open(a.file, encoding="utf-8").read()
    if not raw.strip():
        sys.exit("publish: no Markdown on stdin / in file.")

    # Lift a leading H1 into the document header (avoids a title shown twice); use its
    # text as the title only when --title wasn't given.
    title = a.title
    m = re.search(r"^#\s+(.+?)\s*$", raw, re.MULTILINE)
    if m and raw[:m.start()].strip() == "":
        title = title or m.group(1).strip()
        raw = raw[:m.start()] + raw[m.end():]
    title = title or "Research report"

    author = a.author or _gh_login() or "Anonymous"
    date = "" if a.no_date else datetime.date.today().isoformat()
    filename = a.filename or f"{_slug(title)}.html"
    if not filename.endswith(".html"):
        filename += ".html"

    doc = render_html(raw, title, author, date)
    gist = create_gist(filename, doc, a.description or title, public=a.public)

    raw_url = gist["files"][filename]["raw_url"]
    preview = HTMLPREVIEW + raw_url
    result = {"gist": gist["html_url"], "raw": raw_url, "preview": preview,
              "public": bool(gist.get("public")), "filename": filename}

    if a.json:
        print(json.dumps(result, indent=2))
    else:
        kind = "public" if result["public"] else "secret (unlisted — anyone with the link can view)"
        print(f"published {kind} gist:")
        print(f"  gist:    {result['gist']}")
        print(f"  preview: {preview}")
    if a.open:
        webbrowser.open(preview)
