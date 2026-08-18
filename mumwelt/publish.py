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

MATHJAX = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
HTMLPREVIEW = "https://htmlpreview.github.io/?"


# ---- KaTeX pre-rendering ----------------------------------------------------

_DISPLAY_MATH = re.compile(r"\$\$\s*\n?(.+?)\n?\s*\$\$", re.DOTALL)
_INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$|\s)(.+?)(?<!\$|\s)\$(?!\$)")

_katex_css_cache: str | None = None


def _katex_css() -> str:
    """Return KaTeX CSS with @font-face rules stripped (fallback to Times)."""
    global _katex_css_cache
    if _katex_css_cache is not None:
        return _katex_css_cache
    _NODE_READ_CSS = (
        "const fs=require('fs'),p=require('path');"
        "try{process.stdout.write(fs.readFileSync("
        "require.resolve('katex/dist/katex.min.css'),'utf8'))}catch(e){"
        "const d=process.env.PATH.split(p.delimiter)"
        ".find(d=>fs.existsSync(p.join(d,'katex')));"
        "if(d)process.stdout.write(fs.readFileSync("
        "p.join(d,'..','katex','dist','katex.min.css'),'utf8'))}"
    )
    try:
        r = subprocess.run(
            ["npx", "--yes", "-p", "katex", "node", "-e", _NODE_READ_CSS],
            capture_output=True, text=True, timeout=15)
        if r.returncode == 0 and r.stdout:
            css = re.sub(r"@font-face\{[^}]+\}", "", r.stdout)
            _katex_css_cache = css
            return css
    except Exception:
        pass
    _katex_css_cache = ""
    return ""


def _katex_render(latex: str, display: bool = False) -> str:
    """Render a LaTeX string to HTML via the KaTeX CLI. Returns raw HTML."""
    cmd = ["npx", "--yes", "katex"]
    if display:
        cmd.append("--display-mode")
    try:
        r = subprocess.run(cmd, input=latex, capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    esc = _html.escape(latex)
    return f'<code class="math-fallback">{esc}</code>'


def _prerender_math(md: str) -> tuple[str, list[str]]:
    """Replace $$ and $ math with placeholders; return (md, stash).

    Placeholders survive the Markdown→HTML pass (including HTML-escaping).
    Call ``_restore_math(html, stash)`` after conversion to inject the KaTeX.
    """
    if not shutil.which("npx"):
        return md, []
    stash: list[str] = []

    def _keep(html: str) -> str:
        stash.append(html)
        return f"\x02MATH{len(stash) - 1}\x02"

    def _display(m):
        return _keep(_katex_render(m.group(1).strip(), display=True))

    def _inline_m(m):
        return _keep(_katex_render(m.group(1).strip(), display=False))

    md = _DISPLAY_MATH.sub(_display, md)
    md = _INLINE_MATH.sub(_inline_m, md)
    return md, stash


def _restore_math(html: str, stash: list[str]) -> str:
    """Replace math placeholders with pre-rendered KaTeX HTML."""
    for i, rendered in enumerate(stash):
        html = html.replace(f"\x02MATH{i}\x02", rendered)
    return html


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
_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC = re.compile(r"(?<![\*\w])[*_]([^*_\n]+)[*_](?![\*\w])")
_BARE_URL = re.compile(r"(?<![\"(>=])\bhttps?://[^\s<>)\]]+")
_BLOCK_IMAGE = re.compile(r"^!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)\s*$")


def _inline(text: str) -> str:
    """Escape HTML, then apply inline Markdown. Code spans/links are protected first."""
    stash: list[str] = []

    def keep(htmlfrag: str) -> str:
        stash.append(htmlfrag)
        return f"\x00{len(stash) - 1}\x00"

    text = _INLINE_CODE.sub(lambda m: keep(f"<code>{_html.escape(m.group(1))}</code>"), text)
    text = _IMAGE.sub(
        lambda m: keep(f'<img src="{_html.escape(m.group(2), quote=True)}"'
                       f' alt="{_html.escape(m.group(1), quote=True)}">'), text)
    text = _LINK.sub(
        lambda m: keep(f'<a href="{_html.escape(m.group(2), quote=True)}">'
                       f"{_html.escape(m.group(1))}</a>"), text)
    text = _html.escape(text)
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    text = _ITALIC.sub(r"<em>\1</em>", text)
    text = _BARE_URL.sub(lambda m: f'<a href="{m.group(0)}">{m.group(0)}</a>', text)
    for i in range(len(stash) - 1, -1, -1):
        text = text.replace(f"\x00{i}\x00", stash[i])
    return text


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
        m_img = _BLOCK_IMAGE.match(stripped)
        if m_img:
            close_lists()
            alt = _html.escape(m_img.group(1), quote=True)
            src = _html.escape(m_img.group(2), quote=True)
            cap = f"<figcaption>{_html.escape(m_img.group(1))}</figcaption>" if m_img.group(1) else ""
            out.append(f'<figure><img src="{src}" alt="{alt}">{cap}</figure>')
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
        if re.match(r"^<(table|div|figure|details|section|nav|aside|header|footer|form|fieldset|pre)\b", stripped, re.IGNORECASE):
            close_lists()
            tag = re.match(r"^<(\w+)", stripped).group(1).lower()
            buf = [line]
            depth = 1
            i += 1
            while i < n and depth > 0:
                depth += len(re.findall(rf"<{tag}\b", lines[i], re.IGNORECASE))
                depth -= len(re.findall(rf"</{tag}>", lines[i], re.IGNORECASE))
                buf.append(lines[i])
                i += 1
            out.append("\n".join(buf))
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
# query trace, per the mumwelt skill). Everything after it renders as a muted
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


# ---- hover metadata --------------------------------------------------------

HOVER_MARKER = "<!--hover-->"


def _extract_hover(md: str) -> tuple[str, dict[str, dict]]:
    """Pull the <!--hover--> block out of Markdown.

    Each line after the marker is: url | title | status | owner | description
    Returns (md_without_hover, {url: {title, status, owner, desc}}).
    """
    idx = md.find(HOVER_MARKER)
    if idx == -1:
        return md, {}
    before = md[:idx]
    after = md[idx + len(HOVER_MARKER):]
    # hover block runs until the next sentinel (<!--provenance-->) or end of string
    end = after.find("<!--")
    if end != -1:
        hover_block = after[:end]
        rest = after[end:]
    else:
        hover_block = after
        rest = ""
    hovers: dict[str, dict] = {}
    for line in hover_block.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        url = parts[0]
        hovers[url] = {
            "title": parts[1] if len(parts) > 1 else "",
            "status": parts[2] if len(parts) > 2 else "",
            "owner": parts[3] if len(parts) > 3 else "",
            "desc": parts[4] if len(parts) > 4 else "",
        }
    return before + rest, hovers


def _inject_hover_attrs(html_body: str, hovers: dict[str, dict]) -> str:
    """Add data-hover-* attributes to <a> tags whose href matches a hover entry."""
    if not hovers:
        return html_body

    def _augment_link(m):
        full = m.group(0)
        href = m.group(1)
        info = hovers.get(href)
        if not info:
            return full
        attrs = []
        for key in ("title", "status", "owner", "desc"):
            val = info.get(key, "")
            if val:
                attrs.append(f'data-hover-{key}="{_html.escape(val, quote=True)}"')
        if not attrs:
            return full
        return full.replace(f'href="{href}"',
                            f'href="{href}" {" ".join(attrs)}')

    return re.sub(r'<a\s+href="([^"]+)"', _augment_link, html_body)


# ---- fenced-div extraction (::: sidenote, ::: summary) --------------------

_FENCED_DIV = re.compile(
    r'^::: *(sidenote|summary)\s*\n(.*?)\n::: *$',
    re.MULTILINE | re.DOTALL,
)


def _extract_fenced_divs(md: str) -> tuple[str, list[str], str]:
    """Pull ::: sidenote and ::: summary blocks out of Markdown.

    Returns (md_with_placeholders, html_stash, summary_html). Placeholders are
    restored after the Markdown→HTML pass so neither converter can mangle them.
    """
    stash: list[str] = []
    sidenote_counter = [0]
    summary_parts: list[str] = []

    def _stash(m):
        kind = m.group(1)
        inner = m.group(2).strip()
        inner_html = _inline(inner) if "\n" not in inner else _md_to_body(inner)

        if kind == "summary":
            summary_parts.append(inner_html)
            return ""

        sidenote_counter[0] += 1
        n = sidenote_counter[0]
        html = (f'<label for="sn-{n}" class="sidenote-toggle">&#9654; Note</label>'
                f'<input type="checkbox" id="sn-{n}" class="sidenote-checkbox">'
                f'<aside class="sidenote">'
                f'{inner_html}</aside>')
        stash.append(html)
        return f"\n\x00STASH-{len(stash) - 1}\x00\n"

    cleaned = _FENCED_DIV.sub(_stash, md)
    summary_html = "\n".join(summary_parts)
    return cleaned, stash, summary_html


def _restore_stashed(html: str, stash: list[str]) -> str:
    """Replace placeholder tokens with their stashed HTML."""
    def _put_back(m):
        return stash[int(m.group(1))]
    return re.sub(r'(?:<p>)?\x00STASH-(\d+)\x00(?:</p>)?', _put_back, html)


# ---- HTML document ----------------------------------------------------------

PAGE_CSS = """\
:root {
  --ink: #1a1a2e; --ink-secondary: #555770; --ink-faint: #8a8a9a;
  --accent: #8b2500; --accent-light: #c4530a;
  --surface: #faf9f6; --surface-raised: #f0eeea; --surface-code: #f5f3ef;
  --rule: #d4d0c8; --link: #8b2500; --link-hover: #c4530a; --ref-bg: #f7f5f1;
  --content-width: 650px; --sidenote-width: 230px; --sidenote-gap: 30px;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ink: #d8d5cf; --ink-secondary: #9e9bab; --ink-faint: #6e6b7b;
    --accent: #d4764e; --accent-light: #e8956e;
    --surface: #1a1a24; --surface-raised: #242430; --surface-code: #20202c;
    --rule: #33333f; --link: #d4764e; --link-hover: #e8956e; --ref-bg: #1e1e2a;
  }
}
:root[data-theme="dark"] {
  --ink: #d8d5cf; --ink-secondary: #9e9bab; --ink-faint: #6e6b7b;
  --accent: #d4764e; --accent-light: #e8956e;
  --surface: #1a1a24; --surface-raised: #242430; --surface-code: #20202c;
  --rule: #33333f; --link: #d4764e; --link-hover: #e8956e; --ref-bg: #1e1e2a;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: var(--surface); color: var(--ink);
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  font-size: 16px; line-height: 1.7;
  -webkit-font-smoothing: antialiased;
}
.page {
  max-width: calc(var(--content-width) + var(--sidenote-width) + var(--sidenote-gap) + 80px);
  margin: 0 auto; padding: 3rem 40px 4rem; position: relative;
}
@media (max-width: 1060px) {
  .page { max-width: 100%; padding: 2rem 1.5rem 3rem; }
}
.content { max-width: var(--content-width); }
.paper-header { max-width: var(--content-width); margin-bottom: 2.5rem; padding-bottom: 2rem; }
.paper-header h1 {
  font-family: 'Palatino Linotype', Palatino, 'Book Antiqua', Georgia, serif;
  font-size: 2rem; font-weight: 600; line-height: 1.25; color: var(--ink);
  text-wrap: balance; margin-bottom: 0.6rem; letter-spacing: -0.01em;
}
.paper-meta { font-size: 0.875rem; color: var(--ink-secondary); line-height: 1.5; }
.paper-meta .author { font-weight: 500; }
.paper-meta .mumwelt-link { color: var(--ink-secondary); text-decoration: none; border-bottom: 1px dotted var(--ink-faint); }
.paper-meta .mumwelt-link:hover { color: var(--accent); border-bottom-color: var(--accent); }
.prompt-box {
  max-width: var(--content-width); margin-bottom: 2.5rem; position: relative;
}
.prompt-label {
  position: absolute; left: -0.6em; top: 50%; transform: translateY(-50%);
  font-family: 'Palatino Linotype', Palatino, 'Book Antiqua', Georgia, serif;
  font-size: 5rem; font-weight: 700; color: var(--ink); opacity: 0.08;
  line-height: 1; pointer-events: none; user-select: none;
}
.prompt-text {
  font-family: 'Palatino Linotype', Palatino, 'Book Antiqua', Georgia, serif;
  font-style: italic; font-size: 1.15rem; line-height: 1.55; color: var(--ink);
  position: relative;
}
.abstract { margin-bottom: 2.5rem; max-width: var(--content-width); }
.abstract-label {
  font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--ink-secondary); margin-bottom: 0.5rem;
}
.abstract p { font-size: 0.92rem; line-height: 1.75; color: var(--ink); }
h1, h2, h3 {
  font-family: 'Palatino Linotype', Palatino, 'Book Antiqua', Georgia, serif;
  font-weight: 600; color: var(--ink); text-wrap: balance;
}
h2 { font-size: 1.4rem; margin-top: 2.5rem; margin-bottom: 0.75rem; letter-spacing: -0.005em; }
h3 { font-size: 1.1rem; margin-top: 1.75rem; margin-bottom: 0.5rem; }
p { margin-bottom: 1rem; max-width: var(--content-width); }
a { color: var(--link); text-decoration: none; border-bottom: 1px solid transparent;
    transition: border-color 0.15s, color 0.15s; }
a:hover { color: var(--link-hover); border-bottom-color: var(--link-hover); }
.date-label { cursor: default; border-bottom: 1px dotted var(--ink-faint); }
strong { font-weight: 600; }
.sidenote-checkbox { display: none; }
.sidenote-toggle { display: none; }
.sidenote {
  float: right; clear: right; width: var(--sidenote-width);
  margin-right: calc(-1 * (var(--sidenote-width) + var(--sidenote-gap)));
  margin-top: 0.2rem; margin-bottom: 1rem;
  font-size: 0.8rem; line-height: 1.5; color: var(--ink-secondary);
}
.sidenote-number { font-size: 0.7rem; font-weight: 600; color: var(--accent); margin-right: 0.3em; }
@media (max-width: 1060px) {
  .sidenote-toggle {
    display: inline; cursor: pointer; color: var(--accent);
    font-size: 0.78rem; font-weight: 600; user-select: none;
  }
  .sidenote {
    float: none; display: none; width: 100%; margin: 0.4rem 0 0.75rem 0;
    font-size: 0.84rem; padding: 0.6rem 0.9rem; background: var(--surface-raised);
    border-radius: 4px; border-left: 2px solid var(--accent);
  }
  .sidenote-checkbox:checked + .sidenote { display: block; }
  .sidenote-number { display: none; }
}
blockquote { border-left: 2px solid var(--rule); padding-left: 1.25rem;
  margin: 1.25rem 0; color: var(--ink-secondary); font-style: italic; }
code { font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 0.85em;
  background: var(--surface-code); padding: 0.15em 0.35em; border-radius: 3px; }
pre { background: var(--surface-code); border: 1px solid var(--rule); border-radius: 4px;
  padding: 1rem 1.25rem; overflow-x: auto; margin: 1.25rem 0; max-width: var(--content-width); }
pre code { background: none; padding: 0; font-size: 0.82rem; line-height: 1.6; }
ul, ol { margin-bottom: 1rem; padding-left: 1.5rem; max-width: var(--content-width); }
li { margin-bottom: 0.35rem; }
li::marker { color: var(--ink-faint); }
table { max-width: var(--content-width); border-collapse: collapse; width: 100%;
  margin: 1.25rem 0; font-variant-numeric: tabular-nums; font-size: 0.9rem; }
thead { border-top: 2px solid var(--ink); border-bottom: 1px solid var(--ink); }
th { font-weight: 600; padding: 0.3rem 0.75rem 0.35rem; text-align: left;
  line-height: 1.2; white-space: nowrap; font-size: 0.82rem; vertical-align: bottom; }
td { padding: 0.35rem 0.75rem; border: none; vertical-align: top; }
tbody { border-bottom: 1.5px solid var(--ink); }
th:first-child, td:first-child { padding-left: 0; }
th:last-child, td:last-child { padding-right: 0; }
figure { margin: 2rem 0; max-width: var(--content-width); }
figure img { width: 100%; border-radius: 3px; border: 1px solid var(--rule); }
figcaption { font-size: 0.8rem; color: var(--ink-secondary); margin-top: 0.5rem;
  line-height: 1.5; font-style: italic; }
footer.provenance {
  margin-top: 3rem; padding-top: 1rem; max-width: var(--content-width);
  font-size: 0.78rem; line-height: 1.6; color: var(--ink-faint);
}
footer.provenance p { margin-bottom: 0.3rem; }
footer.provenance a { color: var(--ink-faint); }
footer.provenance blockquote { margin: 0; padding: 0; border: none; color: inherit; }
a[data-hover-title] { position: relative; }
.hover-card {
  position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%);
  width: 320px; max-width: 90vw; padding: 0.65rem 0.8rem;
  background: var(--surface-raised); border: 1px solid var(--rule);
  border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  font-size: 0.78rem; line-height: 1.45; color: var(--ink);
  pointer-events: none; z-index: 100; margin-bottom: 6px;
  opacity: 0; transition: opacity 0.12s;
}
a[data-hover-title]:hover .hover-card,
a[data-hover-title]:focus .hover-card,
a.cite:hover .hover-card { opacity: 1; }
.hover-card .hc-title { font-weight: 600; margin-bottom: 0.2rem; }
.hover-card .hc-meta { font-size: 0.72rem; color: var(--ink-faint); margin-bottom: 0.25rem; }
.hover-card .hc-status {
  display: inline-block; font-size: 0.65rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.04em;
  padding: 0.1em 0.4em; border-radius: 3px; margin-right: 0.4em;
}
.hc-status-open { background: #fff3e0; color: #e65100; }
.hc-status-closed, .hc-status-merged { background: #e8f5e9; color: #2e7d32; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) .hover-card { box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
  :root:not([data-theme="light"]) .hc-status-open { background: #3a2a10; color: #ffb74d; }
  :root:not([data-theme="light"]) .hc-status-closed,
  :root:not([data-theme="light"]) .hc-status-merged { background: #1b3a1e; color: #66bb6a; }
}
:root[data-theme="dark"] .hover-card { box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
:root[data-theme="dark"] .hc-status-open { background: #3a2a10; color: #ffb74d; }
:root[data-theme="dark"] .hc-status-closed,
:root[data-theme="dark"] .hc-status-merged { background: #1b3a1e; color: #66bb6a; }
.hover-card .hc-desc { color: var(--ink-secondary); }
.cite {
  font-size: 0.72rem; vertical-align: super; line-height: 0;
  color: var(--accent); font-weight: 600; text-decoration: none;
  border-bottom: none !important; position: relative;
}
.cite:hover { color: var(--link-hover); }
.references { margin-top: 3rem; max-width: var(--content-width); }
.references h2 { font-size: 1.15rem; margin-bottom: 1rem; }
.ref-list { list-style: none; padding: 0; counter-reset: ref; }
.ref-list li {
  counter-increment: ref; display: flex; align-items: baseline;
  gap: 0.5em; font-size: 0.82rem; line-height: 1.55;
  margin-bottom: 0.4rem; color: var(--ink-secondary);
}
.ref-list li::before {
  content: "[" counter(ref) "]"; flex-shrink: 0;
  font-variant-numeric: tabular-nums; color: var(--ink-faint);
  font-size: 0.78rem; min-width: 2.2em;
}
.ref-list .ref-body { flex: 1; min-width: 0; }
.ref-list .ref-title { font-weight: 500; color: var(--ink); }
.ref-list .ref-url {
  font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 0.75rem;
  color: var(--ink-faint); word-break: break-all; margin-left: 0.4em;
}
.ref-list .ref-url a { color: var(--ink-faint); border-bottom: none; }
.ref-list .ref-url a:hover { color: var(--link-hover); }
.ref-back {
  color: var(--accent); text-decoration: none; border-bottom: none !important;
  margin-left: 0.3em; font-size: 0.78rem;
}
.ref-back:hover { color: var(--link-hover); }
.status {
  display: inline-block; font-size: 0.65rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.04em;
  padding: 0.15em 0.5em; border-radius: 3px; vertical-align: middle;
}
.status-done { background: #e8f5e9; color: #2e7d32; }
.status-open { background: #fff3e0; color: #e65100; }
.status-blocked { background: #fce4ec; color: #c62828; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) .status-done { background: #1b3a1e; color: #66bb6a; }
  :root:not([data-theme="light"]) .status-open { background: #3a2a10; color: #ffb74d; }
  :root:not([data-theme="light"]) .status-blocked { background: #3a1520; color: #ef9a9a; }
}
:root[data-theme="dark"] .status-done { background: #1b3a1e; color: #66bb6a; }
:root[data-theme="dark"] .status-open { background: #3a2a10; color: #ffb74d; }
:root[data-theme="dark"] .status-blocked { background: #3a1520; color: #ef9a9a; }
.cite-card {
  position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%);
  width: 360px; max-width: 90vw; padding: 0.65rem 0.8rem;
  background: var(--surface-raised); border: 1px solid var(--rule);
  border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  font-size: 0.78rem; line-height: 1.45; color: var(--ink);
  pointer-events: none; z-index: 100; margin-bottom: 6px;
  opacity: 0; transition: opacity 0.12s;
  font-weight: 400; vertical-align: baseline; text-align: left;
}
a.cite:hover .cite-card { opacity: 1; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) .cite-card { box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
}
:root[data-theme="dark"] .cite-card { box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
@media print {
  body { font-size: 11pt; }
  .page { max-width: 100%; padding: 0; }
  .sidenote { float: right; width: 180px; margin-right: -210px; }
  .sidenote-toggle { display: none; }
  a { color: inherit; border-bottom: none; }
  .hover-card { display: none; }
  .cite-card { display: none; }
}"""


HOVER_JS = """\
document.addEventListener('DOMContentLoaded', function() {
  var vt = document.getElementById('viewing-time');
  if (vt) {
    function pad(n) { return n < 10 ? '0' + n : n; }
    function updateTime() {
      var d = new Date();
      vt.textContent = d.getUTCFullYear() + '-' + pad(d.getUTCMonth()+1) + '-' + pad(d.getUTCDate()) + ' ' + pad(d.getUTCHours()) + ':' + pad(d.getUTCMinutes()) + ' UTC';
    }
    updateTime();
    setInterval(updateTime, 60000);
  }
  document.querySelectorAll('a[data-hover-title]').forEach(function(a) {
    var card = document.createElement('span');
    card.className = 'hover-card';
    var title = a.getAttribute('data-hover-title') || '';
    var status = a.getAttribute('data-hover-status') || '';
    var owner = a.getAttribute('data-hover-owner') || '';
    var desc = a.getAttribute('data-hover-desc') || '';
    var statusCls = 'hc-status hc-status-' + status.toLowerCase().replace(/[^a-z]/g, '');
    var html = '<div class="hc-title">' + title + '</div>';
    var meta = [];
    if (status) meta.push('<span class="' + statusCls + '">' + status + '</span>');
    if (owner) meta.push(owner);
    if (meta.length) html += '<div class="hc-meta">' + meta.join(' ') + '</div>';
    if (desc) html += '<div class="hc-desc">' + desc + '</div>';
    card.innerHTML = html;
    a.appendChild(card);
  });
  document.querySelectorAll('a.cite').forEach(function(a) {
    var href = a.getAttribute('href') || '';
    if (!href.startsWith('#ref-')) return;
    var li = document.getElementById(href.slice(1));
    if (!li) return;
    var body = li.querySelector('.ref-body');
    if (!body) return;
    var refUrl = body.querySelector('.ref-url a');
    var refHref = refUrl ? refUrl.getAttribute('href') : '';
    var hoverSource = refHref ? document.querySelector('a[data-hover-title][href="' + refHref + '"]') : null;
    if (hoverSource) {
      var card = document.createElement('span');
      card.className = 'hover-card';
      var t = hoverSource.getAttribute('data-hover-title') || '';
      var s = hoverSource.getAttribute('data-hover-status') || '';
      var o = hoverSource.getAttribute('data-hover-owner') || '';
      var d = hoverSource.getAttribute('data-hover-desc') || '';
      var sc = 'hc-status hc-status-' + s.toLowerCase().replace(/[^a-z]/g, '');
      var h = '<div class="hc-title">' + t + '</div>';
      var m = [];
      if (s) m.push('<span class="' + sc + '">' + s + '</span>');
      if (o) m.push(o);
      if (m.length) h += '<div class="hc-meta">' + m.join(' ') + '</div>';
      if (d) h += '<div class="hc-desc">' + d + '</div>';
      card.innerHTML = h;
      a.appendChild(card);
    } else {
      var title = body.querySelector('.ref-title');
      if (!title) return;
      var card = document.createElement('span');
      card.className = 'cite-card';
      card.textContent = title.textContent;
      a.appendChild(card);
    }
  });
});"""


_QUOTED_SPEECH = re.compile(r'"[^"]{4,}"|“[^”]{4,}”')


def _lint_uncited_quotes(md: str) -> None:
    for i, line in enumerate(md.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith(">") or stripped.startswith("<!--"):
            continue
        for m in _QUOTED_SPEECH.finditer(line):
            text = m.group(0)
            if " " not in text:
                continue
            after = line[m.end():]
            before = line[:m.start()]
            if re.search(r"\[[^\]]*$", before) and re.search(r"^[^\[]*\]\(https?://", after):
                continue
            if re.search(r"\]\(https?://[^)]+\)", after[:120]):
                continue
            print(f"publish: warning: line {i}: uncited quote {text[:50]} "
                  f"-- quoted speech should link to its source",
                  file=sys.stderr)


def render_html(md: str, title: str, author: str, date: str,
                query: str = "", corpus_time: str = "") -> str:
    from datetime import datetime, timezone
    gen_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    _lint_uncited_quotes(md)
    md = re.sub(r"^#\s+.*\n+", "", md, count=1)
    md, math_stash = _prerender_math(md)
    md, hovers = _extract_hover(md)
    body_md, prov_md = _split_provenance(md)
    body_md, stash, summary_html = _extract_fenced_divs(body_md)
    body = _restore_math(
        _inject_hover_attrs(_restore_stashed(_md_to_body(body_md), stash), hovers),
        math_stash)
    katex_style = _katex_css()
    ts_parts = []
    if corpus_time:
        ts_parts.append(f"Corpus: {_html.escape(corpus_time)}")
    ts_parts.append(f"Generated: {gen_time}")
    ts_parts.append('Viewing: <span id="viewing-time"></span>')
    timestamps = "<p><em>" + " &middot; ".join(ts_parts) + "</em></p>\n"
    footer = ""
    if prov_md.strip():
        footer = f'\n<footer class="provenance">\n{timestamps}{_md_to_body(prov_md)}\n</footer>'
    else:
        footer = f'\n<footer class="provenance">\n{timestamps}</footer>'
    meta_parts = []
    if author:
        meta_parts.append(
            f'Posed by <span class="author">{_html.escape(author)}</span>, '
            f'answered by <a class="mumwelt-link" href="https://github.com/marin-community/mumwelt">mumwelt</a>'
        )
    if date:
        meta_parts.append(f'<span class="date-label" title="Generated {gen_time}">Published {_html.escape(date)}</span>')
    meta_line = " &middot; ".join(meta_parts)
    summary_block = ""
    if summary_html:
        summary_block = (f'<div class="abstract">\n'
                         f'<div class="abstract-label">Summary</div>\n'
                         f'{summary_html}\n</div>\n')
    query_block = ""
    if query:
        query_block = (f'<div class="prompt-box">\n'
                       f'<div class="prompt-label">?</div>\n'
                       f'<div class="prompt-text">{_html.escape(query)}</div>\n'
                       f'</div>\n')
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)}</title>
<style>
{PAGE_CSS}
{katex_style}
</style>
<script>{HOVER_JS}</script>
</head>
<body>
<div class="page">
<header class="paper-header">
<h1>{_html.escape(title)}</h1>
<div class="paper-meta">{meta_line}</div>
</header>
{query_block}{summary_block}<div class="content">
{body}
</div>{footer}
</div>
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

    doc = render_html(raw, title, author, date,
                      query=getattr(a, "query", "") or "",
                      corpus_time=getattr(a, "corpus_time", "") or "")
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
