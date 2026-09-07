---
name: mumwelt-publish
description: Publish a finished research answer (Markdown with inline URL citations) to a private GitHub gist, rendered as a LaTeX-styled HTML document, and return a one-click htmlpreview.github.io link for easy sharing. Use after mumwelt to hand someone a clean, readable, link-preserving writeup.
---

# mumwelt-publish — share a research writeup as a typeset, linkable page

You've synthesized an answer (e.g. via **mumwelt**) — structured Markdown where
**every claim cites a URL**. This skill turns that into something shareable: a clean,
LaTeX-looking HTML page (serif typeset body, rendered math), stored in a **secret gist**,
with a **htmlpreview.github.io** link anyone can open in a browser. The capability is
`mum publish`; shell out to it.

## What it does

`mum publish` takes Markdown (a file arg, or **stdin**) and:
1. Renders it to a standalone HTML doc — [latex.css](https://latex.css.netlify.app) for the
   LaTeX look, MathJax for `$…$` / `$$…$$`, all your `[text](url)` links preserved as real
   anchors. (Uses `pandoc` if present for best fidelity; otherwise a built-in converter.)
2. Creates a **secret** gist via the `gh` CLI (same GitHub auth this toolkit already uses).
3. Prints the gist URL **and** a `https://htmlpreview.github.io/?<raw-url>` link that
   renders the HTML live — that preview link is the thing you share.

## Use it

Pipe the Markdown straight in (preferred — no temp file):

```
echo "$REPORT_MD" | mum publish --title "June 2026 hero-run recipe"
```

or from a file:

```
mum publish report.md --title "Marin MoE data-mix retro"
```

Useful flags:
- `--title "…"` — document title (default: a leading `# H1` in the Markdown, else "Research report").
- `--query "…"` — the research question. Rendered as a styled prompt box at the top of the
  page (large ghosted "?" with the question in italic). **Always pass this** — it's the
  user's original question (lightly cleaned up for grammar/spelling). Without it the page
  opens cold with no context.
- `--corpus-time "…"` — corpus build timestamp (UTC) for the provenance footer's "Corpus:"
  line. Pull it from `mum status` output.
- `--author "…"` — header byline (default: your `gh` login).
- `--public` — make it a public gist instead of secret.
- `--no-date` — omit today's date from the header.
- `--filename name.html` — name the file in the gist (default: a slug of the title).
- `--open` — also open the preview link in your browser.
- `--json` — emit `{gist, raw, preview, public, filename}` for scripting.

## Conventions

- **Write the report first, publish second.** Keep the inline URL citations from
  mumwelt intact — they become clickable links and are the whole point.
- A leading `# Title` line is lifted into the page header (so it isn't shown twice). Use
  `##`/`###` for the section structure.
- Math: inline `$x$`, display `$$…$$` — rendered by MathJax in the preview.

## Provenance footer

If the Markdown contains a `<!--provenance-->` sentinel (mumwelt emits one — the
data-freshness + query-trace trailer), everything after it is split out and rendered as a
muted `<footer>`: small, gray, set off by a hairline rule, with its links and blockquote
inheriting the gray so it reads as a footnote rather than body copy. The redundant `---`
the writeup puts just above the sentinel is dropped so there's only one separator. Nothing
to do — just leave the trailer (and its marker) in the Markdown you pipe to `mum publish`.

## Sharing caveat (important)

A "secret" gist is **unlisted, not private**: it isn't indexed or searchable, but **anyone
with the link — including the htmlpreview link — can read it.** Don't publish anything
sensitive. Use `--public` only when you actually want it discoverable.

## Auth

Requires the `gh` CLI, authenticated (`gh auth status`; `gh auth login` if not). The gist
is created on that account. (htmlpreview.github.io is a public renderer — it fetches the
gist's raw HTML at view time; no token involved.)

## Update check

Before publishing, check whether the local mumwelt install is behind the remote — unless
the user has opted out. The opt-out flag lives in `~/.config/marin/config.json`:

```bash
python3 -c "import json, pathlib; c=pathlib.Path.home()/'.config'/'marin'/'config.json'; print(json.loads(c.read_text()).get('skip_update_check', False) if c.exists() else False)"
```

If that prints `True`, skip the check entirely. Otherwise, compare local vs remote:

```bash
git -C "$(python3 -c 'import mumwelt, pathlib; print(pathlib.Path(mumwelt.__file__).parent.parent)')" \
  log --oneline -1 --format="%H"
```

```bash
gh api repos/Open-Athena/mumwelt/commits/main --jq '.sha'
```

If they differ, tell the user: "Your mumwelt is behind `main` — the remote has newer
commits. Want me to update it (`git -C <path> pull`)?" Offer three choices:
1. **Update now** — pull and proceed.
2. **Skip for now** — publish with the current version.
3. **Never ask again** — write `{"skip_update_check": true}` to
   `~/.config/marin/config.json` (merge with existing keys if the file exists) and proceed.

Do not update without confirmation. If the check fails (no network, not a git install,
etc.), skip silently and proceed.
