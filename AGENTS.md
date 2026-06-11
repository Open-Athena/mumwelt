# Marin context for agents (portable)

This file is the model-agnostic version of the mumwelt skills — paste it (or the output
of `mum skills print`) into any agent's system prompt or tool docs. Claude users can
instead run `mum skills install` to drop the native skills into `~/.claude/skills`.

You have a local, offline-queryable **mirror of all Marin activity** — GitHub
issues/PRs/comments, Discord, Weights & Biases run metadata + final numbers, and the
distilled **weekly summaries** — searchable by keyword and meaning, every hit citable by
URL. The capability is the `mum` CLI; shell out to it.

## Keep it fresh
- `mum status` — is the local corpus / summaries current?
- `mum refresh` — pull the latest corpus (from marinmirror) + weekly summaries (from
  mws.oa.dev). Run if stale.

## Look things up (single-shot)
- `mum search "<natural language or identifier>"` — fused keyword+semantic hits with URLs.
  Flags: `--source github,discord,wandb,narrative`, `--kind …`, `--since YYYY-MM-DD`, `-k N`, `--json`.
- `mum show <url|ref>` — expand a hit to context (Discord window / GitHub issue+comments).
- `mum run <project>/<run>` — a W&B run's config + final summary numbers.
- `mum summaries show latest` / `mum summaries links <period>` — the weekly overview and its link-leads.

## Research a broad question (decompose → fan out → synthesize)
1. `mum summaries show latest` to orient.
2. Split the question into 3–8 facet sub-queries (decisions · PRs/code · runs/results · people · timeline).
3. Fan out: spawn one subagent per sub-query (each runs `mum search … --json` + `mum show`),
   **or** if you have no subagents, `mum search-multi "q1" "q2" … --json` (parallel, merged).
4. Verify load-bearing claims against their cited source; drop anything ungrounded.
5. Synthesize a structured answer where **every claim cites a URL**; prefer primary
   sources over the weekly-summary narrative.

## Publish / share a writeup
- `mum publish [file] --title "…"` — render a Markdown report (file arg or **stdin**) to a
  LaTeX-styled HTML page (latex.css + MathJax, all `[text](url)` links preserved), drop it
  in a **secret** gist via `gh`, and print a `htmlpreview.github.io` link to share. Flags:
  `--public`, `--author`, `--filename`, `--no-date`, `--open`, `--json`. A "secret" gist is
  unlisted but anyone with the link can read it — don't publish anything sensitive.

## Auth
`mum` talks to marinmirror with a GitHub bearer token (any Open-Athena org member,
`read:org`). It resolves the token from `MARINMIRROR_TOKEN`, then `gh auth token`, then
`~/.config/marin/token`. The weekly summaries (mws.oa.dev) are public — no token.
