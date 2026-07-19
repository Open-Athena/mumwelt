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
  Flags: `--source github,discord,wandb,narrative,code`, `--kind …`, `--since YYYY-MM-DD`, `-k N`, `--json`.
- **`--source code`** — the marin repo's Python symbols (one chunk per function/class/method,
  one per module), on `main` plus every branch touched in the last 30 days. Use it for *what
  the code does now*; use `github` for *what was decided and why*. A `module` chunk lists a
  file's resolved imports, so it answers "where does `ExecutorStep` come from" with a path.
  Branch work is `kind=branch-symbol` and carries the branch name and its last committer —
  never report it as what is on `main`. Search splits camelCase both ways, so `ExecutorStep`,
  `executor_step`, and "executor step" all find each other.
- `mum show <url|ref>` — expand a hit to context (Discord window / GitHub issue+comments).
- `mum run <project>/<run>` — a W&B run's config + final summary numbers.
- `mum summaries show latest` / `mum summaries links <period>` — the weekly overview and its link-leads.

## Research a broad question (decompose → fan out → synthesize)
1. `mum summaries show latest` to orient.
2. Split the question into **8–12 facet sub-queries, more for a complex question — don't stop
   at 5**. Facet breadth is the biggest measured lever: on a live A/B, the 10-facet arm found
   18 genuinely useful documents the 8-facet arms missed, simply because they had no facet for
   those subsystems. An unasked facet cannot be retrieved. Name facets after *subsystems*, not
   themes.
3. Fan out: spawn one subagent per sub-query (each runs `mum search … --json` + `mum show`),
   **or** if you have no subagents, `mum search-multi "q1" "q2" … --json` (parallel, merged).
   `-k` defaults to **50** (gold-citation recall runs 32% @10 → 45% @20 → 58% @50, and the
   vector leg scores every chunk regardless of `k`), so pass `-k` only to go narrower.
   **A facet is a subject, not a source: route it to `github,discord` even when its subject is
   code** — the *story* of code work lives in threads, and routing a facet to `code` instead
   cost one arm an entire subsystem's narrative. Use `wandb` for run numbers, `narrative` to
   orient. Then run a **separate 3–5 search `--source code` lane** alongside (never instead),
   including `--kind branch-symbol` to see in-flight work: branch names carry the committer and
   often an issue number (`codex/6597-moe-mgpu` → #6597). Cite what it finds when relevant;
   let it come back empty when not. Branch work is always "on branch `x`, <person> is doing Y",
   never what `main` does.
4. Verify load-bearing claims against their cited source; drop anything ungrounded. Read to the
   END of a thread before reporting a failure or null result — a sealed negative may already
   have been superseded by a pivot that works.
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
