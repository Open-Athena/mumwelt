# mumwelt

**An agent's *Umwelt* for Marin** — the slice of Marin's world an LLM agent can perceive
and act on. A small client toolkit (`mum`) plus portable skills that let any shell-capable
agent keep a fresh local mirror of Marin's activity, search it well, pull W&B run facts,
read the weekly summaries, and run multi-subagent research — all citable by URL.

It is the **client** to [`marinmirror`](https://marinmirror.exe.xyz) (the hosted,
Open-Athena-gated corpus of GitHub + Discord + W&B activity) and to the public weekly
summaries at [`mws.oa.dev`](https://mws.oa.dev).

## Install

```bash
pip install -e .          # brings in fastembed + numpy for hybrid (keyword+semantic) search
```

Token: `mum` authenticates to marinmirror with a GitHub token belonging to an **Open-Athena**
member (`read:org`), resolved from `MARINMIRROR_TOKEN` → `gh auth token` →
`~/.config/marin/token`. The weekly summaries are public (no token).

## The `mum` CLI

```
mum status                         freshness of the local corpus + summaries vs upstream
mum refresh [--force]              pull the latest corpus (marinmirror) + weekly summaries (mws.oa.dev)
mum search "<q>" [flags]           hybrid keyword+semantic search; cited hits
mum search-multi "q1" "q2" …       run several queries concurrently, merged + deduped
mum show <url|ref> [--window N]    expand a hit to context (discord window / github thread)
mum run <project>/<run>            a W&B run's metadata + final summary numbers
mum summaries [list|show|links|refresh] [<period|latest>]
mum skills [list|print|install [dest]]
```

`mum search` / `search-multi` flags: `-k N`, `--source`, `--kind`, `--since`, `--until`,
`--fts-only` (skip the model load), `--json`.

Local cache lives under `~/.cache/marin/` (`corpus-index.db` + `summaries/`).

## Skills

Two skills, shipped in Claude's `SKILL.md` format and as a portable [`AGENTS.md`](AGENTS.md):

- **`marin-context`** — refresh + search + show + W&B + weekly summaries. The everyday
  "look something up about Marin and cite it" skill.
- **`marin-research`** — decompose a broad question → fan out parallel searches across
  subagents (or `mum search-multi`) → verify → synthesize a cited answer.

```bash
mum skills install            # → ~/.claude/skills/  (Claude Code / claude.ai)
mum skills print              # dump the markdown for any other agent's prompt
```

## How it fits together

```
marinmirror.exe.xyz                         mws.oa.dev
  /manifest.json  /corpus-index.db            /  → summaries/summary-<period>.html
  /wandb/<p>/<r>/config                        (public)
        │ (bearer: Open-Athena token)               │
        ▼                                            ▼
   mum refresh ───────────► ~/.cache/marin/ ◄──── mum refresh
                              corpus-index.db
                              summaries/*.html
        │                          │
        ▼                          ▼
   mum search / show / run    mum summaries
   (FTS5 ∪ vector, RRF)       (overview + link-leads)
        │
        ▼
   agents (marin-context / marin-research skills)
```

The agent (any LLM) reasons and synthesizes; `mum` does retrieval. Search fuses FTS5
keyword search with cosine over the corpus's 384-d `BAAI/bge-small-en-v1.5` embeddings
via reciprocal-rank fusion — so identifiers and natural-language intent both hit.
