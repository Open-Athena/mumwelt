---
name: marin-context
description: Search and cite Marin's activity — GitHub issues/PRs/comments, Discord discussions, W&B runs, and the weekly summaries — via the local marinmirror corpus. Use whenever a question is about Marin: what was decided and why, who did what, a PR/issue/run, a Discord thread, project history, or a training run's setup/results. Refreshes the local mirror first if stale.
---

# marin-context — search the Marin corpus

You have a **local, offline-queryable mirror of all Marin activity** (GitHub, Discord,
W&B run metadata + final numbers, and the distilled weekly summaries), searchable by
keyword **and** meaning, with every hit citable by canonical URL. Reach for this whenever
the user asks anything about Marin. The capability is the `mum` CLI — shell out to it.

## 1. Make sure the mirror is fresh

```
mum status
```
If the corpus is missing or stale (`built >24h ago`), refresh — this pulls the latest
corpus from marinmirror **and** the latest weekly summaries from mws.oa.dev:
```
mum refresh
```
(First corpus pull is ~150 MB; afterward it only downloads when the server has rebuilt.)

## 2. Get oriented with the weekly summaries (optional but valuable)

The weekly summaries are the **overview layer** — human-distilled narratives of each week
with dense outbound links. Great for "what happened recently" or to find entry points
before drilling in:
```
mum summaries list                 # all weeks
mum summaries show latest           # the latest summary as text
mum summaries links 2026-06-01      # outbound links (PRs/issues/runs/people) to chase
```

## 3. Search

```
mum search "why capacity factor 1.0 for MoE"
```
Returns ranked hits (source/kind, title, **URL**, snippet), fusing keyword (FTS5) and
semantic search. Useful flags: `--source github,discord,wandb,narrative`,
`--kind issue,pr,comment,message,run,section`, `--since YYYY-MM-DD`, `-k N`, `--json`.

Tips:
- Search by **concept** ("muon optimizer convergence") *and* by **identifier** ("#5596",
  a run name, a GitHub login) — the index fuses both, so either works.
- Start broad, then narrow with `--source` / `--kind` / `--since`.

## 4. Expand a hit for context

A single Discord message or comment is often a fragment. Reconstruct its context from the
corpus (no extra calls):
```
mum show <url-or-ref>      # discord → surrounding channel window; github → issue + comments
```

## 5. W&B runs — config + final numbers

```
mum run marin_moe/<run>            # or: mum run <wandb.ai run URL>
```
Returns the run's metadata + **final summary numbers** (loss, eval/bpb, MFU, tokens, …)
from marinmirror's SQLite store. (Per-step history time-series is not mirrored.)

## 6. Answer with citations

Every result carries a canonical `url`. **Cite it.** Prefer linking/quoting the source
over paraphrasing from memory; if results conflict, say so and cite both.

---

**Broad or multi-part question?** (e.g. "give me the full picture of X") switch to the
**marin-research** skill — it decomposes the question and searches in parallel.
