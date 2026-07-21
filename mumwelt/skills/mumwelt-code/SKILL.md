---
name: mumwelt-code
description: Answer a single "where is X implemented / how does Y work" question about the Marin codebase by searching the embeddings-backed code lane first (main + in-flight branches), and escalating to the broader mumwelt research skill only if a clear, citable answer doesn't emerge. Use for one-locus code lookups; for "what happened / why was X chosen" or anything spanning subsystems, use mumwelt directly.
---

# mumwelt-code — code-lane-first lookup, escalate if it stays thin

For a **single code question with one likely home** — "where is the transformer implemented",
"how does Iris federation actually work", "where is the LR clamped", "what does `ExecutorStep`
do", "where does `X` come from". Try to answer it **cheaply and directly** from the code lane;
only reach for the full `mumwelt` fan-out if the code doesn't settle it. The `mum` CLI does the
retrieval; *you* judge whether the answer is clear.

## When this skill, when `mumwelt`

| the question is…                                    | use          |
|-----------------------------------------------------|--------------|
| *where is X / how is Y implemented* — one locus     | **mumwelt-code** |
| *what happened / why was X chosen / who owns it*    | `mumwelt` (decisions live in threads) |
| spans several subsystems, or is a retro / overview  | `mumwelt`    |
| needs numbers from runs (`wandb`)                   | `mumwelt`    |

**"How does it work" → code; "what happened / why" → threads.** If the question is really about
a *decision or event*, the code lane will answer the wrong question well — stop and use `mumwelt`.

## 0. Fresh mirror

```
mum status
```
If the corpus is missing, `mum refresh` (required). If it's **>1 day old**, refresh before
relying on it; within a day, proceed as-is (or ask, if the question is time-sensitive). Honor
"don't repull". (Window configurable via `MARIN_MAX_AGE_DAYS`.)

## 1. Search the code lane — deliberately, with the whole budget

```
mum search "<subject or identifier>" --source code -k 30
```
With an explicit `--source code`, the code lane is **primary and gets the whole budget**, so
20–30 hits is the right ask. Two lanes come back, and they have **different truth status**:

- **`code · main`** — how the system works **now**. Cite this for how things work.
- **`code · in-flight branches`** (`kind=branch-symbol`) — what someone is *trying* on a
  branch; it may never land. Each hit carries the **branch name** and its **last committer**.
  Cite a branch **only** for a claim about in-flight work, and say so: "on branch `x`,
  \<person\> is doing Y". Never report a branch symbol as current behaviour.

Sharpen the search:
- **Know the symbol?** Search it literally — identifier search splits camelCase both ways, so
  `ExecutorStep`, `executor_step`, and "executor step" all reach each other.
- **Conceptual "how does X work"?** Add HyDE: write 1–3 short hypothetical code/docstring
  snippets and pass each as `--vec-text "<snippet>"` to drive the semantic leg while the
  literal query still drives keyword.
- **"Where does `X` come from"?** A `module` chunk lists a file's **resolved imports**, so it
  answers the provenance with a path.
- **Might be in-flight work?** The branch lane is already in the output; if the question is
  about something just-filed, lean on it and pull any `#number` out of the branch name
  (`codex/6597-moe-mgpu` → #6597) — but keep attributing it as in-flight.

Expand the few hits that look load-bearing with `mum show <url>` to read fuller context before
you commit to an answer.

## 2. Did a clear answer emerge?

Judge it yourself — a cosine score can't. **Clear** looks like:
- one (or a small, coherent set of) symbol/module whose **name + snippet directly answers**,
  and ideally corroboration — the definition *plus* a call site, or a literal identifier match;
- the locus is on **`main`** (or, for an explicitly in-flight question, a branch you can name).

**Not clear** — treat as a signal to escalate, not to pad:
- only scattered, weakly-related hits, or matches that merely *contain* the words;
- the real answer clearly spans multiple subsystems or files with no single home;
- the finding hinges on *why / what changed* — that lives in threads, not symbols;
- the code lane comes back effectively empty. (That is a **valid** outcome — plenty of
  questions have no code answer. Don't manufacture relevance to look thorough.)

## 3a. Clear → report + cite

Answer directly, pointing at the specific files/symbols with their URLs. Keep `main` and branch
findings distinct. Prefer the **defining** symbol over one that merely references it. Don't cite
code you're not actually leaning on.

## 3b. Not clear → report what you found, then recommend `mumwelt` (don't auto-run)

Say plainly what the lane did and didn't establish, cite whatever partial leads are real, and
**offer** the deeper pass rather than launching it — the fan-out is expensive and the user owns
that call:

> The code lane points at \<partial finding, with URLs\>, but it doesn't settle \<the gap\>
> — that part is about \<why/what-happened / spans several subsystems\>, which lives in the
> issue/Discord threads. Want me to run **`mumwelt`** on this to get the full, cited picture?

If they say yes, hand off to `mumwelt` with the same question (it'll decompose into facets, fan
out across `github,discord,wandb,narrative` plus its own code lane, verify, and synthesize).

## Provenance

Close with a one-line note of what you searched and the corpus freshness (from the `mum status`
in §0), e.g. *"Searched `--source code` (main + branches) over marinmirror — 50231 chunks, built
8h ago."* Terser than the full `mumwelt` footer; this is a lookup, not a report.
