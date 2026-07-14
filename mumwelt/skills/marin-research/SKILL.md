---
name: marin-research
description: Answer a broad, ambiguous, or multi-part question about Marin by decomposing it into parallel searches (across subagents) over the local marinmirror corpus and weekly summaries, then synthesizing a cited answer. Use for "give me the full picture of X", literature-review-style asks, retros, or anything one search query won't cover.
---

# marin-research — multi-subagent research over the Marin corpus

For questions too broad or multi-faceted for a single `mum search`. The pattern is
**decompose → fan out → verify → synthesize**. *You* (the agent) do the reasoning and
writing; the `mum` CLI does fast retrieval.

## 0. Fresh mirror

```
mum status
```
If the corpus is missing, `mum refresh` (required). If it's **>7 days old**, refresh before
relying on it. **Within 7 days, don't auto-pull** — proceed as-is, or *ask* the user first
when the question is time-sensitive. If the user says "don't repull", honor that and use
what's on disk. (The 7-day window is configurable via `MARIN_MAX_AGE_DAYS`.)

## 1. Orient, then decompose

**Orient first — it's a vocabulary bootstrap, not a formality.** A generic question ("how is
GPU training going") only matches generically-titled docs; it will **never** surface the
specifically-titled engineering threads (`Drive 90B MoE MFU from 3% to ~25%`, `Speed up Grug
MoE Muon`) — you can't search for terms you don't know yet, and a broad scout search stays
stuck in that generic basin. The **weekly summaries** are the escape: they're *recency-
structured* (not query-matched), so they're saturated with the **current subsystem vocabulary**
and the live issue numbers.

Read the **last ~2–3 weekly summaries** — one is *not* enough for an ongoing topic (it spans
weeks, and a single week's snapshot reads as current and misleads):
```
mum summaries list                 # see the periods
mum summaries show <latest>        # and the 1–2 weeks before it for anything ongoing
```
From them, **harvest (a) the recurring subsystem TERMS** (specific optimizers, parallelism
strategies, perf metrics, cluster names) **and (b) the issue/PR #numbers named** — then
`mum show` the most relevant of those issues directly.

**Pick the entry point by topic shape.** Summaries are the bootstrap for a *diffuse* state
whose vocabulary you can't guess ("how is X going" — the subsystem names live only in the
summary). But for a **strongly-named or just-filed** topic — a milestone, run name, `#issue`,
"the July hero run" — a **keyword scout** (`mum search "<the name>"`) or a milestone/issue
browse is the better and sometimes *only* entry, because the thing may be **newer than the
latest summary**. Either way the goal is identical: **discover the real structure + vocabulary
before decomposing** — never decompose from generic words.

Then break the question into **3–8 focused sub-queries built from the harvested vocabulary,
not generic words** — a facet named "Muon optimizer cost / pipeline parallelism" surfaces the
right thread; "performance and blockers" does not. Useful facets:
- **decisions / discussion** (*why* was X chosen?), **code / PRs** (*what landed, by whom?*),
  **runs / results** (*the numbers?*), **people** (*who owns it?*), **timeline** (*what changed
  when?*).

## 2. Fan out — two modes

**If your host has subagents (e.g. Claude's Task/Agent tool):** spawn **one subagent per
sub-query**, in parallel. Give each subagent its sub-query and tell it to:
1. **Write 3 short hypothetical-answer paragraphs** (2–3 sentences each) — plausible answers
   to its sub-query, as if excerpted from a Marin issue / Discord thread / weekly summary.
   Invent concrete-sounding specifics; this is **HyDE**, so the text only needs to sit in the
   right *semantic neighborhood*, not be true.
2. Run `mum search "<sub-query>" --json --vec-text "<doc1>" --vec-text "<doc2>" --vec-text "<doc3>"`
   — the three docs are mean-pooled to drive the **semantic (vector)** leg, while the literal
   sub-query still drives the **keyword (FTS)** leg, so exact identifiers (`#1234`, run names)
   aren't diluted.
3. `mum show <url>` on its **top ~20 best hits** to read full context.
4. Return a short digest of findings **with URLs**.

Collect the digests and **dedupe by URL** across sub-queries (overlap collapses the real
expansion count well below 20 × #sub-queries). HyDE earns its keep on broad "how/what's
going" sub-queries; on a pure-identifier sub-query it adds little, but the FTS leg still
nails the identifier, so it's harmless — hence n=3 by default.

**If single-process (no subagent framework):** use the built-in parallel primitive —
```
mum search-multi "<sub-query 1>" "<sub-query 2>" "<sub-query 3>" … --json
```
It runs all queries concurrently and returns a merged, de-duplicated candidate set
(each hit annotated with which sub-queries surfaced it). Then `mum show` the most
promising URLs. Note: `search-multi` doesn't take `--vec-text`, so this path is
keyword+vector on the **literal** queries (no HyDE). To get HyDE without subagents, run
individual `mum search "<sub-query>" --vec-text "<doc1>" --vec-text "<doc2>" --vec-text "<doc3>"`
calls instead.

## 3. Verify (for load-bearing claims)

Before asserting something important, confirm it against its source: re-`mum show` the
cited chunk, or (with subagents) spawn a skeptic that tries to refute the claim from the
corpus. Drop anything you can't ground in a URL.

**Run these adversarial checks on every load-bearing number/claim — they are the failure
modes that bite hardest:**
- **Target vs achieved.** Is this number a *measured result* or a stated *goal / target /
  plan / "aim for X"*? **Never report a target as an achievement.** Prefer a thread's
  **latest / close-out** comment over mid-thread proposals — an experiment log's final
  verdict beats its opening hypothesis (e.g. the close-out "stuck at ~9% MFU" beats an
  earlier "drive to ~25%" goal).
- **Platform / config attribution.** Tag every metric with its hardware/config (GPU vs TPU,
  which cluster, which model size) and **never carry a number across contexts** (a TPU MFU
  reported as a GPU result is a classic inversion).
- **Quote-or-omit.** If you can't tie a specific number / run-name / issue# to a quoted line
  in a source, **drop it** rather than paraphrase it into existence.
- **Temporal / staleness.** Establish *when* each fact was true and prefer the **latest**
  state. A status true a week or two ago may be superseded (a run "still blocked on GPU" last
  week may have **moved to TPU** since) — chase the close-out / most-recent comment before
  reporting a state as current. Don't narrate an old snapshot as the present.

## 4. Synthesize

Merge the findings, **dedupe by URL**, and write a structured answer where **every claim
cites a `url`**. Call out disagreements and gaps explicitly. Prefer **primary sources**
(the actual PR / message / run) over the weekly-summary narrative, but use the summary's
links as a map of what else to chase.

**Fidelity guards — a confidently-wrong answer is worse than an incomplete one:** label
numbers as *achieved / target / projected*; when sources disagree (e.g. 9% vs 25% MFU),
surface the conflict and resolve to the **close-out** result rather than silently picking
one; keep hardware/config attached to every metric; and if the corpus only supports a
partial answer, **say so plainly** instead of padding with plausible-sounding specifics.

## 5. Provenance footer

End every answer with a lightweight **provenance footer**, set off from the body and in
muted/de-emphasized text, so the reader can see *what was searched* and *how fresh it
was*. Pull the freshness line straight from the `mum status` you already ran in §0 (run it
again if you didn't). Two parts:

1. **Data used & freshness** — corpus size + age, and the summaries period/latest week,
   verbatim from `mum status` (e.g. "50231 chunks, built 8h ago · summaries through
   2026-06-01_2026-06-07"). Note any refresh you triggered this run.
2. **Query trace** — the user's **original question**, then the **sub-queries** you
   fanned out on (the §1 decomposition).

Render it after a `---` rule using a small/dim style: a blockquote of italic text, which
the host shows muted in a terminal. Prefix it with the `<!--provenance-->` sentinel —
invisible when the Markdown is rendered, but it tells **marin-publish** where the trailer
starts so it can style the whole block as gray footnote text (and drop the duplicate rule).
Put a blank `>` line between the three parts so each lands on its own line. Shape:

```
---
<!--provenance-->
> *Data: marinmirror — 50231 chunks, built 8h ago · summaries through 2026-06-01_2026-06-07
> (refreshed this run).*
>
> *Query: "<the user's original question>"*
>
> *Sub-queries: "<sub-query 1>" · "<sub-query 2>" · "<sub-query 3>" · …*
```

Keep it terse — it's a trailer, not a section. If a sub-query returned nothing useful,
still list it (a dry facet is signal too).

---

**Notes**
- `mum search-multi` is the portable fallback — it parallelizes retrieval even with no
  subagent framework, so this skill works on any host.
- Keep sub-queries specific; vague sub-queries return noise. Identifiers (run names,
  `#1234`, logins) and concept phrases both work.
