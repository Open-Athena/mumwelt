---
name: marin-research
description: Answer a broad, ambiguous, or multi-part question about Marin by decomposing it into parallel searches (across subagents) over the local marinmirror corpus and weekly summaries, then synthesizing a cited answer. Use for "give me the full picture of X", literature-review-style asks, retros, or anything one search query won't cover.
---

# marin-research — multi-subagent research over the Marin corpus

For questions too broad or multi-faceted for a single `mum search`. The pattern is
**decompose → fan out → verify → synthesize**. *You* (the agent) do the reasoning and
writing; the `mum` CLI does fast retrieval. This complements **marin-context** (use that
for single-shot lookups).

## 0. Fresh mirror

```
mum status
```
If the corpus is missing, `mum refresh` (required). If it's **>7 days old**, refresh before
relying on it. **Within 7 days, don't auto-pull** — proceed as-is, or *ask* the user first
when the question is time-sensitive. If the user says "don't repull", honor that and use
what's on disk. (The 7-day window is configurable via `MARIN_MAX_AGE_DAYS`.)

## 1. Orient, then decompose

Skim the relevant weekly summary for framing and link-leads:
```
mum summaries show latest          # or the period the question is about
```
Then break the question into **3–8 focused sub-queries** along facets:
- **decisions / discussion** (Discord, issue threads): *why* was X chosen?
- **code / PRs**: what landed, by whom?
- **runs / results** (W&B): what was tried, what were the numbers?
- **people**: who owns/drove X?
- **timeline**: what changed when (use `--since`)?

## 2. Fan out — two modes

**If your host has subagents (e.g. Claude's Task/Agent tool):** spawn **one subagent per
sub-query**, in parallel. Give each subagent: its sub-query, and instructions to run
`mum search "<sub-query>" --json` (then `mum show <url>` on its 2–3 best hits) and return
a short digest of findings **with URLs**. Collect the digests.

**If single-process (no subagent framework):** use the built-in parallel primitive —
```
mum search-multi "<sub-query 1>" "<sub-query 2>" "<sub-query 3>" … --json
```
It runs all queries concurrently and returns a merged, de-duplicated candidate set
(each hit annotated with which sub-queries surfaced it). Then `mum show` the most
promising URLs.

## 3. Verify (for load-bearing claims)

Before asserting something important, confirm it against its source: re-`mum show` the
cited chunk, or (with subagents) spawn a skeptic that tries to refute the claim from the
corpus. Drop anything you can't ground in a URL.

## 4. Synthesize

Merge the findings, **dedupe by URL**, and write a structured answer where **every claim
cites a `url`**. Call out disagreements and gaps explicitly. Prefer **primary sources**
(the actual PR / message / run) over the weekly-summary narrative, but use the summary's
links as a map of what else to chase.

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
