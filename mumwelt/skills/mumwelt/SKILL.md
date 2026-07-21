---
name: mumwelt
description: Answer a broad, ambiguous, or multi-part question about Marin by decomposing it into parallel searches (across subagents) over the local marinmirror corpus and weekly summaries, then synthesizing a cited answer. Use for "give me the full picture of X", literature-review-style asks, retros, or anything one search query won't cover. For a single "where is X / how is Y implemented" code lookup, reach for the lighter mumwelt-code first.
---

# mumwelt — multi-subagent research over the Marin corpus

For questions too broad or multi-faceted for a single `mum search`. The pattern is
**decompose → fan out → verify → synthesize**. *You* (the agent) do the reasoning and
writing; the `mum` CLI does fast retrieval.

## 0. Fresh mirror

```
mum status
```
If the corpus is missing, `mum refresh` (required). If it's **>1 day old**, refresh before
relying on it. **Within 1 day, don't auto-pull** — proceed as-is, or *ask* the user first
when the question is time-sensitive. If the user says "don't repull", honor that and use
what's on disk. (The 1-day window is configurable via `MARIN_MAX_AGE_DAYS`.)

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

Then break the question into **focused sub-queries built from the harvested vocabulary,
not generic words** — a facet named "Muon optimizer cost / pipeline parallelism" surfaces the
right thread; "performance and blockers" does not.

**Use 8–12 facets, and MORE for a genuinely complex question — do not stop at 5.** This is
the single biggest lever measured on a live A/B, and it is bigger than any routing choice.
Three arms answered "what happened in June and July so far?"; the one that used **10** facets
found material the two that used **8** never saw — 18 documents an independent review graded
genuinely useful, including a whole subsystem's story. The 8-facet arms did not lose them by
routing badly; they had **no facet at all** for MarinFold, long-context work, the month's
milestones, or storage. An unasked facet cannot be retrieved by any amount of clever
filtering, and each facet is cheap: one parallel subagent.

A two-month or multi-subsystem question needs a facet per *subsystem*, not per *theme*. If
your facet list has fewer entries than the project has active workstreams, it is too short.

Useful facets:
- **decisions / discussion** (*why* was X chosen?), **code / PRs** (*what landed, by whom?*),
  **runs / results** (*the numbers?*), **people** (*who owns it?*), **timeline** (*what changed
  when?*).

## 1.5 Route facets to threads; give code its own lane

**A facet is a SUBJECT, not a source.** Name the facet after the thing you want to know
about — "Iris cross-cluster federation", "vLLM GrugMoE serving" — and send it where the
*discussion* lives: `--source github,discord`. Do this **even when the subject is code.**

| facet shape                            | `--source`       | why                                     |
|----------------------------------------|------------------|-----------------------------------------|
| *why* / *what happened* — any subject   | `github,discord` | decisions and events live in threads     |
| *what did the run produce* (numbers)    | `wandb`          | config + final numbers                   |
| *what happened this week* / orient      | `narrative`      | recency-structured, vocabulary-rich      |
| *how is X implemented* (only)           | `code` (`-k 30`) | see the carve-out below                  |

**Never spend a facet slot on `--source code` for a "what happened" question.** Measured
on a live A/B of "what happened in June and July": routing two of eight facets to `code`
cost **18 genuinely useful documents** — the entire Iris/CoreWeave infrastructure
narrative (the federation refactor landing, a datakit OOM root-caused to a gcsfs
regression, an NFS-lock fix) — because the *story* of that work is in issue threads while
`--source code` returns source files. The code citations it bought were fine; they just
answered a different question than the one asked.

### The standing code lane

**Always search code — but deliberately, in its own lane, never incidentally.** In the same
test, the unrouted arm retrieved **53 code documents and cited zero**; the ones it uniquely
saw graded 6 noise / 1 marginal / 0 valuable (auth tests, a controller `main.py`). Code that
turns up as a by-product of a mixed search is buried at ranks nobody reads. The routed arm
retrieved 46 and cited 5, four of them judged solid.

**This is now enforced by the tool, not left to your discipline.** `mum search` no longer
mixes code into the main results at all — code and prose are ranked separately and never
compete for the same slots. Two consequences:

- **Every `mum search` already gives you the code lanes, free.** Output ends with
  `--- code · main (25) ---` and `--- code · in-flight branches (10) ---`. You do not
  have to spend a facet slot to *see* whether code has anything; you only spend one when
  it does. Read those sections and judge them yourself — no relevance threshold gates
  them, which is deliberate, because you can tell "this symbol answers the question" from
  "this merely contains the word `gpu`" and a cosine score cannot.

- **main and branches are separate lanes because they have different truth status.**
  `main` is how the system works. A branch symbol is what someone is *trying* — it may
  never land. Answering "how does X work" from branch code is not a worse answer, it is a
  **wrong** one. Cite `main` for how things work; cite a branch **only** for claims about
  what is in flight, and say so explicitly. They are budgeted separately because branch
  symbols exist only when they *differ* from the merge base, so the corpus selects for
  actively-edited code and they crowd: measured, 39% of code chunks but 48–84% of an
  undifferentiated top-25, and 21 of 25 on "how do we compute MFU".
- **When the question really is about code, go deep:** `mum search "<subject>" --source
  code -k 30`. With an explicit `--source code` the code lane is primary and gets the
  whole budget, so 20–30 hits is the right ask, not 10.

Why they were separated: fusing the two rankings is zero-sum for a fixed result budget, and
measured across all 9 harness questions, code in the mixed ranking cost **1.9pp of gold
recall at both @10 and @20** while adding nothing at @50. It also could never *gain*
anything on that metric — which counts only issue/PR citations — so mixing cost prose and
could not be credited for code. Split, code costs prose exactly zero.

So run **one** extra subagent (two only if the question genuinely spans systems) alongside
the facet agents — in parallel, fanning in to synthesis like any other digest. Do NOT make
the other agents wait on it; that would serialize the fan-out. Give it the question's main
subjects and have it run keyword+vector+HyDE against `--source code`, then return a SHORT
digest of the few symbols that actually establish something.

**Budget the lane at 3–5 searches, not more.** An arm that spent 10 of its 18 searches on
the code lane starved its facet coverage and still cited nothing from code — the lane was
doing lead generation at the price of an entire subsystem's thread coverage.

**Cite what the lane finds when it is relevant — and only then.** The lane is not merely a
lead generator: a symbol, a module docstring or a branch is a legitimate primary source for
a claim about how something is built or who is building it. If you assert that work is
happening on a branch, **cite the branch chunk itself**, not a general issue that merely
mentions the topic (a review caught exactly that substitution). Equally, do not cite code to
look thorough — if the lane's findings do not support a claim you are actually making,
report them as nothing and move on.

**It must be allowed to come back empty.** Plenty of questions have no code dimension —
on the eval harness's `april` and `classifier` questions, code takes 0% of top-10 slots.
"Nothing relevant in code" is a valid, useful answer; a lane that manufactures relevance
just gets its findings cited to look thorough.

### Read the active branches

Branches are the best available signal for *what is being worked on right now*, and they
are usually ahead of both the threads and the weekly summaries. The code lane should
explicitly mine them:

```
mum search "<subject>" --source code --kind branch-symbol
```

Each hit carries the **branch name** and its **last committer**, and the set of files a
branch touches sketches what it is trying to do. Three things to exploit:

- **Branch names encode intent and often an issue number** — `codex/6597-moe-mgpu`,
  `agent/20260624-fix-6614`, `rav/iris-gpu-image`. Pull the `#number` out and `mum show`
  that issue: it links in-flight code to the thread that motivated it.
- **Who is committing** tells you who owns the work right now, which the thread may not.
- **Which files changed** tells you the shape of the change before any of it is written up.

**Always attribute it as in-flight.** `kind=branch-symbol` is work on someone's branch, not
what `main` does — say "on branch `x`, <person> is doing Y". Reporting a branch symbol as
current behaviour is the §3 temporal trap in a new costume.

### The carve-out

`--source code` may own a facet when the question is genuinely about *implementation*
("how does Iris federation actually work", "where is the LR clamped"). There, code is the
primary source and threads are the supplement — the reverse of the default above. Decide by
question type: **"what happened" → threads; "how does it work" → code.**

Two more code notes: a `module` chunk lists its resolved imports, so "where does
`ExecutorStep` come from" is answerable with a path. And identifier search splits camelCase
both ways, so `ExecutorStep`, `executor_step`, and "executor step" all reach each other.

Leave `--source` off when a facet genuinely spans sources, or when you are still orienting.

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
   `-k` now defaults to **50** (gold-citation recall runs 32% @10 → 45% @20 → 58% @50 on the
   eval harness, and raising it is close to free because the vector leg scores every chunk
   regardless), so you only pass `-k` to go *narrower*. Add `--source` when the facet has an
   obvious home (see §1.5).
3. `mum show <url>` on its **top ~20 best hits** to read full context. Keep this bound at ~20
   even though you searched 50 — `show` expands a whole thread and is what actually consumes
   context. The other 30 are not wasted: a search hit already carries `url`, `title`, `date`
   and a snippet, so it is **citable as a lead** without being expanded. Expand for depth,
   scan the tail for coverage.
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
(each hit annotated with which sub-queries surfaced it). `-k` is candidates **per query**
(default 50) and `--total` caps the merged set (default 60). Then `mum show` the most
promising URLs. Note: `search-multi` doesn't take `--vec-text`, so this path is
keyword+vector on the **literal** queries (no HyDE). To get HyDE without subagents, run
individual `mum search "<sub-query>" --vec-text "<doc1>" --vec-text "<doc2>" --vec-text "<doc3>"`
calls instead.

## 3. Verify

Before asserting something important, confirm it against its source: re-`mum show` the
cited chunk, or (with subagents) spawn a skeptic that tries to refute the claim from the
corpus. Drop anything you can't ground in a URL.

**Run these adversarial checks on every number/claim — they are the failure
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
- **A sealed failure may already have been superseded.** Reporting negative results honestly
  is right, but a *stale* negative is worse than none: it tells the reader something was tried
  and failed when the team has since made it work. An arm reported an Over-Encoding ladder as
  "came back null" from a gate sealed as a failure one day earlier — while the same thread
  showed a reformulated version **winning** at every matched checkpoint. Before reporting any
  failure, dead end, or null result, **read to the END of the thread** and check for a pivot,
  a reformulation, or a retry. Report the failure *and* what happened next.

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
invisible when the Markdown is rendered, but it tells **mumwelt-publish** where the trailer
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
