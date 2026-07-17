# Brief: independently regenerate & evaluate the Marin eval golds

**You are a strong frontier model with a subagent/Task framework. Spend subagents freely — fan out
hard.** Your job has three parts, per question, over a 9-question eval set:

- **(A) Regenerate** your own best-possible *gold* (reference) answer from a frozen corpus.
- **(B) Evaluate** our existing *proposed* gold for the same question — correctness, completeness,
  internal consistency, overreach.
- **(C) Compare** your gold vs ours; say which is better claim-by-claim, and whether our
  deterministic score (below) is **too harsh / fair / too lenient** on the two candidate answers.

Everything is **offline against a frozen corpus** — no live web, no GitHub. A quality delta must be
attributable to reasoning/retrieval, not corpus drift.

---

## 1. Environment & access

- **Frozen corpus** — `2026-07-16` snapshot: 68,026 chunks (github 20,609 / wandb 19,446 /
  discord 27,971) + 15 weekly summaries through `2026-07-12`. Content-addressed in
  `evals/corpus/2026-07-16/` (`FREEZE.json` has the sha256s).
- **Retrieval CLI** — use the pinned wrapper `evals/runners/mum-frozen` for **every** corpus call
  (it points `mum` at the frozen DB and hard-blocks refresh). Run from the repo root. Key verbs:
  ```
  evals/runners/mum-frozen status
  evals/runners/mum-frozen search "<q>" --json --vec-text "<hyde1>" --vec-text "<hyde2>" --vec-text "<hyde3>"
  evals/runners/mum-frozen search-multi "<q1>" "<q2>" ... --json      # parallel, deduped
  evals/runners/mum-frozen show <issue-url|ref>                        # expand a hit to full thread
  evals/runners/mum-frozen summaries list | show <week>               # recency-structured narratives
  ```
  Never call plain `mum` (hits the live mirror). Never refresh.
- **The method to follow** — the `marin-research` skill at `mumwelt/skills/marin-research/SKILL.md`:
  **orient (harvest vocabulary + issue #s from the last 2–3 weekly summaries) → decompose into 3–8
  vocabulary-specific sub-queries → fan out one subagent per sub-query (HyDE on the vector leg) →
  verify every load-bearing claim against its source → synthesize a cited answer.** Read it; it
  encodes the failure modes that bite (target-vs-achieved, platform attribution, close-out-over-
  proposal, quote-or-omit). For a gold, go **deeper** than the skill's default: more sub-queries,
  a skeptic subagent per load-bearing number, and a final verification pass that re-`show`s every
  citation.

## 2. The 9 questions

| id | question |
|---|---|
| `gpu` | explain how training on GPUs is going |
| `h100-67b` | What happened when we tried to bring the 67B A2B up on H100s? What did we try, what worked, what didn't, and what were the final MFUs? |
| `muon` | what is our current muon approach, and how did we end up here? |
| `july` | explain our july 2026 plan |
| `april` | how did we do on our april 2026 milestone? |
| `ablations` | data ablations so far — which mixes / classifiers compared, on what data sizes and test sets? |
| `classifier` | the data-classifier model (see #5810): how trained/evaluated, can weights be shared? |
| `benchmarks` | target benchmarks vs development proxies? |
| `inference` | who's doing inference, where is it, current perf, open issues? |

Exact wording in `evals/questions/<id>.json`.

## 3. What we already produced (read these for part B/C — do NOT let them anchor part A)

- **Our proposed golds:** `evals/golds-proposed/<id>.md` (deep-researched, ~2.6k–3.8k words each).
- **Candidate A** (skill on frozen corpus): `evals/candidates/opus48-marin-research/<id>.md`.
- **Candidate B** (no skill, live GitHub): `evals/baselines/opus48-repo/<id>.md`.
- **Machine targets** folded into `evals/questions/<id>.json`: `citations.must_have / nice_to_have
  / forbidden`, atomic `facts.must_have` (id + regex + note), `trap_checks`, and `review_notes`
  (what we already know is shaky). `gold_prose` is the proposed gold inlined.

**Produce part (A) blind first** — write your gold before reading ours — so your regeneration is
independent. Then open ours for (B)/(C).

## 4. How we score (so you can judge "too harsh") — `evals/runners/score.py`, deterministic, no LLM

```
Q = 100·(0.35·cite_recall_MH + 0.35·fact_cov_MH + 0.30·trap_pass) − 10·(forbidden cited),  clamp 0–100
hard gate ⇒ Q=0: a cited issue# not in the freeze (frozen mode), or a wrong abstention
```
- `cite_recall_MH` = exact issue-number **set membership** vs the gold's `must_have`.
- `fact_cov_MH` = fraction of `must_have` fact **regexes** that match the answer text.
- `trap_pass` = deterministic require/forbid checks.

**Known blind spots — flag every instance you find:**
- a **fact regex too strict** → answer states the same fact in other words/numbers but scores as a miss;
- a **citation counted as missed** when the answer cited a different-but-equally-valid or *newer* issue;
- a **bad target** (our gold cited the wrong issue, or asserts a fact the corpus doesn't support) →
  scoring against it is unfair in *both* directions;
- **too lenient**: an answer clears the regexes/citations but is shallow, wrong, or misattributed.

## 5. Known hot spots (start your scrutiny here)

- **`gpu` is the contested one.** The `2026-07-12` weekly summary narrates the 11B-A1.5B H100 run as
  "healthy, ~23.8% MFU, ~done"; the **W&B run states in the same freeze show those runs crashed
  (07-13/07-14)**. Our gold sided with W&B and overrode the summary — verify which is right. Our
  `gpu` gold is also internally inconsistent: its own generator tagged 11 issues it *cites*
  (#6979/#6998/#7012/#7024/#7013/#7015/#7079/#7064/#7034/#7073/#7074) as "forbidden/superseded."
- **Demanding fact bars.** Some golds set 20–31 atomic must-have facts (e.g. `muon`=31); decide
  which are genuinely load-bearing vs noise that just punishes phrasing.
- **`benchmarks`/`inference`/`ablations`** have low candidate cite-recall — check whether the
  missed must-haves are truly required or just one valid sourcing among several.

## 6. Deliverables (per question)

1. **Your regenerated gold** (`<id>.gold.md`), fully corpus-cited, with the fidelity guards applied
   (achieved-vs-target labels, hardware/config on every metric, close-out-over-proposal). Note the
   subagent fan-out you used.
2. **Verdict on our gold:** `solid | flawed | seriously_flawed` + a list of concrete errors,
   internal contradictions, and unsupported claims (with the issue# you checked).
3. **Head-to-head:** where your gold and ours differ, who is right and why (cite the corpus).
4. **Harshness ruling** on candidates A and B: `too_harsh | fair | too_lenient` + the Q you'd
   consider fair + the specific unfair deductions (bad regex / equivalent citation / bad target).
5. **Scorer fixes:** concrete edits (loosen regex X, swap must-have #A→#B, drop fact Y).

Return a compact structured summary across all 9 (a table of true-quality scores for gold/A/B, the
harshness verdicts, and the top gold defects), plus the per-question gold files.

## 7. Guardrails

Corpus-grounded only; **quote-or-omit** (drop anything you can't tie to a frozen source); cite every
load-bearing claim by issue/PR #; prefer primary sources over the weekly-summary narrative but use
summaries as the map. Don't optimize toward our targets — optimize toward what the **corpus** supports;
if that means contradicting our gold, do it and show the evidence.
