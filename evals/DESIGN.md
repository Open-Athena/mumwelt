# Marin skill evals — design

Status: **draft / proposal** · Owner: ihodes · Last updated: 2026-07-14

A regression + comparison harness for the Marin retrieval skills (`marin-context`,
`marin-research`, and any faster successor). It lets us iterate on **skill + prompt** and
prove, on every change, that answer **quality, citation grounding, and recency ("retrieved
date")** did not regress — and it makes the central bet measurable:

> **Can we build a fast skill (seconds) that matches the quality of the slow
> `marin-research` skill (minutes) on the same frozen corpus?**

Everything below is built around answering that with numbers, reproducibly, offline.

---

## 1. Principles

1. **Frozen corpus, no network.** Every run reads a pinned, content-addressed snapshot of
   the corpus. The harness **blocks network egress** and treats any online access as a hard
   test failure. This removes corpus drift and live-serving flakiness as confounders, so a
   quality delta is attributable to the skill/prompt change under test.
2. **Human-curated golds are ground truth — not `marin-research`.** Golds are independent,
   hand-vetted, and **improvable over time**; they may be *better* than any one-shot run.
   `marin-research` is scored against the golds like any other candidate and **tracked as a
   reference line**, never used as the bar itself (a moving reference drifts under you).
3. **Golds must stay corpus-grounded.** Every must-have citation must exist in the frozen
   corpus, and every must-have fact must be supported by a corpus doc. A gold is "the best
   answer *achievable from this frozen corpus* by a careful human," not omniscience —
   otherwise the eval becomes unwinnable and stops measuring the skill.
4. **Deterministic-first scoring.** As much of the score as possible is computed without an
   LLM (citation-set membership, fact substring/regex/number extraction, trap labels). The
   LLM-judge is confined to prose quality + claim-support and is **gated behind** the
   deterministic checks. (History: our prior harness's own meta-finding was that *the judge*,
   not retrieval, was the unreliable component and reward-hacked coverage.)
5. **Separate retrieval from synthesis.** Score "was the needed doc retrievable?"
   independently of "did the agent use it?" This is the single highest-value diagnostic the
   frozen corpus unlocks.
6. **Cost is first-class, but don't gate on wall-time.** Wall-time is the headline we care
   about (minutes → seconds) but the noisiest signal. Gate on reproducible cost (tokens,
   call counts); report wall-time for humans.

---

## 2. What an eval is

One eval = one question with a curated, corpus-grounded target. Record schema
(`evals/questions/<id>.json`):

```json
{
  "id": "gpu",
  "question": "explain how training on GPUs is going",
  "corpus": "2026-07-14",            // freeze this gold is valid against (see §3)
  "answerable": true,                // false => the correct answer is "not found"
  "skill_hint": "research",          // expected route: context | fast | research

  "citations": {
    "must_have":  ["#6304", "#6493", "#6044"],
    "nice_to_have": ["#6321", "#6367", "#6532"],
    "forbidden":  ["#5167"],         // wrong/superseded/off-platform: citing = penalty
    "recency_critical": ["#6493"]    // subset of must_have that carry the latest state
  },

  "facts": {
    "must_have": [
      {"id": "F2", "match": "9.37%|9.4%", "note": "MFU pure-FSDP ceiling, short of ~25% target"},
      {"id": "F7", "match": "TPU|v4",     "note": "67B flagship moved OFF GPU onto TPU"}
    ],
    "nice_to_have": [
      {"id": "F6", "match": "Triton|ragged_dot"}
    ]
  },

  "traps": {
    "target_vs_achieved": "~25% MFU is a TARGET; 9.37% is the achieved close-out",
    "platform": "H100/GPU — do not report a TPU MFU as the GPU result",
    "temporal": "report the close-out state, not a mid-thread proposal",
    "superseded": ["#5167 optimizer study is not the GPU-MFU answer"]
  },

  "gold_prose": "…full human-curated reference answer with inline issue-URL citations…"
}
```

Notes:
- `match` fields are regexes checked against the candidate answer text; keep each fact
  **atomic** so it scores independently.
- `forbidden` and `traps.superseded` overlap intentionally: a forbidden citation is the
  machine-checkable half; the trap is the human-readable rationale.
- **Unanswerable evals** (`answerable: false`) are a first-class question class: the correct
  behavior is explicit "not found," and answering as if answerable is a **hard fail**. RAG
  systems fail this silently; without negative evals we're blind to it.

Question classes to cover (seed each failure mode a faster skill introduces):
- **standard** — the four seed questions (gpu/july/april/muon).
- **unanswerable** — genuinely not in the frozen corpus.
- **conflict** — corpus disagrees (e.g. 9% vs 25% MFU); correct answer surfaces + resolves
  to close-out.
- **recency-trap** — an early thread state is superseded by a later one; correct answer
  reports the later. (Directly exercises "retrieved date.")
- **platform-trap** — a number exists on both GPU and TPU; correct answer attributes it.

---

## 3. The frozen corpus artifact

A freeze is a content-addressed bundle stored (or referenced by manifest) under
`evals/corpus/<date>/`:

```
evals/corpus/2026-06-28/
  corpus-index.db          # the pinned SQLite corpus
  corpus-manifest.json     # source snapshot metadata
  FREEZE.json              # { date, sha256(corpus-index.db), sha256(manifest), chunk_count }
```

- The harness sets `MARIN_CORPUS_DB=evals/corpus/<date>/corpus-index.db` and refuses to run
  if the on-disk sha256 doesn't match `FREEZE.json` (no accidental live-corpus runs).
- Golds reference a freeze by `corpus` date; a gold is only valid against its stamped freeze.
- **First freeze — decided: mint a fresh `2026-07-14`.** We do **not** have a Jun-28
  `corpus-index.db` saved (the closest local one is Jun-16, which predates the July-hero-run
  issues the golds cite), so rather than chase an archived DB we freeze today's corpus
  (already in `~/.cache/marin/corpus-index.db`) as `evals/corpus/2026-07-14/` and curate
  golds against it. Golds are improvable and corpus-grounded, so a fresh freeze is fine — it
  just needs its own gold pass, and it enables both `--mode full` and `--mode replay`.
  - **Kept as a secondary seed:** the Jun-28 eval cached exact `mum context` output per
    question in `~/workspace/marinmirror/scratch/loop2/cache/*.bundle` (+ `*.valid`). These
    remain usable for `--mode replay` smoke tests, but the four seed golds get re-vetted
    against the `2026-07-14` freeze (a few issue numbers / close-out states will have moved).

---

## 4. Scoring

### 4.1 Facets (per question)

| facet | how measured | in gate? |
|---|---|---|
| **citation recall (must-have)** | set membership of cited `#N` vs `must_have` | ✅ |
| **retrieval sufficiency** | were must-have `#N` present in the *retrieved bundle* (not just the answer)? separates retrieval miss from synthesis miss | ✅ |
| **recency retention** | recall over `recency_critical` subset — the "retrieved date" metric | ✅ |
| **hallucinated citations** | cited `#N` **not in the frozen corpus** | 🔒 hard gate |
| **forbidden citations** | cited `#N` ∈ `forbidden` | ✅ (penalty) |
| **fact coverage (must-have)** | regex match of each must-have fact | ✅ |
| **trap pass** | per-trap deterministic check (label present / platform correct / close-out value not the target value) | ✅ |
| **abstention** | unanswerable Q → "not found"; answerable Q → not fabricated | 🔒 hard gate |
| **claim↔citation support** | LLM/NLI: does the cited source support the adjacent claim? | ⚠️ judge-tier, reported only |
| **prose quality** | LLM judge rubric (structure, calibration/hedging, readability, length discipline) | ⚠️ judge-tier, reported only |
| **cost** | wall-time (report), output tokens + `mum` calls + LLM calls (gate) | ✅ (cost side) |

### 4.2 The one quality number **Q** (0–100, deterministic)

```
hard gates (any true ⇒ Q = 0):
  - any hallucinated citation
  - answerable question answered with fabrication
  - unanswerable question NOT abstained

otherwise:
  Q = 100 · ( 0.35·cite_recall_MH + 0.35·fact_cov_MH + 0.30·trap_pass )
        − 10 · (#forbidden_cited)
  clamp to [0, 100]
```

- `cite_recall_MH`, `fact_cov_MH`, `trap_pass` are fractions in [0,1].
- **Nice-to-haves are NOT in Q.** They form a secondary score `Q_nice` used only as a
  tiebreak / richness signal, so a candidate can't trade a must-have for two nice-to-haves.
- `retrieval_sufficiency_MH` and `recency_retention` are reported alongside Q as named
  sub-metrics (they diagnose *why* Q moved — retrieval vs synthesis, currency specifically).
- `Q_prose` (0–5) and `claim_support` (%) are reported but **never** in the ship gate.

### 4.3 Cost bundle (per question, per run)

```
cost = { wall_s, out_tokens, mum_calls, llm_calls }
```
Gate on `out_tokens`, `mum_calls`, `llm_calls` (reproducible). `wall_s` reported as median
over seeds for the human-facing "minutes → seconds" story.

---

## 5. Harness

```
evals/
  DESIGN.md                 # this doc
  questions/<id>.json       # the eval set (§2)
  golds/<id>.md             # optional long-form gold prose (also inlined in json)
  corpus/<date>/…           # frozen corpus artifact (§3)
  runners/
    run.py                  # run a variant over the eval set → results.jsonl
    score.py                # deterministic scorer + gates → scores.json
    judge.py                # gated LLM-judge tier (prose, claim-support)
    report.py               # leaderboard + quality-vs-cost table (md/html)
  results/<skill>@<hash>/…  # recorded runs (see §6)
```

**Execution modes** (this is the retrieval/synthesis split, operationalized):

- **`--mode full`** — candidate does its own retrieval against the frozen DB, then
  synthesizes. Use for **retrieval-strategy experiments** (fewer bundles, leaner
  `-k/--expand/--follow-links/--tail`, dropping subagents). Retrieval *is* the variable;
  `retrieval_sufficiency` tells you exactly what a leaner strategy stopped surfacing.
- **`--mode replay`** — retrieval is fixed (replay a cached/frozen bundle); only the
  **synthesis prompt** varies. Fully deterministic, fast, ideal for iterating wording/guards
  without paying for retrieval or fighting its noise. Seeds directly from the Jun-28
  `loop2/cache/*.bundle`.

**Offline enforcement:** run under a network-blocking sandbox; the harness asserts
`mum` used only the pinned DB and that no candidate tool hit the network. Any egress ⇒ fail.

**A candidate ("variant")** is identified by `(skill, skill_version/prompt_hash, model,
decoding, corpus_hash)`. The runner records all of these so results are attributable.

---

## 6. Baselines, gates, and the parity question

### 6.1 Recorded history
Every run is stored under `results/<skill>@<prompt_hash>/<corpus>/scores.json`, keyed by
`(skill, prompt_hash, model, corpus_hash)`. This gives:
- a **leaderboard** across all variants against a fixed freeze, and
- a **reference line for one-shot `marin-research`** tracked over time — so we can see it
  regress or improve against the (evolving) golds without ever treating it as the bar.

### 6.2 Ship gate (regression wall)
A skill/prompt change `C` is safe to ship vs its current baseline `B` iff, across **all**
questions (medians over seeds, §7):
- no must-have citation or fact that `B` satisfied is dropped by `C` (set inclusion),
- zero hallucinated citations, zero forbidden-cited, no abstention failures,
- `trap_pass(C) ≥ trap_pass(B)` everywhere,
- gated `cost(C) ≤ cost(B)` (for a "make it faster" change) **or** `Q(C) > Q(B)` if cost held.

### 6.3 Parity definition (the central bet)
A fast variant `F` **matches** `marin-research` `R` iff, per question:
- `median Q_F ≥ median Q_R − ε` (ε = measured noise band, §7), **and**
- `cost_F ≪ cost_R` on the gated cost bundle (the whole point).

The deliverable is a **quality-vs-cost scatter**: `R` sits top-right (high Q, minutes); we
hunt for an `F` on/above the same Q line but far to the left on cost.

---

## 7. Variance & honesty

- Run each eval **n ≥ 3 seeds**; report per-fact drop rate and a per-question **noise band ε**
  (e.g. half the inter-seed Q range, or a small fixed floor, whichever is larger).
- "Regression" / "parity" claims only count when the delta exceeds ε — otherwise we'd
  declare on noise (we saw exactly this single-fact n=1 flicker in the prior harness).
- Wall-time is reported as a median and explicitly **excluded** from gates.

---

## 8. Seed set & migration

Port the four existing questions from `~/workspace/marinmirror/scratch/loop2/`:

| id | question | source of gold |
|---|---|---|
| `gpu`   | "explain how training on GPUs is going" | hand-built facts F1–F7 (in `config.json`) |
| `july`  | "explain our july 2026 plan" | hand-built facts G1–G7 |
| `april` | "how did we do on our april 2026 milestone?" | facts X1–X7 + prose `golds/april.md` |
| `muon`  | "what is our current muon approach…" | facts X1–X7 + prose `golds/muon.md` |

Migration steps:
1. Convert each `config.json` question into `questions/<id>.json` (§2), splitting facts into
   must/nice and extracting `forbidden` + `traps` from the existing guard notes.
2. Seed `--mode replay` from `loop2/cache/*.bundle` + `*.valid` for immediate reproducibility.
3. Add 1 eval per new class: one **unanswerable**, one **recency-trap**, one **platform-trap**.
4. Curate golds against the chosen freeze; record a first `marin-research` reference run.

---

## 9. Open questions / future

- ~~**Freeze source:**~~ **decided** — mint a fresh `2026-07-14` freeze and re-gold (§3).
- **Auto-optimization:** any GEPA/DSPy-style prompt optimizer **must** target the
  deterministic Q + hard gates, never the LLM-judge, or it will reward-hack coverage.
- **Claim-support tier:** start as human spot-check; graduate to NLI only if it proves
  stable enough to trust.
- **Must/nice weighting:** revisit the 0.35/0.35/0.30 split once we have real spread.

## 10. Non-goals
- Not an eval of Marin the project (that's the `marin` repo's own eval suite).
- Not a live-freshness monitor (frozen corpus by construction).
- Not a leaderboard of LLMs — the model is pinned; we're testing **skills + prompts**.
