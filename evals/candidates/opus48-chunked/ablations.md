# Marin data ablations to date: mixes, classifiers, scales, and evals

Short version: there are **five distinct families** of data ablation running in Marin, and they use
different arms, different scales, and different rulers. In order of how directly they answer the
question:

1. **Mixture-selection swarms** — hundreds of proxy runs over bucket weights, used to *produce* a mix.
2. **Mix head-to-head validation** — `curated` vs `proportional` vs `nemotron` on an iso-FLOP ladder.
3. **Mixture-surrogate / optimizer ablations** — which regression form best predicts a mix's loss.
4. **Quality-classifier ablations** — which classifier best reproduces an LLM quality oracle.
5. **Mid-training / cooldown mix ablations** — replay-vs-math ratios on the Delphi ladder.

Plus an adjacent sixth family (tokenizer bake-offs) that is scored on the same rulers and is worth
knowing about because **three of its four results were reversed by measurement bugs**.

A caution that applies to everything below: the two headline "mix" results are *not* clean wins.
Curated-vs-proportional is a **statistical tie**, and the mixture-optimization work is validated
against a perplexity proxy that a team member has explicitly shown **anti-correlates with downstream
benchmarks in at least one sweep**. Details in §7.

---

## 1. At a glance

| Experiment | Arms compared | Training scale | Test data | Status |
|---|---|---|---|---|
| Precursor swarm [#2345](https://github.com/marin-community/marin/issues/2345) | 238 proxy mixes over 39 buckets | 60M params, 1.2B tokens | uncheatable BPB, Paloma, MMLU | Done; MMLU unfittable |
| Production swarm [#5364](https://github.com/marin-community/marin/issues/5364)/[#5365](https://github.com/marin-community/marin/issues/5365) | ~1,000 mixes over 200 datakit buckets | d=1280 (2.3B total / 408M active), 26.2B tok, 6e19 FLOPs/run | uncheatable BPB (SNR-selected) | Done 2026-06-15 |
| Mix head-to-head [#6757](https://github.com/marin-community/marin/issues/6757) | `curated` / `proportional` / `nemotron` | MoE d512–d1280, 3.82e17–3.46e19 FLOPs | 111-task suite + macro uncheatable BPB | Done; curated = weak default |
| Surrogate scaling [#6602](https://github.com/marin-community/marin/issues/6602)/[#6607](https://github.com/marin-community/marin/issues/6607)/[#6608](https://github.com/marin-community/marin/issues/6608)/[#6611](https://github.com/marin-community/marin/issues/6611) | DSP vs OLMix vs proportional vs unimax8 | Delphi ladder 3e18/2e19/3e20/1e21 | uncheatable BPB, OLMoBaseEval Table-9 | Partially complete |
| MDE feature ablation [#6326](https://github.com/marin-community/marin/issues/6326) | MDE checkpoint-likelihood features vs DSP | 300M panel, 39 single-domain experts | uncheatable BPB | Done — **negative** |
| Quality classifier [#6739](https://github.com/marin-community/marin/issues/6739) | fastText / transformer / 3 transfer teachers | 5,613 labelled docs | 961-doc oracle holdout | Done — label ceiling |
| Mid-training mixes [#4547](https://github.com/marin-community/marin/issues/4547) | p33m67 / p50m50 / p67m33 | 1.9B/3.4B/9.7B, 4.94–32.07B tok | math val loss, Paloma retention | 33/36 cells |
| Midtrain→SFT→RL [#6279](https://github.com/marin-community/marin/issues/6279) | 27 midtrained ckpts × 2 SFT recipes | 447M–9.7B, 9 FLOP scales | MATH-500, AIME24, GSM8K | Done (54/54) |

---

## 2. Mixture-selection swarms — how the production mix was actually made

The June/July hero mix was **not** picked from a handful of named candidate mixes. It came out of a
swarm that trains hundreds of small proxy models on different bucket weightings and fits a regression
from weights to loss.

**Precursor swarm ([#2345](https://github.com/marin-community/marin/issues/2345)).** "A swarm
consisting of **238 proxy runs** has been performed"; "Model size is **60M, trained for 1.2B tokens
(Chinchilla)**"; "Phases are **80% / 20%** … WSD, second phase is the Decay". The bucket space was 39
domains: "26 Dolma 3 Common Crawl domains … 4 other Dolma 3 domains, 9 Dolmino domains"
([comment](https://github.com/marin-community/marin/issues/2345#issuecomment-4102921841)). Two results
from this swarm shaped everything after it:

- **MMLU is not an optimizable target.** "we basically cannot reasonably fit on or optimize that
  (R² < 0.1), even though we can fit reasonably on C4EN BPB (R² > 0.7)" (same comment).
- **Scale does not rescue it.** "scaling up to 300M models trained on 3B tokens **did not reduce the
  variance**" ([comment](https://github.com/marin-community/marin/issues/2345#issuecomment-4115490875)).
  Corroborated in Discord: "the variance among 10 runs with the same mix and only different trainer
  seeds is almost as large as the variance of all swarm runs w/ different mixes"
  ([#data-mixing](https://discord.com/channels/1354881461060243556/1462895580064911522/1485029900715819190)).

**Production swarm ([#5364](https://github.com/marin-community/marin/issues/5364) design,
[#5365](https://github.com/marin-community/marin/issues/5365) launch).** "we need a swarm of **~1,000
runs**"; "we don't have any promising results for including scaling laws in our data mixing form, so
we're just going to do a **single scale**"; "each run in the swarm should be a **6e19 flop** run …
the ~correct config to launch there is **d=1280 (2.3B total, 408 active) for 26.2B tokens**"; "we'll
use the last 20% of the run as phase 2"
([comment](https://github.com/marin-community/marin/issues/5364#issuecomment-4453670611)). Launched
2026-05-25; "Production model swarm is finished" 2026-06-15
([#5365](https://github.com/marin-community/marin/issues/5365)). Extra intervention/domain-ablation
tranches ran under [#6606](https://github.com/marin-community/marin/issues/6606).

**What shipped.** The acceptance bar was set in
[#5359](https://github.com/marin-community/marin/issues/5359): the mix "must be better than (or worst
case equivalent to) proportional data mixing over all sources during pretraining **and** proportional
data mixing over hand-determined 'high-quality' sources as the strong baseline." The mix landed via
[#6391](https://github.com/marin-community/marin/issues/6391) →
[PR #6440](https://github.com/marin-community/marin/pull/6440), and is described verbatim in
[#6449](https://github.com/marin-community/marin/issues/6449):

> Source: `datakit/store_8ac06c74` — **167 mixable bucket caches plus a 33-cache `tail` group, ~10.37 T
> tokens**. Buckets named `c{cluster}q{quality}` (**40 clusters × 5 quality bins**). Mixture:
> **`mixture-3.csv` two-phase schedule. Phase 0 (first 80%) and phase 1 (last 20%)** each independently
> normalised. … `target_budget = 10_372_343_704_053`.

So "phase-1 / phase-2" in the hero-run chatter is this 80/20 schedule, not two separate corpora; phase 2
is the cooldown mix. The 67B-A2B run consuming it is
[#6044](https://github.com/marin-community/marin/issues/6044) (67.1B total / 2.01B active, ~10.07T
tokens, v4-2048). Larry's launch note: "we kicked off the 67B-A2B MoE trained on 10T tokens, using the
new Marin datamix from [willheld] and [Calvin]"
([#moe](https://discord.com/channels/1354881461060243556/1365044508546568372/1521008406184460288)).

**A hand-steered alternative was also pre-registered.**
[#6063 "\[Vibe Mixing\]: Pre-registered Will Vibe Mix"](https://github.com/marin-community/marin/issues/6063)
fit a DSP regression on the 241-run swarm over 39 Dolma3/Dolmino domains at 300M across 26 tasks, and
pre-registered confident wins (humaneval Δ_pct +90; github_python +88; gsm8k +85) against confident
losses (uncheatable/bbc_news −6, ao3_english −2), at L1 deviation 0.528 from proportional.

---

## 3. The one clean mix-vs-mix head-to-head — #6757

This is the most direct answer to "what mixes were compared, at what size, on what test data."

[#6757](https://github.com/marin-community/marin/issues/6757) (parent
[#6713](https://github.com/marin-community/marin/issues/6713)) compared **three mixes from the same
datakit store `store_8ac06c74`**: `curated` (the new swarm-optimized mix), `proportional`
(natural-token weights over the same pool), and `nemotron` (the old baseline, with `ep10t` simulated
epoching).

- **Training scale:** MoE at four iso-FLOP points — **d512 (3.82e17), d768 (2.81e18), d1024 (1.16e19),
  d1280 (3.46e19)**, on v5p-16 in us-east5.
- **Test data:** macro uncheatable-BPB bundles plus the full lm-eval mega set, reduced to "**111
  runnable tasks/cell across all 9 cells, fully symmetric**" (the unrunnable ones are deprecated
  dataset-scripts plus gated GPQA). Metric is per-axis **iso-FLOP effective speedup** with a 90% CI
  from the log-log fit residual
  ([comment](https://github.com/marin-community/marin/issues/6757#issuecomment-4847173767)).

**Curated vs nemotron (the old mix) — a real win, with a real regression:**

> ✅ Aggregate loss: MACRO uncheatable bpb **1.78× (90% CI [1.45, 2.24])**
> ✅ Code & math: HumanEval 10-shot [37, 168], GSM8K 5-shot [2.3, 16], MBPP [2.8, 31], all 8 `math_*`
> ⚠️ **Commonsense — curated significantly loses:** hellaswag [0.48, 0.80], winogrande [0.55, 0.94],
> piqa [0.59, 0.75], openbookqa, race, swag
> ⬜ Noise: MMLU (at chance), belebele, include, NLI, boolq, arc-challenge

**Curated vs proportional — a tie, and this is the load-bearing finding**
([close-out](https://github.com/marin-community/marin/issues/6757#issuecomment-4847875950)):

> Aggregate loss: curated **1.14× [0.93, 1.41]** vs proportional — **not significant**. Of **102**
> downstream tasks, only **3** separate the two … with 102 tasks at a 90% CI you'd expect ~10 false
> positives by chance — we found 3, i.e. **fewer than chance**. … take **curated** as a **weak
> default** — but the honest result is that at this scale the mix is not where AA-II gains will come
> from. A confident curated-vs-proportional call would need a larger-scale point (≥ d1280).

In other words: **all the measured benefit of the new pipeline so far is "new data pool beats old data
pool," not "our mixing algorithm beats proportional weighting."** The public speedup table across all
111 evals is at `https://oa.williamheld.com/datakit_sidebyside.html`
([announcement](https://discord.com/channels/1354881461060243556/1462895580064911522/1521595789783076985)).

Consistent with this, Will noted on the hero run itself: "in all my experiments the new mix is slightly
worse to Paloma… So any goodness in Paloma is Larry's arch magic"
([#moe](https://discord.com/channels/1354881461060243556/1365044508546568372/1521208053997895731)).
Paloma was never the selection metric.

---

## 4. Surrogate / optimizer ablations — which regression predicts a mix's loss

A separate axis: holding the swarm fixed, which *functional form* best predicts loss from mixture
weights. Fits run on a **300M panel of 280 rows** ("241 ex-ante qsplit/signal rows + 39 300M
proportional-controllability domain-deletion rows"), with the adaptive row
`baseline_olmix_loglinear_uncheatable_bpb` excluded "because it is an adaptive row produced after
seeing the original swarm"
([#6608](https://github.com/marin-community/marin/issues/6608#issuecomment-4797047321)).

Forms compared: **DSP** (canonical and effective-exposure), **OLMix** (log-linear, Huber-δ, KL-capped),
and objective-agnostic baselines **proportional** and **unimax8**
([#6607](https://github.com/marin-community/marin/issues/6607)).

On **OLMoBaseEval Table-9 macro BPB** (unweighted mean of 51 components), the cleanest head-to-head
([#6611](https://github.com/marin-community/marin/issues/6611#issuecomment-4811907754)):

| model | OOF Spearman | OOF RMSE | regret@1 | lower-tail optimism |
|---|---|---|---|---|
| DSP effective-exposure | **0.918** | 0.01284 | 0.00739 | 0.00696 |
| OLMix `single_tied` (paper-faithful, δ=0.02) | 0.380 | 0.03364 | 0.04660 | ~0.0204 |

Scaling validation runs on the **Delphi four-rung ladder: 3e18 / 2e19 / 3e20 / 1e21 FLOPs** on
v5p-8/16/32/64, 80/20 two-phase ([#6607](https://github.com/marin-community/marin/issues/6607)). Model
sizes per rung: 3.58e8 params / 1.58e9 tokens at 3e18, rising to 3.38e9 / 4.63e10 at 1e21. The one
decisive scaling verdict recorded in text is a *kill*, not a win:

> Stopped the remaining live DSP canonical scaling cells after the corrected progress plot showed it
> **scaling worse than DSP effective-exposure** on the intended target (`eval/uncheatable_eval/bpb`)
> — [#6602](https://github.com/marin-community/marin/issues/6602#issuecomment-4809543112)

**No ladder-level optimized-vs-baseline BPB table exists in this corpus.** As of the last comments
(2026-06-26) the Table-9-optimized OLMix and DSP-effexp arms had 3e18 and 2e19 succeeded with 3e20/1e21
still running ([#6611](https://github.com/marin-community/marin/issues/6611#issuecomment-4811907754)).

**One completed and clearly negative surrogate ablation:**
[#6326 "MDE features for 300M data mixing"](https://github.com/marin-community/marin/issues/6326)
trained "39 cap-1 single-domain 300M vertex experts plus two controls" and tested Mixtures-of-Data-
Experts checkpoint-likelihood features against DSP: "`vertex_mde_ridge` OOF Spearman is **0.8605**, but
**DSP is stronger at 0.9140**"; the 900-column MDE feature set has "effective rank about 22.3". Verdict:
"does not add enough independent signal to justify using it in the main optimizer."

---

## 5. Quality-classifier ablations

**The deployed v0 classifier ([#5810](https://github.com/marin-community/marin/issues/5810)).** fastText
trained on 5,613 usable docs (of 7,000 stratified) scored by an LLM on a 1–5 pretraining-utility rubric,
binary threshold at rubric ≥3. Held-out n=961: **AUC 0.846, Spearman ρ 0.641**. Head-to-head against
the off-the-shelf dolma3 classifier on the same set: "ours vs LLM ρ **0.641**, dolma3 vs LLM ρ
**0.168**, ours vs dolma3 ρ 0.210 — adding new signal rather than rediscovering dolma3"
([results](https://github.com/marin-community/marin/issues/5810#issuecomment-4480271122)).

**The label-ceiling study ([#6739](https://github.com/marin-community/marin/issues/6739), code in
[PR #6741](https://github.com/marin-community/marin/pull/6741)).** All arms trained on the same 5,613
Claude-scored docs across **104 datakit sources** and evaluated on the same **961-doc holdout**:

| classifier | AUC | ρ | FLOPs/token |
|---|---|---|---|
| fastText v0 (baseline) | 0.846 | 0.641 | ~0 |
| off-the-shelf dolma3-quality | — | 0.168 | — |
| **source-of-origin only (reads no text)** | **0.852** | — | 0 |
| pooled fast-transformer (meanmaxmin·w64·d512·L4) | **0.875** | **0.703** | 0.41M |
| L=0 "neural BoW" | 0.868 | 0.699 | 12K |

Three transfer/distillation teachers were tried and **all three were negative**: Nemotron-CC bucket
pretraining (finetuned ρ 0.60–0.62 vs 0.695 from scratch), source-prior weak supervision on ~200k docs
(0.775 pretrain-only vs 0.869 scratch), and FineWeb-Edu distillation (FineWeb-Edu's own score predicts
the oracle at only **ρ 0.28**). Close-out, marked high-confidence: "The **~0.87 plateau is the ceiling
of our specific 5.6k oracle labels** and cannot be broken for free… Going higher requires either more
real oracle labels."

**Two caveats that matter more than the numbers:**

1. **Nothing here is validated downstream.** From the same thread: "we validate against the oracle, not
   downstream — and the oracle is edu-leaning," whereas production filters (FineWeb-Edu, Nemotron-CC,
   DCLM) are validated by benchmark ablation, "never AUC against the annotator"
   ([comment](https://github.com/marin-community/marin/issues/6739#issuecomment-4833511462)).
2. **The classifier is sorting by domain, not quality.**
   [#6849](https://github.com/marin-community/marin/issues/6849): it "scores documents by domain /
   modality / language rather than intrinsic quality" — clean `starcoder2/ir_python` scores ~0.00,
   multilingual mean 0.18 vs oracle 0.51, top bucket q4 is code+math only. And the root cause is
   structural: "source alone predicts the oracle at AUC 0.852 (#6739) — so any faithful distillation
   reproduces the domain bias; **raising oracle-AUC does not fix it**." Companion bugs:
   [#6860](https://github.com/marin-community/marin/issues/6860) (12 of 98 sources have score stddev
   < 0.1) and [#6859](https://github.com/marin-community/marin/issues/6859) (scoring truncates input at
   `MAX_TEXT_CHARS = 4000`, so a 37.8 MB doc is judged on ~0.01% of its content). rav's framing:
   "both domain and quality classifiers are **v0** — the goal was to have e2e pipeline"
   ([#data-mixing](https://discord.com/channels/1354881461060243556/1462895580064911522/1520177514792620155)).

**Two independent ceilings found elsewhere.** A Luxical embedding quality probe plateaus at Spearman
~0.75 with 2,500/5,000/10,000 training docs giving 0.749/0.740/0.750 — "This ceiling is a property of
the Luxical embedding space, not a data limitation"
([#3535](https://github.com/marin-community/marin/issues/3535#issuecomment-4049284288)). And the
literature review in [#3049](https://github.com/marin-community/marin/issues/3049) flags the relevant
prior: "Li 2024 (DCLM) … a linear classifier on BGE embeddings performed poorly (**27.2 CORE vs 30.2 for
fastText**). Also, high agreement with LLM quality labels **anti-correlated** with effective filtering."

**Extraction, not classification —
[#2351](https://github.com/marin-community/marin/issues/2351).** Often lumped in here, this compares
*extraction pipelines* at a fixed 3000-WARC budget: Raw HTML 3.63T tokens / Resiliparse 142.7B / DCLM
2.66B / Nemotron-CC 2.70B / FineWeb-Edu 817M / LLM extraction 56.01B. Claim: "by throwing away less web
content our LLM extracted data can outperform existing curation pipelines when the data pool is held
fixed," with domain specs giving "up to 18 percentage points" over resiliparse. Its compute-matched
DCLM-CORE ladder exists only as W&B runs (`marin-dclm-core`, arms `llm_curated_dedup`,
`resiliparse_dedup`, `dclm`, at 1e17–2e21 FLOPs); **no written verdict for that ladder is in this
corpus.**

---

## 6. Mid-training / cooldown mix ablations

**The mixes ([#4547](https://github.com/marin-community/marin/issues/4547)).** Three pretrain-replay ÷
math ratios against `nemotron_cc_math_v1/4plus` ("the Phi-4-cleaned, top-quality band of
Nemotron-CC-Math"): **`p33m67`** (0.33 replay / 0.67 math), **`p50m50`**, **`p67m33`** (the
"replay-heavy, Mantis-style" arm).

- **Scale:** a 36-cell sweep, 3 mixes × 3 base scales × 4 LRs, each midtraining for **K = 0.20 of its
  base's pretrain token budget** → **4.94B (1e20) / 9.25B (1e21) / 32.07B (1e22)** tokens on bases of
  1.9B / 3.4B / 9.7B params. Completion is **33 of 36** cells with a usable final checkpoint.
- **Test data:** held-out `eval/nemotron_cc_math_v1/4plus/loss` (12,500 sequences ≈ 51.2M tokens), plus
  `c4_en` loss and Paloma macro/micro retention.
- **Findings:** "LR factor barely matters at the high end — within each base/mix/budget cell the spread
  across {0.5, 0.67, 0.83} is ≤ **0.04** in train loss; specialization is dominated by data, not LR."
  Recipe *ranks* generalize across scales. **`p33m67` wins at every scale on the math objective — but
  the Paloma-retention winner is `p67m33`**, so there is no single dominant mix. And the retention tax
  grows: "at fixed `p33m67-lr=0.5`, damage goes **+0.028 → +0.033 → +0.045**" across 1e20 → 1e22.
- **Two self-corrections published in-thread**, both of which weaken the result: (a) "the `1e20` rows
  were **not** trained from a Delphi base," so "claims that rely on 1e20 → 1e22 transfer are weakened";
  (b) **math val-set contamination** — "17% of val docs / 18.6% of val tokens had a J>=0.75 dup in
  train; 54.7% of windows touched," rising to 39.4% at 1e22. Follow-up:
  [#6742](https://github.com/marin-community/marin/issues/6742).

**The downstream consequence ([#6279](https://github.com/marin-community/marin/issues/6279)).** All 27
midtrained checkpoints (9 FLOP scales 3e18→1e22, 447M→9.7B params × 3 mixes) were SFT'd from **both**
cold-start recipes (`magpie_lr1e5`, `wc386k_lr1e5`) → **54/54 cells**, evaluated 0-shot at temp 0.7 on
**MATH-500 pass@1, AIME24 pass@1 (mean±se over 10 seeds), and GSM8K exact-match**.

- "MATH-500 rises monotonically with FLOPs at every mix/recipe — best-cell ≈ **2 → 45** across
  3e18→1e22." Mix ranking is stable: `p33m67` ≥ `p50m50` ≥ `p67m33`.
- The **18/18 base-SFT control arm** is what licenses the headline: math-strong `magpie` SFT on a bare
  base "never clears ~3.6 on MATH500 **across 447M→3.4B**" and **AIME24 = 0.0 across all 18 cells** —
  though at the largest scale it does lift off, base+magpie reaching **10.6** at 9.7B, still ~4× below
  the midtrained 45.0. The math-weak `wc386k` SFT does ~nothing on a bare base (1.6 @ 1e22) but takes a
  midtrained model from **6.0 → 45.0**. Hence: **midtraining supplies the capability, SFT elicits it.**
- pass@k corroborates: at 1e22, MATH500 base pass@128 **10.4 → midtrain 79.6** (≈8×).
- **RL on top is the smallest lever, and its close-out was revised.** The first leg (`rlvr7500_w1`) gave
  MATH500 45.0 → 53.4 but **AIME24 regressed 4.9 → 1.5**. That is *not* the final state: a later
  `dapo17k_w1` leg reached "**AIME24 6.6±0.4 = +1.7 vs SFT 4.9** — the highest absolute AIME24 in the
  entire suite," and an external xorl reproduction reported "MATH500 +12.3, gsm8k-flex +8.8, and
  **AIME24 did not regress**." Report the regression as one leg, not the verdict.
- Caveat the authors state themselves on general-capability retention (MMLU 58.6, HellaSwag 74.8 at
  9.7B): "this is 'looks intact,' **not** a measured no-degradation — the clean test needs the same
  checkpoint without the math-heavy mix as the control, **which we don't have yet**."

**The prediction program is still proposals.**
[#4507](https://github.com/marin-community/marin/issues/4507) /
[#4514](https://github.com/marin-community/marin/issues/4514) /
[#4513](https://github.com/marin-community/marin/issues/4513) pre-register a recipe family
(`R1_mantis`, `R2_more_math`, `R3_more_code`, `R4_more_hq` — e.g. "move 10 points of mass from the
Nemotron PT block into the existing math block", or PT/HQ 70/30 → 60/40) to be fit at 1e21+1e22 and
validated on a held-out 1e23 run, scored by Kendall τ of recipe ranking. **No results comment exists**;
a recorded blocker is "TPU quota for 1e23 midtraining. The 1e23 model is 25B params."
[#4551](https://github.com/marin-community/marin/issues/4551), the "train Isoflops/Delphi ladders on
various mid/post-training mixes" issue, is a one-line stub that **went stale on 2026-07-01** — the work
happened under #4547/#6279.

**The pre-to-post link ([#6096](https://github.com/marin-community/marin/issues/6096)).** Base-model BPB
on a `math_reasoning` PPL bundle (OpenMathReasoning cot+tir, target-only BPB, 256 docs/split, 4096 ctx)
was scored on the 10 SimpleRL-Zoo base models: cot BPB → post-RL Avg **r = −0.89, ρ = −0.87**; within
the 7–8B cohort **r = −0.97**. But the deflating caveat is in the same thread: "it predicts capability
*level*, not RL-friendliness. RL adds a near-constant **+15.8 ± 4.1** pts to everyone… **Predicting the
RL gain (post − base) fails: R² = 0.33**." The Coder/SWE-bench arm is not started.

---

## 7. What the ablations are scored on — and the known problems with the rulers

**The eval inventory.** `eval/uncheatable_eval/bpb` (BPB on post-cutoff freshly scraped text) is the
de-facto north star for mixture optimization; **Paloma** macro BPB/loss is the aggregate for
architecture and hero runs; **DCLM CORE v2** ([PR #6362](https://github.com/marin-community/marin/pull/6362))
and **OLMoBaseEval "Table-9"** (51 BPB components, from a 117-metric panel,
[#6611](https://github.com/marin-community/marin/issues/6611)) are the downstream aggregates; the
**111-task iso-FLOP speedup table** is what #6757 used.

**Measured noise floors (SNR = signal std / seed-noise std, 60M swarm vs 10 same-mix different-seed
runs)** ([#2345](https://github.com/marin-community/marin/issues/2345#issuecomment-4210192639)):
`uncheatable_eval/bpb` **13.69**, `paloma/c4_en/bpb` 13.47, `paloma/macro_bpb` **3.70**,
`sciq_5shot/acc_norm` **1.49**, `mmlu_sl_verb_5shot/bpb` 1.37. At 300M
([#5247](https://github.com/marin-community/marin/issues/5247)): `arxiv_physics/bpb` 44.64,
`github_python/bpb` 26.80, `uncheatable_eval/bpb` 25.04, versus smooth-MCQ proxies at 5–6. Verdict
there: "**Raw hard accuracy is too noisy for direct optimization at 300M**; use it for
validation/reporting."

**Four criticisms that a reader of these ablations should carry:**

1. **The proxy may point the wrong way.** From
   [#2351](https://github.com/marin-community/marin/issues/2351) (2026-07-03): "despite getting **lower
   uncheatable eval val loss we are getting lower benchmark scores at all scales**," reproduced on both
   the Marin CORE sweep and the OLMo Easy suite, with the author's own hypothesis "perhaps uncheatable
   eval val loss is not the right proxy for the downstream benchmarks." Echoed in
   [#2345](https://github.com/marin-community/marin/issues/2345): "The loss advantage on uncheatable
   evals does not translate as strongly to better benchmark / task performance as we hope" — which
   spawned [#5247](https://github.com/marin-community/marin/issues/5247), opening with "We have been
   **benchmaxxing perplexity evals** using the mixture swarms."
2. **MMLU accuracy is nearly uncorrelated with MMLU BPB** at 300M, n=240: Spearman **−0.069** (though
   `choice_prob_norm` reaches +0.866) ([#5247](https://github.com/marin-community/marin/issues/5247)).
3. **Easy factors get over-optimized.** "The factors that are more clearly explainable based on the data
   mix (e.g. math and coding) seem to end up **a bit over-optimized because they are the easiest to
   optimize**" ([#5362](https://github.com/marin-community/marin/issues/5362)) — which is exactly the
   shape of the #6757 result (code/math wins, commonsense losses). Winner's curse is tracked explicitly:
   every fit reports "lower-tail optimism" and "regret@1" alongside OOF RMSE.
4. **Paloma is contaminated.** "Paloma fails \[difficult to cheat\] rather than \[correlated with
   important things\]… Too contaminated since most of it comes from sources that we (and everyone else)
   either directly or indirectly sample from"
   ([#evals](https://discord.com/channels/1354881461060243556/1356487738840318002/1417598503110312051)).

---

## 8. Adjacent: tokenizer and per-source ablations

Included because they share the rulers and because they are a cautionary tale about reading any single
readout.

**[#6796 "experiment: tokenizer research"](https://github.com/marin-community/marin/issues/6796)** — a
SuperBPE bake-off judged on **feBPB**, a FLOP-equivalent BPB that prices serving cost at the ~250B-total
/ ~20B-active deployment MoE rather than the proxy, "since the vocab-dependent LM head is 30–50% of
FLOPs at the d1024 proxy but only ~2–3% at 20B active." The lock-down soak ran 8 tokenizer arms, one
grug-MoE run each at ~10B-total / ~500M-active on 64×H100, over SlimPajama 0.50 / Python 0.20 /
multilingual Wikipedia 0.20 / finemath 0.10, scored on Uncheatable macro BPB. **Rankings flipped
repeatedly as the common budget rose** (at 2.07e19 all five SuperBPE arms beat baseline; at 6e19 only
64k-digits was ahead at −0.9%), and then the whole soak was invalidated by three confounds — trained
SuperBPE arms had learned superword merges from 100% English web text, macro_bpb covered only 7
English+code subsets, and fertility was computed on 3 of 4 domains.

**There is no final SuperBPE verdict in this corpus.** The latest comment (2026-07-05, explicitly
labelled "interim, not the final verdict") has **marin-128k baseline at 0.919 BPB, best SuperBPE arm at
0.937 (+1.9%)** — but the baseline sits at 3.2e20 FLOPs against 7.6e19–1.9e20 for the new arms (not
iso-FLOP, which the author says flatters the baseline), and **feBPB, the metric that arbitrates the
trade-off, is not yet applied**. Durable signal that survived every correction: SuperBPE genuinely helps
C++ (−9.4% BPB) and genuinely hurts Python (+5.8%, mitigated to +3.4% by digit pretokenization) because
superwords corrupt significant whitespace.

**Sibling reversals.** [#5821](https://github.com/marin-community/marin/issues/5821)'s first readout had
tokenmonster-32k best everywhere; after a fix ("BPB was being accumulated with token/loss weights rather
than byte weights") tokenmonster became **worst** and gemma3-262k best — "the earlier d1024 W&B readout
… should be treated as superseded." And [#6571](https://github.com/marin-community/marin/pull/6571)'s
digit-variant numbers are under suspicion: "the **wrong model size was being passed in for digits
variants** which is likely giving the digits variants an unfair advantage of **+0.03 bpb in some
cases**" ([#tokenizer](https://discord.com/channels/1354881461060243556/1500987824206254120/1525252850408620142));
no rerun is in the corpus.

**Per-source ablations are thin and mostly old.** The classic cooldown ablations —
[#845](https://github.com/marin-community/marin/issues/845) (Wikipedia),
[#846](https://github.com/marin-community/marin/issues/846) (arXiv),
[#847](https://github.com/marin-community/marin/issues/847) (StackExchange) — are archived with null or
unreported results; #847's verdict is "**No major improvement compared to control run observed.**"
[#942](https://github.com/marin-community/marin/issues/942) (MegaMath annealing) has an empty Results
section and was auto-closed stale. The one *live* per-source comparison is
[#6570](https://github.com/marin-community/marin/issues/6570), a token-matched focus-crawl vs main-crawl
test on the grug-MoE d512–d1280 ladder scored on `eval/paloma/macro_loss`, partial at d512/d768: "the
focus crawl is much better on science/paper evals … while the **main crawl is better on broad web/news
and Paloma macro loss**." **No recent leave-one-out or add-a-source experiment with numbers exists in
this corpus.**

---

## 9. Gaps and open items

- **The next hero run's data mix is genuinely undecided in this corpus.**
  [#6700 "Data mix for pre–mid-training July hero run"](https://github.com/marin-community/marin/issues/6700)
  is title-only with an empty body and no comments, and the "Data mix" field in
  [#6689 \[Hero Run\] nB-AmB XT on B200s](https://github.com/marin-community/marin/issues/6689) is blank
  alongside FLOP budget `?` and Tokens `?`. [#6037](https://github.com/marin-community/marin/issues/6037)
  (datakit July-hero release) is still open on "new mix evaluated / new mix produced."
- **Corpus ceiling.** GitHub coverage in this frozen snapshot stops at roughly **#6967**. Several items
  the July-12 weekly summary discusses are therefore **not verifiable against primary sources here**:
  `mixing_via_embeddings` (#6969), the OLMix one-phase KL scaling issue (#6972), the "mixture sweep
  evidence is bucket-indexed" problem statement (#7067), the B200 data/architecture preregistration
  (#7073), and the proposed 40–60B-token medical midtraining corpus (#7128). The
  [week-of-July-6 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html) reports that
  #6969 priced a never-swept `dolma_starcoder` bucket at **0.9410 uncheatable BPB** against olmix-reuse
  0.9495, token-proportional 0.9759, and the sweep's own best 0.9554 — with a preregistered winner's-curse
  caveat of ~0.031 BPB — but I could not open the underlying issue, so treat that as summary-level
  evidence only. Note also that the closest in-corpus analogue, the MDE feature ablation
  [#6326](https://github.com/marin-community/marin/issues/6326), was **negative**.
- **Missing controls, named by the authors themselves:** no larger-scale point (≥ d1280) to resolve
  curated vs proportional ([#6757](https://github.com/marin-community/marin/issues/6757)); no
  no-math-mix control checkpoint for general-capability retention
  ([#6279](https://github.com/marin-community/marin/issues/6279)); no downstream-benchmark validation of
  any quality classifier ([#6739](https://github.com/marin-community/marin/issues/6739)); no ladder-level
  optimized-vs-baseline BPB table for the mixture surrogates
  ([#6602](https://github.com/marin-community/marin/issues/6602)/[#6611](https://github.com/marin-community/marin/issues/6611)).

---
<!--provenance-->
> *Data: marinmirror — 86367 chunks, built 0h ago · summaries through 2026-07-06_2026-07-12 (frozen
> eval corpus; no refresh triggered; GitHub coverage ends ≈ #6967).*
>
> *Query: "Can you share more details about the data ablations we've run so far? What data mixes / data
> classifiers were compared, on what training-data sizes and on what test data?"*
>
> *Sub-queries: "data mixture optimization ablations — qsplit240 swarm, OLMix KL, mixing-via-embeddings
> surrogate, baselines and proxy-run scales" · "data quality classifier ablations — high_quality curation
> vs baselines, label ceiling, DCLM CORE compute-matched, 111-eval curated default" · "production hero-run
> data mix — #6045 10T datamix, phase-1/phase-2 mixes, candidate mixes and the open B200 data decision" ·
> "mid-training / cooldown / post-training mix ablations — isoflops & Delphi ladders, medical midtraining,
> BPB-to-post-RL correlation" · "evaluation and test data for data ablations — uncheatable BPB, Paloma,
> DCLM CORE, Table-9, noise floors and scoring critiques" · "tokenizer and per-source ablations — SuperBPE
> / feBPB bake-off, digit variants, Nemotron tokenize-and-train, leave-one-source-out" · "adversarial
> verification of load-bearing numbers across all of the above"*
