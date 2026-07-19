# Data ablations run so far — mixes, classifiers, scales, and test data

Six distinct ablation families are live in the corpus. They differ in what they vary, and it matters
to keep them apart — "data ablation" at Marin covers *source extraction*, *mixture weights*,
*mixture policy class (curriculum)*, *mixture surrogates*, *document quality classifiers*, and
*cooldown/phase-2 mixes*. Below, each family with its arms, training scale, and eval surface.

Two framing facts up front, because they recur:

- **Almost nothing was ablated at production scale.** The 10T-token hero run is the *deployment*;
  every mix decision behind it was made at 3e17–1e21 FLOPs on 60M–3.4B proxy models.
- **The headline results are mostly negative or null.** The two biggest verdicts of the last month
  are "curated ≈ proportional, indistinguishable" ([#6757](https://github.com/marin-community/marin/issues/6757))
  and "our extraction wins on val loss but *loses* on benchmarks at all scales"
  ([#2351](https://github.com/marin-community/marin/issues/2351)).

---

## 1. Source extraction / curation pipeline ablation — #2351

**What was compared.** Six curation pipelines, each producing a training pool, trained head-to-head:

| arm | what it is |
|---|---|
| `curation-high_quality_10k` | Marin's own LLM spec-driven extraction |
| `curation-dclm_10k` | DCLM-baseline filtered pool |
| `curation-fineweb_edu_10k` | FineWeb-Edu classifier-filtered |
| `curation-nemotron_10k` | Nemotron-CC pipeline |
| `curation-resiliparse_10k` | near-raw Resiliparse extraction (baseline) |
| `curation-fineweb_cc_10k` | FineWeb CC extraction (baseline) |

Run by Michael Ryan (XenonMolecule), tracked on
[#2351](https://github.com/marin-community/marin/issues/2351) (opened 2026-01-16, latest substantive
comment 2026-07-03). The Marin arm re-extracts the DCLM pool: *"Extract the **10,364 WARC files**
that make up the DCLM 400m-1x pool with our high quality Spec"* — note the other five are
pre-existing filtered pools, not re-extractions, so extraction and filtering are partly confounded.

**Training scale.** `2e+21` FLOPs, shape `d1536 / L16 / B1024`, **TPU v5 (32 chips)**, ~4.3–4.4 days
per arm, `1.80e21` model FLOPs. This is the top rung of a full isoFLOP ladder running `1e+17` →
`2e+21`; at 2e+21 there are three shapes (`d1536-L16-B1024`, `d2432-L24-B256`, `d3584-L35-B128`), so
d1536 is one point on an isoFLOP curve, not "the" config. A seventh arm,
`curation-fastpipe_v3_40-expFM_natural-2e+21-d2432-L24-B256` (BPB 0.940), sits at a different shape.

**Test data.** Three eval passes per arm: DCLM CORE v2 (`agieval_lsat_ar, arc_easy, arc_challenge,
boolq, commonsense_qa, copa, hellaswag, lambada_openai, openbookqa, piqa, winogrande`), a second
10-task logprob set, and `mmlu_sl_verb`. Loss side: Paloma macro loss, LIMA loss, and uncheatable-eval
val loss. Decontamination rule: *"A document was removed if it contained even a single 15-word-gram
that also appears in a DCLM CORE v2 eval item — provided that 15-gram is distinctive."*

**Results (achieved).** Per-arm BPB at the 2e+21 rung, from the weekly run table
([summary 2026-06-29–07-05](https://mws.oa.dev/summaries/summary-2026-06-29_2026-07-05.html)):

| arm | BPB |
|---|---|
| `high_quality_10k` | **0.891** |
| `resiliparse_10k` | 0.934 |
| `dclm_10k` | 0.969 |
| `nemotron_10k` | 1.030 |
| `fineweb_cc_10k` | 1.044 |
| `fineweb_edu_10k` | 1.725 |

Three caveats that materially change how to read this table. **(a)** These numbers appear *only* in
the weekly-summary run tables — issue #2351 reports its own results as embedded plot images, so no
per-arm BPB is retrievable from the primary source. **(b)** The column is labeled only "BPB"; the
surrounding prose calls it "overall bpb", and uncheatable-eval macro BPB is the strong inference, but
no line in the corpus defines it. **(c)** `fineweb_edu` at 1.725 is anomalous — the same arm read
**1.519** at the 9e+20 rung and **1.441** at 1.21e21 before *rising* to 1.725. The prior week's
narrative explains the arm's weakness (*"FineWeb-Edu trails badly in aggregate (bpb 1.44), dragged
down by catastrophic code performance even as it stays competitive on prose"*) but not the reversal.

**The close-out is a negative result**, and it supersedes the BPB win. The 2026-07-03 comment on
[#2351](https://github.com/marin-community/marin/issues/2351) carries a section headed literally
*"(Negative Result) DCLM Benchmark Scores"*:

> "The plots make it clear as we scale what is happening... despite getting lower uncheatable eval
> val loss we are getting lower benchmark scores at all scales"

Four hypotheses were triaged. Eval misconfiguration: **disproven** (reproduced on the Marin CORE
sweep and OLMO Easy Eval). Missing decontamination of baselines: **unlikely** — *"Decontamination
doesn't seem like it could explain a 2-4% gap"* (this 2–4% is the gap *against* the Marin arm, the
only numeric magnitude stated in text). Val loss being a bad proxy, and domain-specific strengths:
both **plausible**. Worst task LAMBADA, best BoolQ. On why 3k-scale looked good: *"I was picking the
optimal point to evaluate based on uncheatable eval and methods like DCLM and Nemotron-CC hit optimal
val loss at way lower compute budgets at the 3k WARC scale."*

One clean win at a different scale (2.9B models, not the isoFLOP ladder): *"models trained on our
High-Quality extraction only produce toxic completions on **3.3%** of prompts compared to **5.1%**
with DCLM"* (RealToxicityPrompts / Detoxify).

---

## 2. Mixture ablation: curated vs proportional vs old mix — #6757

The single most decision-relevant data ablation, and the one behind the 10T hero-run mix.

**Arms** — three mixes over the same datakit pool `store_8ac06c74`
([#6757](https://github.com/marin-community/marin/issues/6757), Will Held, 2026-06-29):

> "3 mixes from datakit store `store_8ac06c74`: **curated**, **proportional** (natural-token weights),
> **nemotron** (baseline; simulated-epoching `ep10t`)."

These map to Will's public framing of *"our new data mix, our new data pool mixed proportionally, and
our old data mix"* ([Discord](https://discord.com/channels/1354881461060243556/1462895580064911522/1521595789783076985);
table at `oa.williamheld.com/datakit_sidebyside.html`).

**Training scale.** MoE, four iso-FLOP scales — *"d512 (3.82e17), d768 (2.81e18), d1024 (1.16e19),
d1280 (3.46e19)"* — on **v5p-16, us-east5**, W&B group `datakit-moe-sidebyside`. Nine cells total
(curated + proportional × 4 scales, plus nemotron d1280). **This is not a 10T comparison**: the mix
was *optimized* for a ~100× overtrained regime but *tested* at compute-optimal, a mismatch Will
flagged himself (*"the mix was optimized for 100X overtrained, so this is a weak setting"*).

**Test data.** *"111 runnable tasks/cell across all 9 cells, fully symmetric"* out of a 137-task mega
logprob set (the ~37 unrunnable are deprecated dataset-scripts + gated GPQA). Includes MMLU, GSM8K,
HumanEval, MBPP, ARC, PIQA, Winogrande, HellaSwag, BBH, MMLU-Pro, the 8 `math_*` subjects, plus
perplexity-gap bundles aggregated as Uncheatable Eval macro BPB. Metric is **per-axis iso-FLOP
effective speedup vs nemotron**, extrapolated through a 4-point scaling fit with the per-task exponent
floored at the macro exponent 0.0941 (from the MoE isoFLOP law in
[#6074](https://github.com/marin-community/marin/issues/6074)).

**Results (achieved).** *Curated vs the old nemotron mix* — curated wins on aggregate loss and
code/math, and **significantly loses** on commonsense:

- ✅ *"MACRO uncheatable bpb **1.78× (90% CI [1.45, 2.24])**, clearly significant."*
- ✅ HumanEval 10-shot [37, 168], GSM8K 5-shot [2.3, 16], MBPP [2.8, 31], all 8 `math_*` subjects.
- ⚠️ *"hellaswag [0.48, 0.80], winogrande [0.55, 0.94], piqa [0.59, 0.75], openbookqa, race, swag —
  real, below 1×."*

*Curated vs proportional* — the actual head-to-head — is **null**:

> "**Finding: curated and proportional are statistically indistinguishable at this compute.**
> Aggregate loss (macro uncheatable bpb): curated **1.14× [0.93, 1.41]** — not significant (a tie).
> Of **102** downstream tasks, only **3** separate the two... with 102 tasks at a 90% CI you'd expect
> ~10 false positives by chance — we found 3, i.e. *fewer than chance*."

The recorded recommendation, verbatim: *"for AA-II, take **curated** as a weak default — but the
honest result is that at this scale the mix is not where AA-II gains will come from."* Roughly
**70% of per-task mix differences read as noise** at 3e17–3e19 FLOPs, *"most downstream tasks aren't
learned yet."* Significance is a 90% CI on the log-log fit residual (Student-t, n−2 = 2 dof), *not* a
per-example bootstrap — deliberately, since per-cell stderr (~0.009) is ~10× smaller than fit residual.

Note the DoD in [#5359](https://github.com/marin-community/marin/issues/5359) required the optimized
mix be *"better than (or worst case equivalent to) proportional data mixing"* — the verdict landed
exactly on "worst case equivalent." Both [#5359](https://github.com/marin-community/marin/issues/5359)
and [#6045](https://github.com/marin-community/marin/issues/6045) are targets/specs carrying **no
measured results**; [#6713](https://github.com/marin-community/marin/issues/6713) is a milestone
umbrella. The result lives in #6757.

---

## 3. Curriculum ablation: one-phase vs two-phase mixture policy — #6607/6608/6609/6602

Despite the name, "curriculum" here is **not** a data-ordering sweep. The 80/20 WSD schedule is fixed
(`PHASE_BOUNDARIES = [0.8]`, phase 2 = decay); what varies is whether the *mixture policy class* is
one-phase or two-phase, on both the fitting and deployment side. Calvin Xu's own 2×2
([#6609](https://github.com/marin-community/marin/issues/6609#issuecomment-4813873642)): **{OLMix, DSP}
× {one-phase, two-phase}**, against objective-agnostic baselines (proportional, UniMax-8).

**Training scale** ([#6607](https://github.com/marin-community/marin/issues/6607) body): *"39
Dolma3/Dolmino top-level buckets"*; rungs `3e18, 2e19, 3e20, 1e21` FLOPs; TPU ladder `v5p-8, v5p-16,
v5p-32, v5p-64`; east5 only. Measured shapes: 3e18 = 358M params / 1.577B tokens; 1e21 = 3.38B params
/ 46.26B tokens. (Rung→TPU pairing is the natural reading of two parallel lists, not an explicit quote.)

**Test data.** `eval/uncheatable_eval/bpb` — perplexity on post-cutoff text (arxiv cs/physics, bbc_news,
github cpp/python, wikipedia_english, ao3_english) so it can't be contaminated. Plus Paloma
macro BPB and the OLMoBaseEval Table-9 51-component macro BPB.

**Results (achieved), `eval/uncheatable_eval/bpb` at the 1e21 rung** — every value below re-verified
directly against the W&B run record:

| arm | 3e18 | 2e19 | 3e20 | 1e21 |
|---|---|---|---|---|
| [proportional](https://wandb.ai/marin-community/marin/runs/proportional_1e21-2f1a48) | 1.0383 | 0.9022 | 0.7699 | 0.7214 |
| [UniMax-8](https://wandb.ai/marin-community/marin/runs/unimax8_1e21-d685cd) | 1.0223 | 0.8918 | 0.7615 | 0.7150 |
| [OLMix **two**-phase](https://wandb.ai/marin-community/marin/runs/olmix_d001_kl005_cap4_1e21-e79ef0) | 1.0508 | 0.9314 | 0.8106 | 0.7643 |
| [OLMix **one**-phase](https://wandb.ai/marin-community/marin/runs/olmix_onephase_uncheatable_d001_kl005_cap4_1e21-72de81) | 1.0039 | 0.8736 | 0.7485 | 0.7026 |
| [DSP effexp **two**-phase](https://wandb.ai/marin-community/marin/runs/dsp_effexp_kl01_1e21-450566) | 0.9908 | 0.8638 | 0.7377 | **0.6929** |
| [DSP effexp **one**-phase](https://wandb.ai/marin-community/marin/runs/dsp_onephase_effexp_uncheatable_kl0p1_1e21-d0e715) | 0.9907 | 0.8643 | 0.7389 | 0.6947 |

Three readings, none of which appear in any issue comment — the threads stop at "running/pending":

1. **Two-phase OLMix loses to plain proportional at every rung** (0.7643 vs 0.7214 at 1e21). Its
   *predicted* 300M BPB of 0.7757 vs proportional 0.9908 did not transfer.
2. **One-phase OLMix beats two-phase at every rung** (0.7026 vs 0.7643) — the direct curriculum answer.
3. **For DSP the one/two-phase gap is ~0.002 BPB, i.e. nil.**

Important fairness caveat: the one-phase and OLMix/DSP arms are tagged
`delphi-uncheatable-optimized-mixtures` — they were optimized *against uncheatable eval itself*, so
comparing their BPB to objective-agnostic proportional/UniMax-8 is not apples-to-apples.

On OLMoBaseEval Table-9 macro BPB at 1e21 the same shape holds: one-phase T9-optimized OLMix 0.6044 /
0.5973 beats two-phase T9-optimized 0.6146, both beat proportional 0.6731.

**Status of the tracking issues.** [#6801 "Ablation: curriculum: Olmix"](https://github.com/marin-community/marin/issues/6801)
is an **empty stub** — its entire body is a pointer to #6609, and the triage bot declined it as *"Not
actionable as-is — need scope."* [#6609](https://github.com/marin-community/marin/issues/6609)'s
latest comment is a *submission*, not a result. So: the runs completed, but the OLMix curriculum
ablation was never written up. Canonical DSP was stopped mid-ladder for
*"scaling worse than DSP effective-exposure on the intended target"*
([#6602](https://github.com/marin-community/marin/issues/6602#issuecomment-4803563305)).

The interpretive close-out is [#6931 "Phase Literature Audit"](https://github.com/marin-community/marin/issues/6931)
(2026-07-04): *"they do not imply that a two-phase schedule must substantially beat the best
single-phase mixture for broad smooth objectives... Our current two-phase solver has not reliably
harvested a transferable phase-asymmetry advantage."* The proposed decisive test — a matched-exposure
phase-order experiment holding aggregate exposure fixed — **has not been run**.

---

## 4. Mixture-surrogate ablations (predicting mixes without training them)

**The swarm.** `qsplit240` — Calvin Xu's proxy swarm,
[#2345](https://github.com/marin-community/marin/issues/2345): *"A swarm consisting of **238 proxy
runs**... Model size is **60M**, trained for **1.2B tokens** (Chinchilla). Phases are 80% / 20%...
Domains are: 26 Dolma 3 Common Crawl domains formed by taking 13 retained topics and splitting each
into `high` and `low` quality buckets, 4 other Dolma 3 domains, 9 Dolmino domains."* Later, a
**840-row D-optimal** `swarm_fisher_dsp` swarm at 300M, of which 241 rows fed the DSP fit.

Two important negative results from the swarm itself:

- **MMLU can't be optimized at proxy scale**: *"we basically cannot reasonably fit on or optimize that
  (R² < 0.1), even though we can fit reasonably on C4EN BPB (R² > 0.7)... variance of one fixed data mix
  with different seeds is almost as large as the variance of the entire swarm"*. Scaling to 300M/3B
  didn't reduce variance.
- **Early checkpoints don't predict final rank**: *"60M/1.2B: rank Spearman at first eval (20%) vs
  final: 0.309   300M/6B: rank Spearman at first eval (13%) vs final: 0.225"*
  ([Discord](https://discord.com/channels/1354881461060243556/1462895580064911522/1509261264457699459)).

**MDE feature ablation — measured, negative.**
[#6326](https://github.com/marin-community/marin/issues/6326) (Calvin Xu, 2026-06-11) trained *"39
cap-1 single-domain 300M vertex experts"* and compared Mixtures-of-Data-Experts features against DSP.
Out-of-fold Spearman on `eval/uncheatable_eval/bpb`: `vertex_mde_ridge` **0.8605** vs **DSP 0.9140**;
on TruthfulQA 0.1005 vs `phase_log_exposure` 0.3237. Verdict: *"the current MDE aggregation mostly
reparameterizes mixture/exposure geometry. It does not add enough independent signal to justify using
it in the main optimizer."*

**Embedding surrogate — the newest and most promising, but summary-only.** `mixing_via_embeddings`
(#6969) re-featurizes a mixture by the content it induces, so it can price a bucket never seen in the
sweep. Per the [2026-07-06–07-12 weekly summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html):

> "a live pre-registered test priced a genuinely never-swept bucket, dolma_starcoder, proposed a
> mixture, and realized **0.9410** uncheatable-eval bits per byte (BPB) against olmix-reuse **0.9495**,
> token-proportional **0.9759**, and the sweep's own best-ever run **0.9554** (a −0.0145 margin the
> significance check put at roughly 8σ versus measured 300M repeat noise)."

With a preregistered caveat: *"the optimized point was optimistic by about **0.031 BPB** (winner's
curse), so a trust-region / lower-confidence-bound proposer is the flagged next step."* The motivating
problem is #7067: *"names the structural weakness in how the team stores data-mixing evidence: it is
bucket-indexed... bucket #7 is just a name and the surrogate has no representation of what the bucket
contains."*

**Retrieval caveat, stated plainly:** the frozen GitHub mirror tops out at **#6966**. Issues #6969,
#6972 and #7067 are *not* retrievable as primary sources in this corpus — the figures above are
summary-level and could not be checked against the issue bodies or their comments.

---

## 5. Document quality-classifier ablations — #6739

**Arms compared**, all on the same held-out set and harness
([#6739](https://github.com/marin-community/marin/issues/6739), rjpower, 2026-06-28; PR
[#6741](https://github.com/marin-community/marin/pull/6741)):

| classifier | AUC | Spearman ρ | FLOPs/tok |
|---|---|---|---|
| fasttext v0 `sonnet46-thr05` (deployed) | 0.846 | 0.641 | ~0 |
| **fast-transformer** meanmaxmin·w64·d512·L4 | **0.875** | **0.703** | 0.41M |
| `L=0` (pooling + head, no attention) | 0.868 | 0.699 | 12K |
| `neural-bow` (mean-pool) | 0.861 | 0.675 | ~0 |
| **source-of-origin prior (reads no text)** | 0.852 | — | 0 |
| dolma3 off-the-shelf | — | 0.168 | — |

**Training data for the classifier**: oracle labels from **claude-sonnet-4-6** on a 1–5
pretraining-utility rubric, 7,000 docs stratified over 104 datakit sources → 5,613 usable train labels
+ 961 holdout ([#5810](https://github.com/marin-community/marin/issues/5810)). Ran on a single
**v6e-4** slice, "in minutes."

**The headline is a label ceiling.** Three free weak teachers all failed to break ρ≈0.70 / AUC≈0.87 —
source-prior (0.858), nemotron-bucket (0.814), FineWeb-Edu score (0.833), each *below* from-scratch
training. *"You generally can't distill your way above a teacher weaker than your student... the
plateau is a genuine label-quality ceiling."* Source-of-origin alone explains **η² = 0.41** of oracle
variance.

**Critical caveat, from the author**: *"we validate against the oracle, not downstream — and the
oracle is edu-leaning."* rjpower notes production filters (FineWeb-Edu, Nemotron-CC, DCLM) are all
validated by downstream benchmark ablation, never AUC, and concludes: *"The honest test of a quality
filter is a small-model data ablation across a CORE-style suite, not oracle-AUC."* **No downstream
training run has validated these classifiers.**

**Deployed-classifier pathologies** (all ravwojdyla-agent, 2026-07-02, against `store_8ac06c74`):

- [#6849](https://github.com/marin-community/marin/issues/6849) — sorts by *domain*, not quality:
  clean `starcoder2/ir_python` scores ~0.00 (88–99% of docs exactly 0); bucket q4 is code+math only.
  Same root cause as the #6739 ceiling: source identity predicts the oracle at AUC 0.852, so any
  faithful distillation reproduces the bias. Also: no absolute anchor — *"`0.85` means 'excellent
  arXiv' in one place and 'a fan-wiki parody cast list' in another."* Claimed fix on branch
  `rav/quality-coherence` lifts Spearman 0.44 → 0.69 (branch only, no merged PR in corpus).
- [#6860](https://github.com/marin-community/marin/issues/6860) — **12 of 98 sources have score
  stddev < 0.1**; `massive_function_calling` has stddev 0.0, one distinct value.
- [#6859](https://github.com/marin-community/marin/issues/6859) — quality decided from
  `MAX_TEXT_CHARS = 4000`; docs >100 KB are 14% of `starcoder2/documentation` docs but **80.5% of its
  characters**.

**Web-graph centrality** ([#6750](https://github.com/marin-community/marin/issues/6750)) is a
**scoping issue only — no experiment run**. The one substantive reply (wumpus) notes centrality is
already used upstream at crawl-budget granularity: *"We use a particular form of centrality, harmonic
centrality, to create a rank for every host and domain. We then use the host rank to set the host
crawl budget"* — host/domain level, not per-document.

**Adjacent diagnostic** ([#6096](https://github.com/marin-community/marin/issues/6096), Will Held):
base-model BPB on OpenMathReasoning `cot`/`tir` traces predicts post-RL math across n=10 SimpleRL-Zoo
bases — Pearson **r = −0.89** (cot) / **−0.92** (tir), and **−0.97** within the 7–8B cohort. The
deflating caveat is stated by the author: *"RL adds a near-constant +15.8 ± 4.1 pts to everyone"*, so
predicting the *RL gain* fails (R² = 0.33). The Coder/SWE-bench arm is **planned, not run**.

---

## 6. Cooldown / phase-2 mix, and midtraining contamination

**The hero-run phase-2 mix.** The 67B-A2B 10T run
([#6044](https://github.com/marin-community/marin/issues/6044)) uses a 2-block schedule over 191
datakit components; phase 2 begins at step 127,500 of 157,500 — the last **30,000 steps ≈ 19% of
training ≈ 2.01T tokens**. L1 distance between phases = 0.401. Larry Dial's characterization:
*"swapping to datamix-phase2 + seqlen 8k->65k. **Phase2 mix tends to have higher loss.**"* Bucket
semantics (`cXXqY`) are never mapped to human-readable domains in the corpus, so I won't assert a
direction on "quality."

**The intermediate cooldown — and a number worth getting right.** The widely-cited **2.277 Paloma
macro-loss** result is real ([W&B run](https://wandb.ai/marin-community/marin_moe/runs/moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step39k),
macro_bpb 0.8242) but it is **not** the endpoint of the 2.269 preregistration, and reporting it as
"beat the target" would be wrong on three counts: it is at ~2.83T tokens (not 8T); it is a *cooled*
checkpoint while the preregistration was for the *uncooled* stage-1 endpoint; and it evaluates at
`max_seq_len=65536` where the preregistration specified *"seqlen 8192 on 1024 sequences per eval"*
([#6044](https://github.com/marin-community/marin/issues/6044#issuecomment-4820008980)). Seqlen
sensitivity is material — Larry: *"8k looks 0.02 better, which is quite a lot, maybe equal to 30%
compute."* The preregistered 2.269 itself is a 3-point extrapolation whose value swings with the
assumed irreducible loss (E=1.3 → 2.258; E=1.6 → 2.295). No comment in the corpus reconciles the two.
At freeze the uncooled main run sat at **2.3655** macro loss at 3.58T tokens.

**A YaRN mscale mini-ablation** ran alongside (5 arms, all at 2.6186T tokens): mscale01 won at 2.3755
Paloma macro loss vs mscale00 2.3846 — and is what the production cooldown used.

**Midtraining validation contamination** ([#6742](https://github.com/marin-community/marin/issues/6742),
ahmeda14960) is the most consequential eval-integrity finding. Delphi p33m67 math midtraining looked
mispredicted by **+19%** at 1e22 FLOPs. Cause: fuzzy near-dup leakage into the val split — **0 exact
duplicates**, but at Jaccard ≥ 0.75, **9,757 / 57,243 validation docs** had a train near-dup. Rebuilt
against a clean seen-set, error fell **+18.56% → +2.83%**. Recommendation: *"Treat the old 4plus
validation anchor as a contaminated historical baseline."* Separately, 13-word n-gram containment
found **28.20% of MATH500 test** and **56.67% of AIME24** present in Nemotron-CC-Math 4plus — *"nemotron
cc math claims they explicitly deduped against gsm8k and math500!"* Magpie-Pro-300K SFT: no contamination.

**Historical cooldown-mix ablations** (2025, dense era):
[#820](https://github.com/marin-community/marin/issues/820) landed on 70% DCLM + 15% FLAN + 15% HQ as
a Pareto improvement (HQ beat DCLM on Paloma macro but not downstream);
[#934](https://github.com/marin-community/marin/issues/934) ran 4 arms judged on Paloma loss, Tulu3
NLL and MMLU — *"Definite improvements across the board from dolmino. Small but persistent improvement
from adding in our stuff too."*

---

## 7. Cross-cutting: what "test data" actually means here

- **Uncheatable Eval macro BPB** — the primary optimization target. Post-cutoff text (arxiv cs/physics,
  bbc_news, github cpp/python, wikipedia_english, ao3_english) so it can't be contaminated.
- **Paloma macro loss / macro BPB** — 16 sources (c4_en, dolma-v1_5, redpajama, m2d2, ptb,
  wikitext_103, 4chan, gab, twitterAAE, …). Known artifact: `twitterAAE_HELM` loss *rises* while
  everything else falls — dlwh: *"Not worth worrying about. Means you're fitting the training
  distribution better"* — but it materially inflates the macro (bpb 2.214 in the cooldown run).
- **OLMoBaseEval Easy / Table-9** — 51-component BPB macro; now scored natively in Levanter on TPU
  ([PR #6726](https://github.com/marin-community/marin/pull/6726)) at ~1e-7 parity with the SC oracle.
- **DCLM CORE v2** and the **137-task mega logprob suite** (111 runnable/cell) — downstream accuracy.
- **RealToxicityPrompts / Detoxify** — used only in the #2351 extraction ablation.

The recurring theme across all six families is that **loss-side wins do not transfer to benchmarks**.
#2351: better val loss, worse CORE at all scales. #6757: 1.78× aggregate BPB speedup, but commonsense
regressions and ~70% of tasks pure noise. The swarm: *"no free lunch if you care about HellaSwag or
Paloma macro BPB — both are maxed by proportional mixing."* #6739: AUC gains never validated
downstream at all.

---

## 8. Gaps and things the corpus does not support

- **No production-scale mix ablation exists.** Everything is 3e17–1e21 FLOPs; the 10T run is a
  deployment, not an arm. #6757 explicitly asks for a larger-scale point (≥ d1280) that had not landed.
- **#2351 reports no per-arm BPB in retrievable text** — results are plot images; the numbers in §1
  come from weekly-summary run tables, whose BPB column is never formally defined.
- **The OLMix curriculum ablation was never written up** — runs completed in W&B, #6801 is an empty
  stub, #6609 ends at a submission comment. The §3 table is an aggregation of run records, not a
  published result.
- **#6969 / #6972 / #7067 postdate the GitHub freeze (max #6966)** — summary-level only, unverifiable
  against primary sources here.
- **No downstream validation of any quality classifier**, which rjpower names as the only honest test.
- **The matched-exposure phase-order experiment (#6931) and the Coder/SWE-bench arm of #6096 are both
  planned, not run.**

---
<!--provenance-->
> *Data: marinmirror — 68026 chunks, built 45h ago · summaries through 2026-07-06_2026-07-12
> (frozen eval corpus `MARIN_EVAL_FREEZE=2026-07-16`; no refresh this run). Note: the GitHub mirror
> tops out at #6966, so issues above that number are reachable only via weekly summaries.*
>
> *Query: "Can you share more details about the data ablations we've run so far? What data mixes /
> data classifiers were compared, on what training-data sizes and on what test data?"*
>
> *Sub-queries: "curation ablation #2351: HTML extraction and quality-classifier comparison at 2e+21
> FLOPs" · "10T data mix optimization: optimized vs proportional vs old Marin mix across 111 evals" ·
> "mixing_via_embeddings embedding surrogate #6969; bucket-indexed mixture evidence #7067" ·
> "curriculum ablations: Olmix #6801, uncheatable eval #6609, scaling baseline mixtures #6607/#6608" ·
> "data-selection diagnostics #6712: fast-transformer vs fasttext #6739, web-graph centrality #6750,
> base BPB predicts post-RL #6096" · "cooldown / midtraining / phase-2 mix ablations; Delphi
> midtraining validation contamination #6742" · verification passes on #6969/#7067 corpus coverage,
> the 1e21 curriculum-ladder W&B values, and the #2351 / #6757 headline claims.*
