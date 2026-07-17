# Data ablations so far: mixes, classifiers, training-data sizes, and test data

The recorded ablations split into two tracks that feed each other, and a third layer that
gates whether either track's numbers can be believed:

1. **Data-mixture ablations** — which blend of sources, in what proportions (and in what phase
   order), trains the best model.
2. **Data-classifier / data-selection ablations** — which document filter or feature (quality
   classifier, embedding, web-graph centrality) picks the best data.
3. **Test-data integrity** — decontamination and eval-choice work that changes the scores
   the first two tracks are judged on.

All 2026 work is organized under two epics: **[#6713 Pretraining data curation & mix](https://github.com/marin-community/marin/issues/6713)**
(ihodes, 2026-06-26 — "Curate and tune the pretraining data mix, and productionize the CoreWeave
data pipeline") and **[#6712 Data-selection diagnostics](https://github.com/marin-community/marin/issues/6712)**
(ihodes, 2026-06-26 — "Strengthen the evals / PPL / diagnostic signals that predict downstream
quality"). Older seed experiments (2024–early-2025) are marked as vintage where they appear.

> **Provenance conventions.** Every number is tagged *achieved* / *target* / *predicted*. Where a
> result's primary GitHub thread post-dates the 2026-07-16 freeze (e.g. #6969, #6972, #7067,
> #7127, #7128), it is cited to the **weekly summary
> [2026-07-06_2026-07-12](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)**, the
> newest source in the frozen corpus. Per-rung W&B training scalars (e.g. the achieved one-phase
> BPB ladder) are **not** in the frozen text corpus and are described structurally, not quoted.

---

## 1. Data-mixture ablations

### 1a. Datakit curated vs proportional vs nemotron — iso-FLOP MoE side-by-side ([#6757](https://github.com/marin-community/marin/issues/6757))

The primary recent head-to-head on *how to weight a mix* (Will Held / Helw150, agent-generated
tracking issue under epic #6713, methodology from #6711/#6074, closed 2026-06-30).

- **Mixes compared (3):** `curated`, `proportional` (natural-token weights), and `nemotron`
  (baseline, simulated-epoching `ep10t`) — all from rav's east5 datakit store `store_8ac06c74`.
- **Training-data sizes (4 iso-FLOP scales):** MoE, EP=1, **d512 (3.82e17), d768 (2.81e18),
  d1024 (1.16e19), d1280 (3.46e19) FLOPs**. Headline verdict is the **v5p-16 us-east5
  reproduction** against the east5-mirrored datakit (original side-by-side was v4-32 / us-central2);
  W&B group `datakit-moe-sidebyside`. All 8 curated/proportional cells + nemotron d1280 trained
  (d1280×2 hit a teardown SIGABRT only; checkpoints committed, executor SUCCESS).
- **Test data:** PPL → macro uncheatable BPB; lm-eval logprob headline (mmlu, gsm8k, humaneval,
  arc easy/challenge, piqa, winogrande, hellaswag, bbh, mmlu_pro) **+ a 137-task mega set → 111
  runnable tasks/cell across all 9 cells, fully symmetric** (~37/cell unrunnable = datasets≥4.0
  deprecated scripts + gated gpqa).
- **Significance method (load-bearing):** each iso-FLOP effective-speedup is an extrapolation
  through a 4-point scaling fit, so uncertainty = **lack-of-fit** — a 90% CI from the log-log fit
  residual (Student-t, n−2 = 2 dof); "significant" = CI excludes 1×. Per-example stderr
  (`choice_logprob ≈ 0.009`, `bpb ≈ 0.004`) is ~10× smaller and deliberately *not* used, to avoid
  calling at-chance MMLU "significant." ~70% of per-task differences are noise at 3e17–3e19 FLOPs.

**Achieved results:**
- **Curated vs nemotron (significant cells):** macro uncheatable BPB **1.78× [1.45, 2.24]**;
  code/math win big — HumanEval 10-shot [37, 168], GSM8K 5-shot [2.3, 16], MBPP [2.8, 31], all 8
  `math_*` subjects ≫1×. **Curated significantly *loses* commonsense** — hellaswag [0.48, 0.80],
  winogrande [0.55, 0.94], piqa [0.59, 0.75], openbookqa/race/swag. MMLU + all multilingual = noise.
- **Curated vs proportional (the actual question):** aggregate macro uncheatable BPB
  **1.14× [0.93, 1.41] — NOT significant (a tie)**. Of **102** downstream tasks only **3** separate
  them (curated>proportional HumanEval 4.2× [1.6, 14]; proportional>curated copal-id Indonesian
  commonsense 0.30× / 0.69×) — fewer than the ~10 false positives expected by chance.
- **Recorded verdict:** *"curated and proportional are statistically indistinguishable at this
  compute"* — take **curated as a weak default** for a reasoning/code/math (AA-II) target, an
  explicit tie-breaker; *"the mix is not where AA-II gains will come from,"* and a confident call
  "would need a larger-scale point (≥ d1280)." Both mixes beat nemotron on loss+code/math and both
  regress on commonsense, so those don't differentiate them.

### 1b. OLMix vs DSP mixture optimization on the Delphi ladder (Calvin Xu, June–July 2026)

The most systematic mixture-optimization program: fit a mixture on a 60M/300M swarm, then scale it
on the **Delphi scaling ladder — 3e18 / 2e19 / 3e20 / 1e21 FLOPs** (TPU ladder v5p-8/16/32/64,
east5 only) and check that the optimized mixture holds up. Common basis across the threads: **39
Dolma3/Dolmino top-level buckets**, two separately-normalized phase-weight vectors, phase schedule
fixed at **0.8/0.2** (`PHASE_BOUNDARIES=[0.8]`).

**Objective-agnostic anchors — [#6607](https://github.com/marin-community/marin/issues/6607).**
`proportional` and `UniMax-8` over the 39 buckets, two-phase with `phase_0==phase_1`, 4 rungs, with
downstream incl. **DCLM Core v2** as follow-up. These are scaling anchors, not optimized mixtures.

**Uncheatable-BPB fit — [#6608](https://github.com/marin-community/marin/issues/6608).** Target
`eval/uncheatable_eval/bpb`. Fit panel = **280 rows** (241 ex-ante qsplit/signal + 39 300M
domain-deletion rows; the adaptive `baseline_olmix_loglinear_uncheatable_bpb` row is excluded;
proportional target = mean of 11 proportional obs, **mean BPB 0.990787**). Selected OLMix candidate
via a Huber-delta sweep: **`olmix_d001_kl005_cap4`** — two-phase, Huber **delta=0.01, KL=0.05,
aggregate rep-cap=4**, predicted BPB 0.7768, max simulated epoch 4.0.

**Table-9 macro fit — [#6611](https://github.com/marin-community/marin/issues/6611).** The
objective pivoted from single-target uncheatable BPB to the **paper-style 51-component OLMoBaseEval
Easy Table-9 macro** (`table9_macro_bpb = mean_51_table9_bpb_components`). On the 300M fit,
**effective-exposure DSP `linear_reg=1e-4` gives OOF-Spearman 0.918 / RMSE 0.01284**, beating
paper-faithful **`single_tied` OLMix (Spearman 0.380)** and the **`two_phase_adapted`** Marin
extension (0.891). Reporting rule fixed here: paper-faithful OLMix = `single_tied`; `two_phase_adapted`
is a *stronger Marin extension*, not the OLMix-paper baseline. (Panel of 736 runs × 117 BPB metrics
across 60M/1.2B and 300M/6B swarms.)

**June two-phase panel closed/superseded.** The earlier "3 mixtures × 4 scales" two-phase panel
(canonical-DSP + effexp-DSP + OLMix) is dead: canonical-DSP scaling was **stopped at 3e20/1e21**
after scaling worse than effexp-DSP, and the entire two-phase uncheatable-OLMix parent
(`…dm-delphi-uncheatable-olmix-rest-20260625-141100`) was **`JOB_STATE_KILLED`** because it was
superseded by the Table-9 objective and kept preempting Table-9 rows (only `olmix_d001_kl005_cap4_2e19`
was preserved) ([#6608 close-out](https://github.com/marin-community/marin/issues/6608)). Do **not**
present that panel as current.

**Current live one-phase validation — [#6609](https://github.com/marin-community/marin/issues/6609).**
The curriculum question crystallized as one-phase vs two-phase (a 2×2 over {OLMix,DSP}×{1-phase,
2-phase}). The live main panel (single Iris parent `…/dm-delphi-onephase-uncheatable-scaling-20260629`,
submitted 2026-06-30) compares:
- **OLMix one-phase uncheatable:** delta=0.01, **KL=0.05**, cap=4.
- **DSP one-phase effective-exposure uncheatable:** **LINEAR_REG=0.01, KL=0.1**.
- 4 Delphi scales each, with **native OLMoBaseEval Table-9** attached after each HF export.
Its sibling stub [#6801](https://github.com/marin-community/marin/issues/6801) ("Ablation:
curriculum: Olmix") tracks the OLMix arm. The phase-order pivot is motivated by
[#6931 (Phase Literature Audit)](https://github.com/marin-community/marin/issues/6931): the
literature (*Replaying pre-training data…*, *The Finetuner's Fallacy*) motivates phase-order tests
but does **not** rule out single-phase near-optimality for broad smooth objectives; a clean test
must hold aggregate exposure fixed and vary only placement, and Marin's internal StarCoder/Nemotron
landscape shows the optimum near the single-phase diagonal.
*(The achieved per-rung one-phase BPB ladders are W&B training scalars, not in the frozen text
corpus.)*

**Target-specific KL — [#6972] (summary-sourced).** Calvin Xu opened #6972 to scale the best
one-phase OLMix KL settings from the 3e18 sweep, which **showed the best KL is target-specific —
KL=0.1 for uncheatable BPB, KL=0.005 for OLMoBaseEval Table-9 macro BPB — launching both winners at
2e19, 3e20, 1e21** with native Table-9 eval
([summary 2026-07-06_2026-07-12](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).
The DSP Table-9 KL (0.025) was already deliberately lower than the DSP uncheatable KL (0.1) in #6611
— the first hint of this story. #6972's write-up post-dates the freeze; treat the KL claim as
summary-recorded.

### 1c. Embedding-based mixture surrogate — [#6969] / [#7067] (summary-sourced)

The week's headline mixture-optimization result (Rafal Wojdyla; primary threads post-date the
freeze — all specifics from the
[2026-07-06_2026-07-12 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).
**[#7067]** names the structural weakness: the mixture surrogate is **bucket-indexed** — it learns
"weight on bucket #7 → loss" but has no representation of what bucket #7 *contains*, so any dedup /
relabel / source-refresh / re-partition silently strands the evidence, new data has no coordinates,
and swarms under different bucket eras can't be pooled. **[#6969] (mixing-via-embeddings)**
re-featurizes a mixture as the token-weighted histogram it induces over a frozen embedding codebook
(**h = V·w**) and learns the surrogate over that content distribution, so bucket identity is consumed
in featurization, old sweeps re-featurize under any future bucketing, and a new dataset is priced
from one embedding pass + a CPU refit. It reuses the **qsplit240 swarm (~240 mixtures × 2 scales,
hundreds of proxy runs)** — the same ex-ante sweep behind #6608's 280-row fit.

**Preregistered validation.** Stage-1 retrodictive gates (information audit, leave-one-domain-out
content→domain-value gate, held-out-dose retrodiction; each vs shuffled + matched-random controls
under a two-basis kill rule) **all passed**. Stage-2 live test priced a **genuinely never-swept
bucket, `dolma_starcoder`**, and realized:

| Mixture | Uncheatable-eval BPB (achieved / realized) |
|---|---|
| **embedding surrogate → dolma_starcoder** | **0.9410** (winner) |
| olmix-reuse | 0.9495 |
| sweep's own best-ever run | 0.9554 |
| token-proportional | 0.9759 |

The **−0.0145 BPB** margin vs the sweep-best was **~8σ vs measured 300M-repeat noise**, at a cost of
one embedding job + minutes of CPU + three validation runs (no re-sweep). Preregistered caveat: the
optimized point was **optimistic by ~0.031 BPB (winner's curse)** — 0.9410 is already the honest
realized number, and the flagged next step is a trust-region / lower-confidence-bound proposer. Will
Held pushed the swarm's sweep pointers to a public HuggingFace dataset. This lives under epic #6712.

### 1d. Curation reality check — high_quality vs dclm_random ([#2351](https://github.com/marin-community/marin/issues/2351))

"Small Model for Raw Web Data to Training Tokens" (Michael Ryan / XenonMolecule, opened 2026-01-16,
weekly-sync thread through 2026-07-03). Replaces multi-stage curation with a single LLM call per HTML
doc conditioned on a natural-language spec (distill GPT-OSS-120B → Qwen3-8B). Compares the
LLM-extracted **`high_quality`** spec vs **`dclm_random`** (random-WARC/DCLM baseline), on a
**held-fixed WARC pool**.

- **Training-data sizes:** WARC-budget scaling **{100, 500, 1000, 2000, 3000} WARCs** (~8B tokens at
  3000 WARCs for LLM extraction vs only ~120M DCLM-filtered tokens on the same 3000 WARCs), then a
  **10k-WARC random sample** (10k = DCLM's minimum release size). Original DoD: 400M-param model on
  ~8.2B tokens; scaling ladder (W&B group `marin-dclm-core`) spans 3e18 → 2e21 FLOPs.
- **Test data:** DCLM CORE / **COREv2**, plus uncheatable-eval val loss and LIMA loss; corroborated
  by a Marin CORE sweep and the OLMO easy-eval suite.
- **Recorded reversal (close-out negative result):** on the 10k-WARC sample, despite achieving
  **lower uncheatable-eval val loss**, the LLM-extracted data gets **lower DCLM CoreV2 scores at all
  scales** ("SAME findings" independently confirmed twice). Diagnosis: **uncheatable val loss is a
  poor proxy** (skewed toward code; loses on QA/Math/research vs a proprietary CC mix). *"Why missed
  at 3k?"* — the checkpoint was picked at optimal uncheatable-eval loss, but DCLM/Nemotron-CC hit
  their optimum at much lower compute at the 3k budget, flattering the LLM method.
- Per the [2026-07-06_2026-07-12 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html):
  at compute-matched DCLM CORE, Marin's `high_quality` curation is **not Pareto-dominant even at the
  3k scale** — the apparent win relied on scale and on scoring only at each run's minimum
  uncheatable-eval val-loss point. Gap analysis: Marin strong on code, on par at math/Wikipedia, real
  gaps in fiction, QA benchmarks, news. To attack it, Michael Ryan **hand-labelled a 935-document
  dev set (keep / weak-keep / weak-drop / drop, plus must-keep + tricky edge cases)**, drawn by
  cross-pipeline disagreement and stratified by register. *(This 935-doc set is recorded in the
  weekly summary, not the #2351 thread text.)*

**Cross-cutting caution:** #6757's positive claims all rest on macro uncheatable BPB; #2351 is direct
evidence that uncheatable/val loss can move opposite to downstream benchmarks at scale — attach that
proxy caveat to every aggregate-loss "win."

### 1e. Cooldown / annealing mixes (vintage, 2025 — archived)

- **[#934](https://github.com/marin-community/marin/issues/934)** (opened 2025-04-02, **ARCHIVED
  2025-11-14**, PR #1999, tag `archive/20251114`): controlled cooldown-mix ablation over four mixes —
  (a) original DCLM+StarCoder, (b) Nemotron only, (c) Nemotron+Code+Dolmino, (d) +Marin's other data
  — scored on **Paloma loss, a Tulu3 instruction-data NLL, and MMLU accuracy**. Result: *"definite
  improvements across the board from dolmino,"* + a small persistent gain from Marin's own data.
- **[#820](https://github.com/marin-community/marin/issues/820)** (opened 2025-02-19): evaluated HQ
  cooldown datasets (Finemath, Starcoder, Wikipedia, pes2o, StackExchange). **Load-bearing finding:**
  HQ data beat the DCLM control on the **Paloma macro average, but that improvement did NOT show up
  in downstream benchmarks** — and the team concluded much of the Dolmino cooldown gain was from FLAN
  instruction/MCQ-format data (teaching MCQ structure), not "reinforcing high-quality knowledge."
  This is the same *"a val-loss win doesn't transfer"* lesson the 2026 #2351 reality check
  re-discovers.

> Note: **[#2404]** (Dolmino-3 Pool / 24 WebOrganizer-category RegMix replication, filed Jan 2026,
> auto-closed stale April 2026) is **superseded** by the 39-bucket Delphi work above; it is historical
> and should not be presented as the current many-domains structure.

---

## 2. Data-classifier / data-selection ablations

### 2a. Deployed fastText quality classifier v0 — the baseline ([#5810](https://github.com/marin-community/marin/issues/5810))

`experiments/datakit/cluster/quality/v0` (`sonnet46-thr05`), shipped across all 104 active datakit
sources. Oracle = **claude-sonnet-4-6, 1–5 pretraining-utility rubric**; **7,000 docs stratified
over 104 sources → 5,613 usable** after 1,387 API/parse failures; binary threshold 0.5 (raw ≥3).
fastText HPs `dim=100, epoch=5, lr=0.5, wordNgrams=2, minCount=3, bucket=200K`, vocab 67,934; cost
**$32.73** (<$50), ~4.5 h wall, CPU only.

**Achieved (held-out n=961 fresh oracle docs, seed 43):** **AUC 0.846, Spearman ρ 0.641**, acc/P/R/F1
@0.5 = 0.78/0.74/0.70/0.72, base rate 39%. Off-the-shelf dolma3-fasttext-quality reaches only ρ 0.168
on the same set (ours vs dolma3 ρ 0.21 — adding signal, not rediscovering it). This 5,613-train /
961-eval split is reused by every later classifier experiment.

### 2b. Fast-transformer vs fastText at fixed FLOP budget ([#6739](https://github.com/marin-community/marin/issues/6739))

rjpower, agent-driven; **target = a model at <1M FLOPs/token that beats fastText's AUC 0.846 /
ρ 0.641** on the same oracle splits. All runs on a single **v6e-4** (eu-west4), minutes/run.

- **Achieved deliverable:** `meanmaxmin · w64 · d512 · L4` pooled model — **AUC 0.875, ρ 0.703,
  acc 0.788, F1 0.751 at 0.41M FLOPs/token, 32M params** (acc/F1 at a val-calibrated 0.329 threshold).
  Beats fastText on all four metrics, under budget.
- **Attention is nearly free to drop:** `L=0` (learned embeddings + mean/max/min pooling + head) hits
  ρ 0.699 / AUC 0.868 at **12K FLOPs/token (34× cheaper)**; even `neural-bow` beats fastText
  (0.675 vs 0.641). The win over fastText is learned embeddings + spread-aware pooling, not the
  transformer.
- **The plateau and its diagnosed root cause (the actual answer to "why do classifiers plateau"):**
  the pooled model sits at AUC ≈0.87 / ρ ≈0.70 and has not moved across capacity/context/NTP/
  weak-supervised sweeps. Diagnosis = a **label-quality ceiling** of the 5.6K-doc oracle set,
  established by three independent negative distillations — every *free* teacher lands **below**
  training from scratch (0.869):

  | free teacher | pretrain-only → oracle ρ | pretrain+finetune vs from-scratch |
  |---|---|---|
  | source-of-origin prior | 0.499 (AUC 0.775) | 0.858 < 0.869 |
  | Nemotron-CC quality buckets (60K-doc slice) | 0.281 (AUC 0.667) | 0.814 (ρ 0.599) < 0.870 |
  | FineWeb-Edu score (~200k) | 0.282 (AUC 0.678) | 0.833 (ρ 0.627) < 0.861 |

  *"You generally can't distill your way above a teacher weaker than your student"* — so ~0.87 is a
  label-quality ceiling, not a modeling limit; the only lever above it is paid Sonnet relabeling
  (~$20–80 for 5–20k docs, held pending sign-off). **Source-of-origin alone predicts the oracle at
  AUC 0.852, η²=0.41 — reading no text** (already beating fastText). The best number ever seen
  (0.877/0.715) is **discarded**: 14.7M FLOPs/tok (14× over budget) and non-reproducing.
  Second diagnosis: oracle-AUC is edu-leaning (arxiv 0.85; pubmed/peS2o/math-textbooks 0.69–0.77),
  so pushing past 0.875 "most likely tracks MMLU/ARC and is neutral-to-negative on PIQA/HellaSwag" —
  the honest test is a downstream small-model data ablation, explicitly **not yet run**. Status: a
  *recommendation pending sign-off* (PR #6741), not a deployment; the store still runs the 0.846
  fastText during the audit below.

### 2c. Datakit v0 audit — "sorts by domain, not quality" ([#6849](https://github.com/marin-community/marin/issues/6849), [#6860](https://github.com/marin-community/marin/issues/6860))

Root-cause audit of the *deployed* fastText v0 on `store_8ac06c74` (ravwojdyla-agent, 2026-07-02),
part of a larger datakit-audit cluster (**#6849–#6860 + #7124/#7126**, spanning quality, luxical
embeddings, fuzzy dedup, decontam recall, over-sharding, truncated-4KB scoring):

- **[#6849] sorts by domain/modality/language:** clean code (`starcoder2/ir_python`) scores ~0.00
  (88–99% exactly 0); pristine `cp/arxiv_abstracts` math lands q0; `multilingual` raw-mean 0.18 vs
  type-aware oracle 0.51; store bucket q4 is code+math only. No absolute anchor — within `cp/wikiteam`,
  q4 is still fan-wiki parody cast-lists. Root cause = the same edu label definition (source predicts
  the oracle at AUC 0.852), so any faithful distillation reproduces the bias — raising oracle-AUC does
  not fix coherence.
- **[#6860] near-constant within-source scores:** **12 of 98 sources have score stddev <0.1**;
  `massive_function_calling` = exactly 0.0 (1 distinct value); synthetic/math saturated high
  (`nemotron_specialized/math_textbooks` 0.968), non-English saturated low (`finepdfs/jpn_Jpan` 0.044).
- **Validated fix:** a content-type-aware, source-blind rubric + pooled ranker (branch
  `rav/quality-coherence`) reaches **Spearman vs coherent-quality 0.44 → 0.69**, beating the deployed
  model on every content type. *(Note the metric switch — 0.44/0.69 are vs a new coherent-quality
  target, not the vs-oracle ρ of §2a–2b.)*

### 2d. Vintage classifier seed experiments (2024–2025, largely superseded)

Marked as historical context, not current findings:

- **[#163 fastText vs BERT](https://github.com/marin-community/marin/issues/163)** (opened 2024-08-26):
  both trained on **100k MMLU positives + 100k DCLM-pool negatives**, filtered at top-{1,5,10,20}%.
  A small blind human study preferred BERT and preferred top-1% to top-20%. On 1.4B-model training,
  **BERT top-20% won (≈0.015 gain over fastText on Paloma bits/byte)**, with a very large ~0.1
  bits/byte gain on the Dolma 100-programming-languages subset despite MMLU-only positives. Builds on
  the MMLU-classifier seed [#274](https://github.com/marin-community/marin/issues/274), the BERT
  pipeline [#100](https://github.com/marin-community/marin/issues/100) / [PR #185](https://github.com/marin-community/marin/pull/185),
  and HP tuning [#399](https://github.com/marin-community/marin/issues/399).
- Per-source / Dolmino classifiers: DCLM fastText on FineWeb
  [#63](https://github.com/marin-community/marin/issues/63), FineWeb-Edu Llama-3-annotation fastText
  [#390](https://github.com/marin-community/marin/issues/390), StackExchange
  [#596](https://github.com/marin-community/marin/issues/596), cascading filters
  [#963](https://github.com/marin-community/marin/issues/963), and a per-source Dolmino-mix classifier
  [#605](https://github.com/marin-community/marin/issues/605).
- **Embedding filtering (Luxical)** [#3535](https://github.com/marin-community/marin/issues/3535)
  (2026-03-11, under [#3049](https://github.com/marin-community/marin/issues/3049)): Luxical-One +
  RidgeCV probe reached **Spearman ≈0.75 after scaling to 10,000 docs** (below the 0.8 go-threshold);
  topic clustering a no-go (best NMI 0.478). Promising, not production-ready.

### 2e. New data-selection features under evaluation

- **Web-graph centrality [#6750](https://github.com/marin-community/marin/issues/6750)** (ravwojdyla-agent,
  2026-06-29): scope PageRank/betweenness/Katz over the crawl hyperlink graph (Common Crawl publishes
  harmonic centrality per host/domain) as a data-selection signal — decision recorded as
  adopt/reject/needs-experiment; not yet ablated.
- **Medical midtraining corpus [#7128] / SFT pilot [#7127]** (summary-sourced): Jeff Hammerbacher's
  proposal for a ~40–60B-token medical midtraining corpus (GAP-Replay analog from Common Pile PMC/
  pubmed, biomedical peS2o, EPFL clinical guidelines), grounded in the Delphi finding that
  midtraining supplies capability while SFT elicits it
  ([summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).

---

## 3. Test-data integrity (gates every mixture number)

### 3a. Nemotron-CC-Math contamination ([#6742](https://github.com/marin-community/marin/issues/6742))

ahmeda14960, branch `deconamint`. Reproducible Datakit pass over `nemotron_cc_math_v1/4plus`
(**45,096,087 docs**, us-east5), exact 13-word n-gram containment (ngram=13, overlap≥0.5,
FP-rate 1e-9 — **not** MinHash/LSH). **Eval-side coverage (achieved):**

| split | eval records | hit | % contaminated |
|---|---|---|---|
| MATH-500 test | 500 | 141 | **28.20%** |
| GSM8K train | 7,473 | 146 | 1.95% |
| GSM8K test | 1,319 | 32 | 2.43% |
| AIME24 | 30 | 17 | **56.67%** |

So MATH-500 and AIME24 are seriously contaminated; GSM8K is essentially clean. **Effect on the Delphi
1e22 math endpoint** (p33m67 K=0.20 lr0.50): cleaning the validation split cut the endpoint error from
**+18.56% (old full 4plus) → +2.83% (retained clean)**, while the dropped-contaminated complement kept
+15.01%. Base step-0 math loss was smooth (held-out 1e22 +2.4%), so the miss was eval-side, not a real
scaling break. Two confounds: fuzzy near-duplicate leakage (0 exact dupes, but 9,757/57,243 val docs
had a train near-twin at Jaccard≥0.75; on one high-J doc median token loss collapsed 1.99→0.003 nats at
1e22) and a token-budget artifact (K=0.20 is iso-FLOP, not fixed-token, so bigger models see more
midtraining tokens and more contamination). **Recommendation: treat old 4plus as a contaminated
historical baseline; use actual-seen-clean + fixed-token iso-token ladders.**

### 3b. BPB targets predict post-training capability ([#6096](https://github.com/marin-community/marin/issues/6096))

Helw150; the diagnostic rationale for why BPB targets are used at all. On 10 SimpleRL-Zoo base models
(scored on OpenMathReasoning cot/tir traces via Levanter on v5p-8), **base cot BPB → post-SimpleRL math
accuracy r = −0.89** (tir −0.92; within the 7–8B cohort **−0.97**). Distribution-specific: real math
traces (−0.89) ≫ templated synthetic reasoning (−0.68…−0.72); adding an embedding effective-rank term
lifts the joint fit to R²=0.931. **Critical caveat:** it predicts capability *level* (which RL
preserves), **not** the RL *gain* — predicting post−base fails (R²=0.33); n=10, non-independent, so
p-values are suggestive not established. This is the mixture-optimization arm's downstream anchor under
epic #6712.

### 3c. Decontamination-tool integrity ([#6852](https://github.com/marin-community/marin/issues/6852))

The production datakit decon is precision-favoring / low-recall: it flags ~0.01% overall (0% on
arXiv/news) against a comprehensive 600-benchmark-family bloom index, but brittle matching means
**short-line contamination escapes** (re-wrapping to 8-word lines drops overlap 0.87→0.00) and
**embedded contamination dilutes below threshold** (0.87→0.02). So a ~0% flag rate is *not* proof a
mix is clean — the common contamination case is invisible, and "decontaminated" mixes may carry
undetected MMLU/ARC/HellaSwag leakage.

---

## 4. Summary — training-data sizes and test data used

**Training-data sizes / compute budgets**
- **Delphi scaling ladder** (mixture validation): **3e18, 2e19, 3e20, 1e21 FLOPs** (v5p-8/16/32/64) —
  [#6607](https://github.com/marin-community/marin/issues/6607)/[#6608](https://github.com/marin-community/marin/issues/6608)/[#6609](https://github.com/marin-community/marin/issues/6609)/[#6611](https://github.com/marin-community/marin/issues/6611).
- **Iso-FLOP MoE scales** (datakit side-by-side): d512 (3.82e17) → d1280 (3.46e19) —
  [#6757](https://github.com/marin-community/marin/issues/6757).
- **Swarm sizes** for mixture fitting: **60M/1.2B and 300M/6B**; the noise reference for #6969 is
  300M-repeat spread.
- **WARC-budget scaling** {100…3000} + 10k-WARC sample; 400M model on ~8.2B tokens; 3e18→2e21 FLOPs —
  [#2351](https://github.com/marin-community/marin/issues/2351).
- **Classifier training sets:** 5,613-doc oracle train / 961 holdout ([#5810](https://github.com/marin-community/marin/issues/5810),
  [#6739](https://github.com/marin-community/marin/issues/6739)); 100k+100k label sets, top-{1,5,10,20}%
  ([#163](https://github.com/marin-community/marin/issues/163)); Luxical to 10k docs
  ([#3535](https://github.com/marin-community/marin/issues/3535)).
- **GPU feature-ablation chains:** two d512 compute-optimal chains, CoreWeave 8×H100, datakit data,
  under the 11B-A1.5B validation run [#6716](https://github.com/marin-community/marin/issues/6716) —
  Paloma macro loss walked down **3.7103 (pure main) → 3.6954 (+VMAP w_gate fix) → 3.6359 (+256
  experts)**; the residual gap to the TPU nemotron-mixture reference traces to *data, not hardware*,
  and re-scoring on Uncheatable-eval flips the ranking (datakit ahead)
  ([summary 2026-07-06_2026-07-12](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html);
  #6716 itself is a 3-line stub, so these numbers are summary-sourced). The **10T June-run datamix
  [#6045](https://github.com/marin-community/marin/issues/6045)** closed completed 2026-07-07.

**Test data / evals**
- **BPB / perplexity (optimization targets):** uncheatable-eval BPB; Paloma macro loss; **OLMoBaseEval
  Easy Table-9 macro (51 components)**; DCLM CORE / COREv2; Tulu3 NLL.
- **Downstream logprob (sanity gate):** the ~14-eval swarm set (arc, hellaswag, winogrande, piqa,
  mmlu, gsm8k, humaneval, …) + the 137-task mega set in #6757.
- **Classifier metrics:** held-out AUC / Spearman vs Sonnet-4.6 oracle ([#6739](https://github.com/marin-community/marin/issues/6739),
  [#5810](https://github.com/marin-community/marin/issues/5810)); blind human preference
  ([#163](https://github.com/marin-community/marin/issues/163)); Spearman vs a coherent-quality target
  ([#6849](https://github.com/marin-community/marin/issues/6849)).
- **Known bias / the recurring lesson:** aggregate BPB (esp. uncheatable, which tilts toward code/math)
  can move *opposite* to natural-language commonsense benchmarks — #6757's commonsense regression and
  #2351's / #820's val-loss-doesn't-transfer reversals are the same failure mode. The recommended
  operating combination is "paloma+uncheatable+mmlu train together," not any single set.

### Data-source A/B: focus vs main crawl ([#6570])

[#6570](https://github.com/marin-community/marin/issues/6570) (Helw150, opened 2026-06-23) token-matched **science-focused CC-SUPPLEMENTAL-2026-22 vs general CC-MAIN-2024-18** on an 8-run v5p-8 ladder (d512–d1280, scored on eval/paloma/macro_loss). At d512 the **general "main" crawl won: focus 4.712 vs main 4.402 (Δ −0.310)**; the science-vs-general contrast replicated at d768 (4/8 rungs at freeze, in-flight).

---
<!--provenance-->
> *Corpus: frozen 2026-07-16 Marin eval corpus (marinmirror + weekly summaries through
> 2026-07-06_2026-07-12; refresh disabled). Every GitHub thread cited was re-verified with
> `mum-frozen show`; every load-bearing number was checked against its primary thread, except where
> explicitly marked summary-sourced (primary thread post-dates the freeze: #6969, #6972, #7067, #7127,
> #7128, and the #6716 feature-chain Paloma numbers) or W&B-only (the achieved one-phase BPB ladders,
> which are training scalars absent from the frozen text corpus). #2404 is flagged superseded; #934
> and #820 are dated to their 2025 vintage.*
