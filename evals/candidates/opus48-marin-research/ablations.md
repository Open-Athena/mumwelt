# Data ablations run so far: mixes, classifiers, sizes, and test sets

This pulls together the data ablations recorded in the Marin corpus. They fall into two
overlapping tracks: **quality-classifier ablations** (which document filter selects the best
data) and **data-mixture ablations** (which blend of sources, and in what proportions, trains
the best model). Both have been evaluated at a range of compute/token budgets against a fairly
stable set of held-out test corpora. Where a number is a *target* rather than a measured result,
or is sourced only from the weekly narrative rather than a primary thread, that is flagged.

## 1. Quality-classifier ablations (which filter picks the best data)

**fastText vs BERT** — the anchor classifier comparison ([#163](https://github.com/marin-community/marin/issues/163)).
Both a BERT and a fastText classifier were trained on the same labels — **100k MMLU examples as
the positive set, 100k DCLM-pool documents as the low-quality set** — then used to filter at
several thresholds (keep top **{1, 5, 10, 20}%**). A small blind human study preferred BERT over
fastText, and preferred top-1% to top-20%; the note is that this qualitative gap was "to some
extent predictive of downstream performance," while citing prior work (DCLM appendix) that
human-classifier agreement is *not* generally predictive
([#163 comment, RohithKuditipudi](https://github.com/marin-community/marin/issues/163#issuecomment-2889102723)).
This built on the "train an MMLU quality classifier" seed experiment
([#274](https://github.com/marin-community/marin/issues/274)) and the BERT training pipeline
([#100](https://github.com/marin-community/marin/issues/100), [PR #185](https://github.com/marin-community/marin/pull/185)),
with hyperparameter tuning tracked in [#399](https://github.com/marin-community/marin/issues/399).

**Source-specific / Dolmino classifiers.** Beyond the single MMLU filter, per-source high-quality
classifiers were built and compared: a DCLM fastText classifier run on FineWeb
([#63](https://github.com/marin-community/marin/issues/63)), a fastText classifier trained on
FineWeb-Edu Llama-3 annotations ([#390](https://github.com/marin-community/marin/issues/390)), a
StackExchange quality classifier ([#596](https://github.com/marin-community/marin/issues/596)),
cascading quality filters ([#963](https://github.com/marin-community/marin/issues/963)), and a
**Dolmino-mix classifier per source** (StackExchange, wiki, pes2o, synthetic math, DCLM web) with
the hypothesis that Q&A/conversational data would help most and an ensemble would be preferred
([#605](https://github.com/marin-community/marin/issues/605)).

**Embedding-based filtering (Luxical).** Cached text embeddings were tested as a cheaper quality
filter and topic-clusterer ([#3535](https://github.com/marin-community/marin/issues/3535), part of
[#3049](https://github.com/marin-community/marin/issues/3049); [PR #3191](https://github.com/marin-community/marin/pull/3191)).
Best result: **Luxical-One + a linear RidgeCV probe reached ≈ Spearman 0.75 for quality scoring
after scaling to 10,000 documents**, still below the issue's 0.8 go-threshold; topic clustering was
a no-go (best NMI 0.478). Recommendation: promising but not production-ready.

**Fast-transformer vs fastText at a fixed FLOP budget** ([#6739](https://github.com/marin-community/marin/issues/6739)).
A pooled fast-transformer classifier was compared against the deployed fastText classifier under a
**< 1M FLOPs/token** budget, using the **same Sonnet 4.6 oracle-labeled train/eval splits**. Best
under-budget model (mean/max/min-pooled, window 64, d512, L4) reached **AUC 0.875** vs the fastText
baseline.

**datakit classifier diagnostics (July 2026) — a negative result.** Two bug reports found the
deployed datakit fastText quality classifier (`experiments/datakit/cluster/quality/v0`) does not
measure intrinsic quality: it **sorts documents by domain/modality/language, not quality**
([#6849](https://github.com/marin-community/marin/issues/6849)) — clean code and dense math land in
the junk bucket, non-English is uniformly bottom-bucketed, and the top bucket is code+math only.
Root cause: the Sonnet-rubric label is ~recoverable from source identity alone (**source predicts
the oracle at AUC 0.852**), so any faithful distillation inherits the domain bias. Separately, for
many sources it gives **near-constant per-source scores** (12 of 98 sources have score stddev < 0.1;
`massive_function_calling` is exactly 0.0 for every document), so it can't rank within a source
([#6860](https://github.com/marin-community/marin/issues/6860)). A source-blind, content-type-aware
rubric fix is reported to raise Spearman-vs-coherent-quality from 0.44 → 0.69.

## 2. Data-mixture ablations (which blend, in what proportions)

**Cooldown / annealing mixes.** [#820](https://github.com/marin-community/marin/issues/820)
evaluated candidate high-quality cooldown datasets (Finemath, Starcoder, Wikipedia, pes2o,
StackExchange) for the cooldown mix. [#934](https://github.com/marin-community/marin/issues/934)
(PR [#936](https://github.com/marin-community/marin/pull/936)) ran a controlled cooldown-mix
ablation comparing four mixes — **(a) original DCLM + StarCoder, (b) Nemotron only, (c) Nemotron +
Code + Dolmino, (d) Nemotron + Code + Dolmino + Marin's other data** — scored on **Paloma loss, a
Tulu3 instruction-data NLL, and MMLU accuracy**. Result: "definite improvements across the board
from dolmino," plus a small persistent gain from adding Marin's own data. A related cooldown/"midtrain"
ablation was run off a **Nemotron 1.3B** checkpoint lineage
([#2351](https://github.com/marin-community/marin/issues/2351), XenonMolecule, 2026-03-05).

**Many-domains / RegMix-style optimization.** [#2404](https://github.com/marin-community/marin/issues/2404)
replicated the Olmo-3 / RegMix "many domains, one phase" result on the **Dolmino-3 Pool** (optimizing
weights over the 24 WebOrganizer categories), with the bar being to beat the proportional baseline
after regression.

**DSP / OLMix mixture optimization scaled on the Delphi ladder (June–July 2026).** The most systematic
recent mixture work optimizes a mixture on a small (300M-param) swarm, then scales it on the **Delphi
scaling ladder — budgets 3e18, 2e19, 3e20, 1e21 FLOPs** — and reports how the optimized mixture holds
up. Threads: optimize-for-Uncheatable-Eval ([#6602](https://github.com/marin-community/marin/issues/6602)),
optimize-for-OLMoBaseEval-Easy ([#6603](https://github.com/marin-community/marin/issues/6603)), the
OLMix baseline scaling run ([#6608](https://github.com/marin-community/marin/issues/6608)), and the
OLMoBaseEval-Easy swarm eval ([#6611](https://github.com/marin-community/marin/issues/6611)). Concrete
setup: **3 mixtures × 4 Delphi scales = 12 runs**, a fixed **80/20 two-phase schedule**, comparing
OLMix (Huber-delta / KL / repetition-cap variants, e.g. `olmix_d001_kl005_cap4`) against
Effective-exposure DSP and canonical DSP
([#6608 launcher-invariant comment](https://github.com/marin-community/marin/issues/6608#issuecomment-4803008438)).
A key finding is that the **best KL is target-specific** and that Effective-exposure DSP fit directly
on the OLMoBaseEval-Easy **51-component Table-9 macro BPB** (best `linear_reg=1e-4`, OOF Spearman
0.918, OOF RMSE 0.0128) beats the paper-faithful OLMix reference on the same target
([#6611 comment](https://github.com/marin-community/marin/issues/6611#issuecomment-4804956805),
[#6602 cross-ref](https://github.com/marin-community/marin/issues/6602#issuecomment-4804956790)).
The production **10T mix for the June hero run** was produced separately
([#6045](https://github.com/marin-community/marin/issues/6045)).

**datakit vs nemotron feature-ablation chain (GPU validation, July 2026).** On CoreWeave 8×H100, two
d512 compute-optimal chains checked that the GPU stack tracks the TPU feature-ablation chain; final
**Paloma macro loss walked down cleanly per stack change: pure main 3.7103 → +VMAP w_gate fix 3.6954
→ +256 experts 3.6359**. Most of the residual gap to the TPU reference traced to *data, not hardware*:
the TPU ablations run on the **nemotron mixture**, which lands lower on Paloma at this d512 point,
whereas **datakit** is tuned toward code/math. Re-scoring the same runs on the shared **Uncheatable-eval**
cache *flipped the ranking* (datakit ahead), i.e. a natural-language-vs-code trade-off that sharpens
near the compute-optimal frontier. *(Source: weekly summary
[2026-07-06_2026-07-12](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html); the referenced
primary issue #6716 is only a stub in this frozen corpus, so treat these exact loss numbers as
summary-sourced rather than verified against the thread.)*

**Curation Pareto reality check.** A sobering result on curation was reported: at **compute-matched
DCLM CORE eval**, Marin's `high_quality` curation is **not Pareto-dominant even at the 3k scale** — the
earlier apparent win relied on scale and on scoring only at each run's minimum val-loss point. A gap
analysis put Marin strong on code, on par at math/Wikipedia, but with real gaps in fiction, QA
benchmarks, and news; to attack it a **935-document dev set** was hand-labeled (keep / weak-keep /
weak-drop / drop). This lives in the web-data-to-tokens / DCLM-CORE track
([#2351](https://github.com/marin-community/marin/issues/2351), weekly-sync updates 6/29–7/3;
corroborated by the [2026-07-06_2026-07-12 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).
The underlying DCLM-CORE competition scaling suite compares curations such as `dclm_random` vs
`high_quality` across compute budgets (e.g. wandb `marin-dclm-core` runs
`curation-dclm_random_3000-…-3e18-d1536` and `curation-high_quality_3000-…-2e21-d1536`).

## 3. What training-data sizes / compute budgets were used

- **Delphi scaling ladder**, the standard scaling harness for mixture validation: **3e18, 2e19, 3e20,
  1e21 FLOPs** ([#6608](https://github.com/marin-community/marin/issues/6608)).
- **Swarm sizes** for mixture optimization: **60M and 300M** parameter models
  ([#6611](https://github.com/marin-community/marin/issues/6611), [#6602](https://github.com/marin-community/marin/issues/6602)).
- **Compute-optimal proxy models** for feature/curation chains: **d512, d768, d1536** (L8/L16), e.g.
  the GPU feature-ablation chain at d512 and the DCLM-CORE curation runs at d1536
  (summary [2026-07-06_2026-07-12](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html); wandb `marin-dclm-core`).
- **DCLM CORE data pool**: the full **DCLM 400M-1x** pool, plus fixed 10k-WARC scaling sweeps
  ([#2351](https://github.com/marin-community/marin/issues/2351)).
- **Cooldown ablations**: run off a **Nemotron 1.3B** checkpoint lineage
  ([#2351](https://github.com/marin-community/marin/issues/2351)) and the annealing setup
  ([#934](https://github.com/marin-community/marin/issues/934)).
- **Classifier ablations**: trained on ~100k-document positive/negative label sets and filtered at
  top-{1,5,10,20}% ([#163](https://github.com/marin-community/marin/issues/163)); Luxical scaled to
  10k labeled documents ([#3535](https://github.com/marin-community/marin/issues/3535)).
- **Production mix**: the 10T-token June hero-run datamix ([#6045](https://github.com/marin-community/marin/issues/6045)).

## 4. What test data / evals were used

- **Paloma macro loss** — the primary pretraining-loss yardstick (cooldown ablation
  [#934](https://github.com/marin-community/marin/issues/934); feature-ablation chain).
- **DCLM CORE / COREv2** — compute-matched curation comparison and the competition scaling suite
  ([#2351](https://github.com/marin-community/marin/issues/2351)).
- **Uncheatable Eval (BPB)** — mixture-optimization target that flips the datakit-vs-nemotron ranking
  ([#6602](https://github.com/marin-community/marin/issues/6602), [#6608](https://github.com/marin-community/marin/issues/6608)).
- **OLMoBaseEval Easy — the 51-component Table-9 macro BPB** (and top-3 QA/code/math aggregates)
  ([#6611](https://github.com/marin-community/marin/issues/6611), [#6603](https://github.com/marin-community/marin/issues/6603)).
- **MMLU accuracy** and a **Tulu3 instruction-data NLL** (cooldown ablation
  [#934](https://github.com/marin-community/marin/issues/934)).
- **Task benchmarks** in the extraction/curation track: **HumanEval pass@1, GSM8K, MATH**
  ([#2351](https://github.com/marin-community/marin/issues/2351)).
- **Classifier-level metrics**: held-out **AUC / Spearman** against Sonnet-4.6 oracle labels
  ([#6739](https://github.com/marin-community/marin/issues/6739), [#6849](https://github.com/marin-community/marin/issues/6849)),
  and blind **human preference** ([#163](https://github.com/marin-community/marin/issues/163)).

## Gaps / caveats

- The **datakit-vs-nemotron feature-chain Paloma numbers** (3.7103 / 3.6954 / 3.6359) come from the
  weekly summary; the primary issue is a stub in this frozen corpus, so they are not verified against
  the thread.
- Two recent mixture threads referenced in the July summary — the **embedding-based mixture surrogate**
  (`mixing_via_embeddings`, #6969) and the **OLMix-KL scale-up** (#6972) — are **not present in this
  frozen corpus**, so their specifics can't be cited beyond the summary narrative.
- Several classifier threads record the *setup* but link results out to WandB reports / the data
  browser (e.g. [#163](https://github.com/marin-community/marin/issues/163),
  [#934](https://github.com/marin-community/marin/issues/934)), which aren't in the corpus, so
  head-to-head downstream deltas beyond the summaries above aren't fully quotable here.

---
<!--provenance-->
> *Data: marinmirror — 68026 chunks, built 1h ago · summaries through 2026-07-06_2026-07-12 (frozen 2026-07-16 eval corpus; refresh disabled).*
>
> *Query: "Can you share more details about the data ablations we've run so far? What data mixes / data classifiers were compared, on what training-data sizes and on what test data?"*
>
> *Sub-queries: "data mixture ablation datakit vs nemotron Paloma" · "quality classifier comparison document filtering" · "DCLM CORE high_quality curation Pareto compute-matched" · "OLMix KL data mixing sweep 3e18–1e21" · "mixing via embeddings surrogate mixture" · "data ablation compute-optimal d512 feature chain" · "Uncheatable eval data mix ranking" · "fastText vs BERT quality classifiers" · "Dolmino/Olmo midtraining mixture" · "cooldown pretraining vs HQ data" · "datakit quality classifier sorts by domain" · "OLMoBaseEval Table-9 macro scaling" · "Luxical embeddings quality filtering"*
