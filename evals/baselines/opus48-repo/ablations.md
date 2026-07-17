# Marin data ablations: what we've run so far

This summarizes the data-focused ablations run in the `marin-community/marin` repo,
organized by (a) the ablation *methodology* we use, (b) **data-classifier** comparisons,
and (c) **data-mix** comparisons — noting training-data sizes and the test data used in each.
All references are live GitHub issues/PRs.

---

## 1. Methodology: how we run data ablations

- **[RFC] Dataset Evaluation Methodology ([#722](https://github.com/marin-community/marin/issues/722))** —
  establishes the standard way we ablate adding a dataset to the mix: pretrain a base model on a
  large-scale CC/DCLM run, take an intermediate checkpoint, then **anneal** on a mix of
  high-quality base data + the candidate domain data, and measure whether the target-domain
  benchmark improves without regressing others. The concrete Marin design: train a Llama-3 8B on
  **1T tokens of DCLM**, take the **80% (0.8T)** checkpoint, then anneal on a **70/30
  (high-quality DCLM / domain-specific)** mix for **50B tokens**. Literature basis cited: Llama-3
  domain upsampling (7.5T base then 40B anneal at 70/30), Mosaic "Does your data spark joy?" (1T
  budget, 0.8T CC + 0.2T mix), and OLMo-2 microanneals (50B for 7B, 100B/300B for 13B; and 200M
  math microanneals). This anneal/microanneal recipe is the backbone of most subsequent data
  ablations.

- **Scaling laws for quality classifiers ([#669](https://github.com/marin-community/marin/issues/669))** —
  asks whether the *relative* ranking of two classifiers flips as you vary the **keep fraction**
  (number of filtered documents trained on). Motivation for varying training-data size in classifier
  ablations.

- **Simulated Data Constraints for SLs and Data Ablations ([#718](https://github.com/marin-community/marin/issues/718))** —
  simulated epoching / data-constraint handling so ablations can model repeated data.

---

## 2. Data-classifier ablations (what classifiers were compared)

Marin's quality filtering is fastText-based (DCLM-style), and there is a long line of ablations
comparing classifier *training data*, *model architecture*, and *ensembling*, mostly evaluated on
**bpb** (bits-per-byte on held-out sets like MMLU/bpb, eval/bpb, c4_en/bpb, Paloma/bpb) and on
**MMLU** downstream accuracy.

- **Replicate DCLM OH-2.5 + ELI5 fastText classifier ([#102](https://github.com/marin-community/marin/issues/102))** —
  reproduce the DCLM paper's OH-2.5 + ELI5 fastText quality classifier, compare our trained
  classifier vs the released DCLM one, then filter data and train **DCLM-style 1B models** to
  measure downstream performance.

- **fastText vs BERT for quality classifiers ([#163](https://github.com/marin-community/marin/issues/163))** —
  apples-to-apples comparison (same positive/negative examples, same token budget, same fine-tuning)
  of a fastText classifier vs a stronger BERT classifier for data selection.

- **Train MMLU quality classifier ([#274](https://github.com/marin-community/marin/issues/274))** —
  train a classifier to distinguish MMLU-like text vs not, and measure the correlation between the
  classifier's MMLU-discrimination ability and the resulting LM's MMLU score.

- **StackExchange quality classifier ([#596](https://github.com/marin-community/marin/issues/596))** —
  compares a classifier trained on **StackExchange** (task-specific conversational) vs the baseline
  **ELI5 + OH-2.5** classifier. Result: best **c4_en/bpb** and **eval/bpb**, and higher **mmlu/bpb**
  than the dedicated MMLU classifier.

- **Dolmino mix quality classifier ([#605](https://github.com/marin-community/marin/issues/605))** —
  build a separate classifier for each high-quality Dolmino source (StackExchange, wiki, pes2o,
  DCLM web, synthetic/web math) and compare individually vs as an ensemble. Result: **pes2o and
  wiki** classifiers gave very strong **MMLU/bpb** and **eval/bpb**, slightly weaker on **Paloma/bpb**
  than ELI5-100k.

- **Ensemble quality classifiers ([#615](https://github.com/marin-community/marin/issues/615))** —
  compares 5 ways of combining classifiers from different sources (max/min score, union training,
  weight averaging for BERT, score averaging, iterative multi-stage pruning).

- **Related earlier classifier experiments:** train classifiers on WildChat
  ([#400](https://github.com/marin-community/marin/issues/400)), on reasoning traces
  ([#616](https://github.com/marin-community/marin/issues/616)), and on explicit low-quality data
  ([#618](https://github.com/marin-community/marin/issues/618)).

### Newer "datakit" classifier work (2026)

- **Fast-transformer document quality classifier: beat fastText at <1M FLOPs/token
  ([#6739](https://github.com/marin-community/marin/issues/6739), PR
  [#6741](https://github.com/marin-community/marin/pull/6741))** — the most rigorous recent
  classifier ablation. Same **Sonnet-4.6 oracle-labeled** train/eval splits as the deployed fastText
  baseline: **train n≈5,613** (`train-n7000-seed42-sonnet46`), **eval holdout n≈961**
  (`eval-n1000-seed43-sonnet46`), drawn from **104 datakit sources** (web, arxiv, caselaw, code, …);
  labels are rubric 1–5 → normalized {0,.25,.5,.75,1.0}. Metrics: **AUC, Spearman ρ, acc/P/R/F1 at
  0.5**. Result: best under-budget model (mean/max/min pooled fast-transformer, window 64, d512, L4)
  hit **AUC 0.875 / Spearman 0.703 at 0.41M FLOPs/token**, beating trained fastText (**AUC 0.846 /
  Spearman 0.641**) and off-the-shelf dolma3 (Spearman 0.168). Most of the gain came from learned
  embeddings + multi-statistic pooling, not attention.

- **LLM-oracle fastText quality classifier ([#5810](https://github.com/marin-community/marin/issues/5810))** —
  distilling a Claude/Sonnet oracle rubric into fastText (the pipeline that produced the labels
  reused by #6739).

- **datakit quality-classifier bugs / limitations** — an important negative-result cluster showing
  the deployed classifier sorts by **domain, not intrinsic quality**
  ([#6849](https://github.com/marin-community/marin/issues/6849)): source alone predicts the oracle
  at **AUC 0.852**, so any faithful distillation reproduces domain bias (clean code/dense math land
  in the junk bucket, non-English uniformly bottom-bucketed). Companion findings: near-constant
  scores with no within-source discrimination
  ([#6860](https://github.com/marin-community/marin/issues/6860)) and quality judged only on a
  truncated 4KB document lead ([#6859](https://github.com/marin-community/marin/issues/6859)). A
  content-type-aware, source-blind rubric + pooled ranker improved Spearman vs coherent quality
  **0.44→0.69**.

- **Classifier throughput characterization** — fastText
  ([#5406](https://github.com/marin-community/marin/issues/5406)) vs the Luxical-One
  quality-embedding ([#5410](https://github.com/marin-community/marin/issues/5410)) on normalized
  sources; and **fast-transformer perf on TPUs**
  ([#7187](https://github.com/marin-community/marin/issues/7187)).

- **Dolma3 quality scores for all sources
  ([#5812](https://github.com/marin-community/marin/issues/5812))** and **WebOrganizer domain
  buckets ([#5811](https://github.com/marin-community/marin/issues/5811))** — apply/compare scoring
  across the full source catalog.

---

## 3. Data-mix ablations (what mixes were compared, sizes, test data)

### Swarm-based data-mixture optimization

- **Data Mixing Many Domains Swarm Run ([#2345](https://github.com/marin-community/marin/issues/2345))** —
  the foundational mixture-optimization swarm (RegMix / OLMo-3 spirit, extended to epoching and
  multiple training phases). Data source: **Dolma 3 Pool (9.31T tokens)** + **Dolma 3 Dolmino Pool
  (2.19T)**, partitioned into fine-grained subsets — CC into **24 topics × quality buckets** (≈290
  subsets), olmOCR science PDFs (24 topics), Stack-Edu (15 programming languages), arXiv, FineMath
  3+, Wikipedia — **332 total subsets**. Trains many small proxy models over different mixtures to
  fit a regression predicting downstream performance.

- **MDE features for 300M data mixing ([#6326](https://github.com/marin-community/marin/issues/6326))** —
  ablation of *features* used to predict good mixtures: trained **39 cap-1 single-domain 300M vertex
  experts** on the qsplit240 swarm and compared **MDE** (mixture-of-data-experts checkpoint
  likelihood) features vs **DSP**, phase-log exposure, and DSP-derived predictors. Negative result:
  MDE mostly reparameterizes mixture/exposure geometry; DSP remained stronger (e.g. OOF Spearman on
  `uncheatable_eval/bpb` DSP **0.914** vs MDE **0.861**; TruthfulQA choice-logprob MDE only 0.10).

- **June production data mixture ([#5359](https://github.com/marin-community/marin/issues/5359),
  epic)** — determine the pre- and mid-training mixture for the June overtrained **67B-A2B 10T**
  model. Sub-issues: **select the optimization metric** (high-SNR, de-risked on Calvin's swarm;
  [#5362](https://github.com/marin-community/marin/issues/5362)), **finalize the "buckets" of data
  to mix over** ([#5363](https://github.com/marin-community/marin/issues/5363)), build/launch the
  "Production" model swarm ([#5364](https://github.com/marin-community/marin/issues/5364),
  [#5365](https://github.com/marin-community/marin/issues/5365)), and deliver the final mix for
  forecasting ([#6391](https://github.com/marin-community/marin/issues/6391)). The bar (definition of
  done): the learned mixture must **beat, or at worst tie, (a) proportional mixing over all sources
  and (b) proportional mixing over hand-picked high-quality sources**, de-risked on UncheatableEval,
  HumanEval, MMLU, GPQA, and David's PPL sets. Upstream: quality-score + dedup param selection
  ([#5360](https://github.com/marin-community/marin/issues/5360)).

### Head-to-head mix comparisons

- **Curated vs proportional datakit mix — iso-FLOP side-by-side (MoE)
  ([#6757](https://github.com/marin-community/marin/issues/6757))** — the cleanest recent mix
  comparison. Three mixes from datakit store `store_8ac06c74`: **curated**, **proportional**
  (natural-token weights), and **nemotron** (baseline, simulated-epoching). MoE, **4 iso-FLOP
  scales**: d512 (3.82e17), d768 (2.81e18), d1024 (1.16e19), d1280 (3.46e19 FLOPs). Test surface:
  **perplexity-gap PPL / macro bpb** plus **lm-eval logprob** on mmlu, gsm8k, humaneval, arc
  easy/challenge, piqa, winogrande, hellaswag, bbh, mmlu_pro, and a **137-task mega set** (111
  runnable/cell). Verdict: **curated and proportional are statistically indistinguishable at this
  compute** — of 102 downstream tasks only 3 separate them (curated wins HumanEval; proportional
  wins two Indonesian-commonsense tasks), i.e. fewer than the ~10 false positives expected by chance.
  Both mixes beat nemotron on aggregate loss + code/math but both regress vs nemotron on commonsense.
  Weak default: **curated** (its one edge, code, is in-scope for an AA-II target).

- **MEDU Sciences crawled-data ablations ([#931](https://github.com/marin-community/marin/issues/931))** —
  anneal-based comparison of three arms: (1) anneal on **MEDU Sciences**, (2) anneal on the
  **crawled outlinks** dataset (8-step crawl/curate/filter pipeline), (3) anneal on **MEDU +
  crawled**. Hypothesis: adding HQ crawled data lifts MMLU.

### Selection-method and feature ablations (mix-adjacent)

- **Evaluate MATES / Group-MATES data selection
  ([#6521](https://github.com/marin-community/marin/issues/6521))** — evaluate learned per-example
  data-selection methods.
- **Web-graph centrality as a document feature for the next mix
  ([#6750](https://github.com/marin-community/marin/issues/6750))** and **metadata conditioning for
  data efficiency ([#5197](https://github.com/marin-community/marin/issues/5197))**.
- **[Vibe Mixing] pre-registered vibe mix ([#6063](https://github.com/marin-community/marin/issues/6063))**
  and **June MoE prep runs on new data mix
  ([#6449](https://github.com/marin-community/marin/issues/6449))**.

### Forward-looking / standing ablation loop

- **[Proposal] Recurring per-crawl data ablation
  ([#7162](https://github.com/marin-community/marin/issues/7162))** — proposes a standing
  Marin × Common Crawl feedback loop that re-runs data ablations on each new crawl.
- **Pretraining data curation & mix ([#6713](https://github.com/marin-community/marin/issues/6713))** —
  the umbrella tracking issue the curated-vs-proportional and classifier decisions feed into.

---

## 4. Quick reference: training sizes and test data used

| Ablation | Model / training size | Test data |
|---|---|---|
| Dataset-eval RFC [#722](https://github.com/marin-community/marin/issues/722) | Llama-3 8B, 1T DCLM base + 50B anneal (70/30) | domain benchmarks + retained-quality checks |
| Fast-transformer classifier [#6739](https://github.com/marin-community/marin/issues/6739) | classifier: 5.6K train / 961 eval docs (104 sources) | AUC, Spearman, F1 vs Sonnet-4.6 oracle |
| Classic classifier expts [#102](https://github.com/marin-community/marin/issues/102) [#596](https://github.com/marin-community/marin/issues/596) [#605](https://github.com/marin-community/marin/issues/605) | DCLM-style ~1B LMs on filtered data | MMLU/bpb, eval/bpb, c4_en/bpb, Paloma/bpb, MMLU acc |
| Many-domains swarm [#2345](https://github.com/marin-community/marin/issues/2345) | proxy swarm over Dolma3 (9.31T pool + 2.19T dolmino), 332 subsets | UncheatableEval, HumanEval, MMLU, GPQA, PPL sets |
| MDE features [#6326](https://github.com/marin-community/marin/issues/6326) | 39 × 300M single-domain vertex experts | uncheatable_eval/bpb, TruthfulQA (OOF Spearman) |
| Curated vs proportional [#6757](https://github.com/marin-community/marin/issues/6757) | MoE, 4 iso-FLOP scales d512–d1280 (3e17–3e19) | macro bpb PPL + 137-task lm-eval logprob |
| June production mix [#5359](https://github.com/marin-community/marin/issues/5359) | 67B-A2B 10T target; swarm de-risking | UncheatableEval, HumanEval, MMLU, GPQA, PPL |
| MEDU crawled [#931](https://github.com/marin-community/marin/issues/931) | anneal runs (MEDU / crawled / both) | MMLU |

---

### Headline takeaways

1. **Classifiers:** we have repeatedly ablated classifier *training data* (ELI5+OH-2.5 vs
   StackExchange vs Dolmino sources vs MMLU) and *architecture* (fastText vs BERT vs pooled
   fast-transformer). The fast-transformer beats fastText under a 1M-FLOP budget
   ([#6739](https://github.com/marin-community/marin/issues/6739)), but a key negative finding is
   that the deployed quality classifier sorts by **domain, not intrinsic quality**
   ([#6849](https://github.com/marin-community/marin/issues/6849)).
2. **Mixes:** at the scales tested so far (≤ d1280, 3e19 FLOPs), **curated vs proportional data
   mixes are statistically indistinguishable** on aggregate loss and nearly all downstream tasks
   ([#6757](https://github.com/marin-community/marin/issues/6757)); a confident call needs a
   larger-scale point. Mixture-optimization via swarms + regression is the main machinery
   ([#2345](https://github.com/marin-community/marin/issues/2345),
   [#5359](https://github.com/marin-community/marin/issues/5359)), and MDE features did **not** beat
   the existing DSP predictor ([#6326](https://github.com/marin-community/marin/issues/6326)).
3. **Test data** across ablations is consistently: **bpb / perplexity-gap PPL** (UncheatableEval,
   MMLU/bpb, eval/bpb, Paloma, c4_en) plus **lm-eval downstream** (MMLU, GSM8K, HumanEval, ARC,
   GPQA, BBH, MMLU-Pro, commonsense).
