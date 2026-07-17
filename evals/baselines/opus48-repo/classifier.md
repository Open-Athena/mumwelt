# The Marin data-classifier model (`sonnet46-thr05`)

The "data classifier" referenced in [#5810](https://github.com/marin-community/marin/issues/5810) is an **LLM-oracle fastText quality classifier**. It is a lightweight fastText model whose training labels come not from humans but from a Claude "oracle" scoring documents against a written pretraining-utility rubric. It was built as a child of the May data-pipeline milestone [#5360](https://github.com/marin-community/marin/issues/5360) ("quality scores + dedup param selection"), to provide a `attributes.high_score = P(high) ∈ [0,1]` quality signal that slots in alongside the existing `dolma3-fasttext-quality` classifier.

## What the model is

- **Architecture**: a plain **fastText** binary classifier (not a neural LM). Hyperparameters: `dim=100, epoch=5, lr=0.5, wordNgrams=2, minCount=3, bucket=200K`; final vocab = 67,934.
- **Output**: at inference it emits a continuous `P(high)` in `[0,1]`, drop-in compatible with Marin's existing `classify_fasttext_step` consumer shape.
- **Design choices (v0)**: a single rubric across all sources, one trained model, continuous score. Explicit non-goals were per-domain rubrics/classifiers, multi-class/regression heads, and threshold tuning beyond the rubric-aligned default ([#5810](https://github.com/marin-community/marin/issues/5810)).

## How it was trained

- **Oracle / labels**: `claude-sonnet-4-6` scoring documents with a **1–5 pretraining-utility rubric** (cached system prompt; documents wrapped in `<document>` tags so the model scores rather than completes instruction-tuned docs).
- **Training sample**: 7,000 docs **stratified across 104 active Datakit sources** using skew-aware allocation `floor=20 + extra ∝ sqrt(tokens)` so giant sources don't dominate. `safety_pt/*` and `climblab-ja` were carved out to match existing fan-outs. After 1,387 API/parse failures (mostly Sonnet rate-limit spikes), **5,613 usable labels** remained.
- **Binary target**: label = "rubric ≥ Average", i.e. a fixed **0.5 normalized threshold (= rubric raw ≥ 3)**. This gave a train balance of 3116 / 1936 (62% / 38%); a naive median split would have been ~87% positive, so the fixed cutoff was chosen for a healthier balance.
- **Cost & time**: total API spend $32.73 ($27.92 train + $4.81 eval), under the $50 cap; ~4.5 h wall clock (the all-sources inference fan-out was the long pole at ~3 h).

## How it was evaluated

Held-out set of **n=961 fresh oracle-labeled docs** (seed=43, never seen in training) ([#5810](https://github.com/marin-community/marin/issues/5810)):

| Metric | Value |
|---|---|
| AUC | 0.846 |
| Spearman ρ vs LLM score | 0.641 |
| Accuracy / Precision / Recall / F1 @ 0.5 | 0.78 / 0.74 / 0.70 / 0.72 |
| Base rate (positive) | 39% |

**Side-by-side vs the existing baseline** on the same held-out set:

| Spearman ρ | Value |
|---|---|
| ours vs LLM | 0.641 |
| `dolma3-fasttext-quality` vs LLM | 0.168 |
| ours vs dolma3 | 0.210 |

Interpretation from the issue: the new model tracks the Sonnet rubric ~4× better than dolma3, and the two largely **disagree** (ρ=0.21), i.e. it adds new signal rather than rediscovering dolma3 — plausibly because dolma3's rubric is more education-centric while this holdout spans synthetic + code sources.

**Per-source calibration spot check** (all 104 sources) surfaced useful caveats: math/QA/synthetic and academic sources (peS2o, PubMed) score highest; `starcoder2/ir_*` (LLVM IR) and `massive_function_calling` (tool JSON) score literally ~0.000 because the rubric treats their surface syntax as noise; every non-English `finepdfs` subset ranks bottom-third (possible English bias); and `nemotron-terminal`'s top docs share an identical template prefix, hinting the classifier may reward the wrapper rather than the trace ([#5810](https://github.com/marin-community/marin/issues/5810)).

## Where the weights live / can they be shared?

The trained model and all artifacts were written to an **internal Marin GCS bucket**, not to a public model hub ([#5810](https://github.com/marin-community/marin/issues/5810)):

```
gs://marin-eu-west4/datakit/llm-quality-classifier/
  model/sonnet46-thr05/{model.bin, metadata.json}
  eval/holdout-sonnet46-thr05.{tsv, summary.json}
  eval/vs-dolma3-sonnet46-thr05.{tsv, summary.json}
  inference/sonnet46-thr05/quality-llm/<source>_<hash>/part-*.parquet   # 104 sources
  samples/{train-n7000-seed42, eval-n1000-seed43}.parquet
  scored/{train-n7000-seed42-sonnet46, eval-n1000-seed43-sonnet46}.parquet
```

Code lives at `experiments/datakit/cluster/llm_quality/` (sample, score, train, eval_holdout, eval_vs_dolma3, all_sources_quality_llm), with a ~$1 smoke test at `exp_smoke.py`.

**On sharing**: The weights themselves are a single small fastText `model.bin`. Nothing in [#5810](https://github.com/marin-community/marin/issues/5810) records a public release, a license grant, or an explicit approval to share, and the artifact currently sits only in a private `marin-eu-west4` bucket — so as of the issue there is no public/shareable copy, and no maintainer statement authorizing distribution. From a technical standpoint the `.bin` is trivially portable, and Marin is an open project, but the labels were generated by Claude (Sonnet) against a rubric, so any redistribution should be checked against (a) Marin's own release process for classifier artifacts and (b) any terms attached to using model-generated labels/outputs. In short: **shareable in principle (small fastText binary, open project), but the issue itself does not grant or document permission to share it, and the weights are not currently published anywhere public.** This should be confirmed with the Marin maintainers before distribution.

> Note: this is a v0 model. A later, unrelated follow-up ([#6739](https://github.com/marin-community/marin/issues/6739)) explores a "fast-transformer document quality classifier" aiming to beat fastText at <1M FLOPs/token — a successor line of work, not the same artifact.
