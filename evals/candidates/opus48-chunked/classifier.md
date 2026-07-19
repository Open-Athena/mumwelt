# The Marin data-quality classifier (#5810): what it is, how it was trained and evaluated, and whether the weights can be shared

## TL;DR

The "data classifier" in [#5810](https://github.com/marin-community/marin/issues/5810) is a **binary fastText document-quality classifier distilled from a Claude-Sonnet LLM oracle**. It was trained on ~5.6k documents that `claude-sonnet-4-6` scored against a 1–5 pretraining-utility rubric, and it emits `attributes.high_score = P(high) ∈ [0,1]` for every document in Datakit. It was evaluated almost entirely on **oracle fidelity** — AUC 0.846 / Spearman ρ 0.641 on a held-out 961-document oracle-labeled set — and **never** by a downstream training ablation. It is the deployed quality axis of the datakit store (5 quality bins × 40 domain clusters = 200 mixing buckets), and it is now **known to be substantially broken**: it sorts documents by domain and modality rather than intrinsic quality. A blind intruder test scored its buckets at 0.292 against a 0.20 chance floor as early as May, five weeks before the July audit made the diagnosis explicit. Two separate replacement efforts are in flight, neither rolled out.

On **weight sharing: the corpus records no decision, no request, and no legal analysis about releasing this model.** The answer below explains what that silence does and does not license you to conclude.

---

## 1. What the model is

- **Filed** 2026-05-18 by `ravwojdyla-agent` as [#5810](https://github.com/marin-community/marin/issues/5810), "LLM-oracle fastText quality classifier", a child of the epic [#5360](https://github.com/marin-community/marin/issues/5360) "Data pipeline: quality scores + dedup param selection" (owner ihodes, May milestone).
- **Stated goal** ([#5810](https://github.com/marin-community/marin/issues/5810)): "Train a fastText quality classifier whose labels come from an LLM oracle scoring documents against a written rubric, and apply it across every active Datakit source as `attributes.high_score = P(high) ∈ [0, 1]`."
- **Architecture**: plain supervised **fastText** — a linear bag-of-hashed-n-grams classifier. Not a neural LM. Code at `experiments/datakit/cluster/llm_quality/`, later referred to as `experiments/datakit/cluster/quality/v0` and called "the deployed fasttext quality classifier" in [#6739](https://github.com/marin-community/marin/issues/6739).
- **Scope discipline**: the issue's explicit v0 **non-goals** were per-domain rubrics or per-domain classifiers, a multi-class/regression head, and threshold tuning beyond the rubric-aligned default; and it was to land "as a parallel attribute first," *not* to replace the incumbent dolma3 quality score in the consolidated mixture ([#5810](https://github.com/marin-community/marin/issues/5810)).

Do not confuse it with its sibling [#5812](https://github.com/marin-community/marin/issues/5812), which scores the same corpus with the **off-the-shelf** `allenai/dolma3-fasttext-quality-classifier` (pinned at `bb89085994fef638ca8dc2ca25169db328e314bb`) as a baseline. Both write an attribute named `high_score`, to different GCS prefixes; the LLM-oracle one is what feeds the store buckets.

## 2. How it was trained

All figures from the close-out run report, "Run 1: sonnet46-thr05" ([#5810 comment](https://github.com/marin-community/marin/issues/5810#issuecomment-4480271122), 2026-05-18):

| Element | Value |
|---|---|
| **Oracle** | `claude-sonnet-4-6`, 1–5 pretraining-utility rubric, cached system prompt, documents wrapped in `<document>` tags "to prevent the model from completing instruction-tuned docs instead" of scoring them |
| **Sampling** | 7,000 docs stratified across all 104 active Datakit sources; skew-aware allocation (`floor=20 + extra ∝ sqrt(tokens)`) "so giant sources don't dominate the training set" |
| **Usable labels** | **5,613** (1,387 API/parse failures dropped) |
| **Label binarization** | positive = normalized score ≥ 0.5, i.e. rubric raw ≥ 3 ("Average or better"). Train balance 3116/1936 (62/38) |
| **fastText hyperparameters** | `dim=100, epoch=5, lr=0.5, wordNgrams=2, minCount=3, bucket=200K`; vocabulary 67,934 |
| **Oracle API cost** | **$32.73** ($27.92 train + $4.81 eval), under the issue's $50 cap |
| **Wall clock** | ~4.5 h, long pole the all-sources inference fan-out at 3h 4m |
| **Carve-outs** | `safety_pt/*` and `climblab-ja` excluded, matching existing fan-out conventions |

Note the binarization choice was deliberate and consequential: the **median Sonnet score was raw=2**, so splitting at the median would have made ~87% of documents positive; the rubric-aligned threshold instead yields a 39% base rate.

Artifacts landed under `gs://marin-eu-west4/datakit/llm-quality-classifier/` — `model/sonnet46-thr05/{model.bin, metadata.json}`, the eval TSVs, and `inference/sonnet46-thr05/quality-llm/<source>_<hash>/part-*.parquet` across **104 sources, all `SUCCESS`** ([#5810 comment](https://github.com/marin-community/marin/issues/5810#issuecomment-4480271122)).

## 3. How it was evaluated

The evaluation plan in the issue body was scope, not a numeric target: "held-out oracle-labeled sample (fresh seed) for AUC + Spearman ρ; side-by-side comparison to dolma3-fasttext-quality on the same held-out set to confirm we're adding signal vs. the existing baseline" ([#5810](https://github.com/marin-community/marin/issues/5810)). No numeric success threshold was ever set.

**Holdout**: `eval-n1000-seed43` — 1,000 freshly sampled documents scored by the same oracle, **961 usable**, seed 43, "model never saw these." This same 961-doc set is reused as the standard harness by every later effort.

**Achieved** ([#5810 comment](https://github.com/marin-community/marin/issues/5810#issuecomment-4480271122)):

| Metric | Value |
|---|---|
| AUC | **0.846** |
| Spearman ρ vs LLM score | **0.641** |
| Accuracy / Precision / Recall / F1 @ 0.5 | 0.78 / 0.74 / 0.70 / **0.72** |
| Base rate (positive) | 39% |

**Baseline comparison**, same n=961: `ours vs LLM` ρ **0.641**; `dolma3-fasttext-quality vs LLM` ρ **0.168**; `ours vs dolma3` ρ 0.210. Read as: it predicts the Sonnet rubric ~4× better than the incumbent, and the two largely disagree — so it adds new signal rather than rediscovering dolma3. **No FineWeb-Edu, DCLM, or Nemotron-bucket baseline was run in #5810** (those comparisons came later, in #6739).

Two further, non-metric evaluations:

- **Per-source spot check** across all 104 sources ([#5810 comment](https://github.com/marin-community/marin/issues/5810#issuecomment-4481294989)). Top: `nemotron_specialized/math_textbooks` 0.968, `synthetic_student_teacher` 0.914, `cp/peS2o` and `cp/pubmed` 0.857. Bottom: `massive_function_calling` 0.000, `starcoder2/ir_{python,rust,cpp}` 0.000–0.001, `finepdfs/jpn_Jpan` 0.043. The report flagged its own failure modes at the time — "hard-format zeros" on LLVM-IR and tool-JSON that "look like noise to a general rubric but ARE valuable for code understanding," and possible template-prefix gaming. One genuinely positive finding: Chinese-language math textbooks topped the ranking, so it is "*not* an English-only classifier despite English-heavy training data."
- **Blind LLM intruder test** of bucket coherence — an LLM panel is shown 5 documents (4 from one bucket, 1 from another) and must spot the intruder, so **higher accuracy means more coherent buckets**, with chance at 0.20 ([#5811 comment](https://github.com/marin-community/marin/issues/5811#issuecomment-4570234285) and [#5812 comment](https://github.com/marin-community/marin/issues/5812#issuecomment-4570232958), via PR [#5822](https://github.com/marin-community/marin/pull/5822), 2026-05-29):

  | Kind | n | Correct | Acc | 95% Wilson CI |
  |---|--:|--:|--:|---|
  | domain-v0 (luxical K=40) | 100 | 71 | 0.710 | [0.615, 0.790] |
  | domain-dolma3 (weborganizer) | 100 | 49 | 0.490 | [0.394, 0.587] |
  | **quality-v0 (oracle LLM)** | 551 | 161 | **0.292** | [0.256, 0.332] |
  | quality-dolma3 (dolma3 fasttext) | 569 | 122 | 0.214 | [0.183, 0.250] |

  This is the most under-appreciated result in the whole thread. The **quality** buckets score 0.292 — barely above the 0.20 chance floor, and far below the 0.710 the domain buckets achieve. In other words, as early as 2026-05-29 the quality axis was measurably close to incoherent, which **prefigures the July diagnosis in #6849 by five weeks**. It beats the dolma3 quality baseline (0.214), but both are near chance; winning that comparison says little.

**The critical evaluation caveat**: there is **no downstream training ablation** anywhere in the corpus that trains a model on data filtered or mixed by this classifier and measures benchmark effect. Both #5810 and its successor are scored purely on agreement with the oracle. The project itself names this as the wrong test — [#6739 comment](https://github.com/marin-community/marin/issues/6739#issuecomment-4833511462): production filters "are validated by **downstream benchmark ablation** (train on filtered data, measure MMLU/ARC/PIQA/HellaSwag), never AUC against the annotator… **The honest test of a quality filter is a small-model data ablation across a CORE-style suite, not oracle-AUC.**"

## 4. How it is deployed

The score is **not used as a hard filter**. It is discretized into **5 quality bins** and crossed with **40 luxical-embedding domain clusters** to give **200 buckets**, which are what the automated data-mixing operates over ([#5360 comment](https://github.com/marin-community/marin/issues/5360#issuecomment-4464342884), 2026-05-18: "quality - via fastText classifier trained on llm-oracle - 5 buckets * domain - via luxical-one embedding clustering - 40 buckets… `gs://marin-eu-west4/datakit/store/v0.1_20260518`"). Buckets are named `c{cluster}q{quality}`, and by [#6449](https://github.com/marin-community/marin/issues/6449) (June MoE prep) the mix consumed 167 mixable bucket caches plus a 33-cache tail group, ~10.37 T tokens. Asked directly on 2026-06-02 whether the oracle-LLM scores are the ones in use, ravwojdyla confirmed: "yes, they are used to the ongoing mixing swarm" ([#5812 comment](https://github.com/marin-community/marin/issues/5812#issuecomment-4599394015)).

Scoring cost was characterized up front in [#5406](https://github.com/marin-community/marin/issues/5406) — though note that benchmark measured the **two off-the-shelf** fastText classifiers, not the #5810 model, so read it as a cost model for fastText-class scoring generally rather than a measurement of this classifier. Measured single-core rates were **1,747 docs/s (quality) and 2,078 docs/s (edu)**; extrapolated to 15 B documents that is ~99.4 days on 1 CPU, or ~2.4 h on 1000 perfectly-parallel CPUs ([#5406 comment](https://github.com/marin-community/marin/issues/5406#issuecomment-4372487617)). The actual #5810 all-sources inference fan-out took **3h 4m**.

## 5. Known limitations — the deployed model is broken in a way that matters

Three issues filed 2026-07-02 from a ducky-SQL audit of the clustered store, **all still open** as of the corpus freeze:

- **[#6849](https://github.com/marin-community/marin/issues/6849) — it sorts by domain, not intrinsic quality.** "the 5 quality buckets sort by source, not quality." Clean, correct `starcoder2/ir_python` scores ~0.00 (88–99% of docs exactly 0); pristine `cp/arxiv_abstracts` lands in q0; multilingual raw mean 0.18 vs a type-aware oracle mean of 0.51; **bucket q4 is code+math only (0% multilingual, 0% prose)**. The root cause is structural, not a tuning bug: "source alone predicts the oracle at **AUC 0.852**… so any faithful distillation reproduces the domain bias — **raising oracle-AUC does not fix it**." A [follow-up comment](https://github.com/marin-community/marin/issues/6849#issuecomment-4870686601) adds that the score has no absolute anchor — "`0.85` means 'excellent arXiv' in one place and 'a fan-wiki parody cast list' in another."
- **[#6859](https://github.com/marin-community/marin/issues/6859) — long documents are judged on a truncated 4 KB lead.** The rubric caps input at `MAX_TEXT_CHARS = 4000`. `starcoder2/documentation`'s largest document is **37.8 MB**, and docs >100 KB are 14% of that source's documents but **80.5% of its characters**: "A 37 MB document is quality-scored on its first ~4 KB (**≈ 0.01%**)."
- **[#6860](https://github.com/marin-community/marin/issues/6860) — near-constant scores within many sources.** `massive_function_calling` has stddev **0.0** and one distinct value; **12 of 98 sources have score stddev < 0.1**.

Rafal Wojdyla's own framing, quoted in the [Jun 29–Jul 5 weekly summary](https://mws.oa.dev/summaries/summary-2026-06-29_2026-07-05.html): the classifiers are intentionally v0 — "the goal was to have e2e pipeline. We can definitely do better!"

## 6. Two replacement tracks (do not conflate them)

**(a) Oracle fidelity under a FLOPs budget — [#6739](https://github.com/marin-community/marin/issues/6739) / PR [#6741](https://github.com/marin-community/marin/pull/6741)** (rjpower, closed out 2026-06-29). A pooled "fast-transformer" (`embed → pool@64-token boundaries → transformer layers → regression head`), trained on the *same* 5,613 labels and scored on the *same* 961 holdout:

| model | AUC | ρ | FLOPs/tok |
|---|--:|--:|--:|
| fastText v0 `sonnet46-thr05` | 0.846 | 0.641 | ~0 |
| fast-transformer meanmaxmin·w64·d512·L4 | **0.875** | **0.703** | 0.41M (budget was <1M) |

Caveat: its accuracy/F1 (0.788/0.751) use a **val-calibrated threshold of 0.329**, not 0.5 — at a fixed 0.5 it scores *worse* than fastText. AUC and ρ are threshold-free and win outright. Attention contributes almost nothing: the `L=0` variant reaches ρ 0.699 at **12K FLOPs/token**, 34× cheaper. Distilling three free teachers (source prior, Nemotron-CC buckets, FineWeb-Edu) all landed **below** training from scratch, yielding the diagnosis that **~0.87 is a label-quality ceiling**, breakable only by paying for more oracle labels (~$20–80 for 5–20k docs, ~$200–400 for ~100k) ([#6739 comment](https://github.com/marin-community/marin/issues/6739#issuecomment-4833679263)). *One 0.877/0.715 run is explicitly disclaimed as unreproduced and 14× over budget — it should not be cited as a result.* I found no evidence the fast-transformer was ever wired into the production inference path.

**(b) Bucket coherence — PR [#7040](https://github.com/marin-community/marin/pull/7040)**, "datakit/quality: coherent quality buckets via a calibrated pooled classifier" (Rafal Wojdyla). A content-type-aware, **source-blind** rubric plus a pooled ranker (branch `rav/quality-coherence`, `experiments/datakit/cluster/quality/calibrated/`), claiming Spearman vs coherent-quality **0.44→0.69** ([#6849](https://github.com/marin-community/marin/issues/6849)). This targets a *different* objective than (a), since #6849 states that raising oracle-AUC does not fix domain sorting. **Status: an open RFC, +2083/−0 with zero comments** in the [Jul 6–12 weekly summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html); its body is not in the frozen corpus, so I cannot quote its design. A sibling open question, [#7124](https://github.com/marin-community/marin/issues/7124) "decide quality bucketization scheme (absolute vs quantile)", was newly filed the same week.

**Bottom line on current state**: as of the corpus freeze, **v0 fastText is still the deployed scorer**, with no replacement rolled out.

## 7. Can the model weights be shared?

**The honest answer: the corpus contains no discussion of this at all, so there is no decision to report — but the question is live and unresolved, and there is a specific reason it is not trivially "yes."**

What I can establish:

1. **Nothing in the #5810 thread addresses release or licensing.** The thread is fully enumerated in the frozen corpus — three body chunks and two comments — and none mentions weights, release, licensing, or terms of service. Independent searches on release/licensing phrasings surfaced no discussion of releasing *any* Marin quality classifier.
2. **The artifacts are in a private working bucket**, `gs://marin-eu-west4/datakit/llm-quality-classifier/model/sonnet46-thr05/model.bin` — not in `marin-public`, the durable public-hosting bucket Marin does maintain for shareable artifacts ([#6816](https://github.com/marin-community/marin/pull/6816)). Notably, the *analysis report* for the successor classifier was published publicly (`storage.googleapis.com/marin-public/rav/quality-score-debugging/…`) while no model artifact was.
3. **The blocking consideration is that every weight in this model is derived from `claude-sonnet-4-6` outputs.** The model is by construction a distillation of a commercial LLM's judgments — that is its entire training signal.
4. **Marin has clear precedent for treating exactly this as a real constraint.** In [#5745](https://github.com/marin-community/marin/issues/5745), Will Held excluded the GPT-5 portions of AgentTrove "due to anti-distill ToS from OpenAI," and when the agent noted adjacent rows from other OpenAI teachers, he escalated to a **positive allowlist** — "To avoid any accidental use of data with unclear ToS, let's do a positive type filter!" Separately, [#5770](https://github.com/marin-community/marin/issues/5770) records the general shape of the problem for weight release: OEIS's "license is CC BY-SA 4.0, which makes training a model and then **releasing the weights** under Apache complicated."

**What follows.** Marin demonstrably reasons about provider anti-distillation terms when *ingesting* another lab's model outputs as training data, and demonstrably reasons about how training-data licenses constrain *weight release*. This classifier sits precisely at the intersection of those two precedents, yet no one in the corpus has joined them up for it. So the accurate statement is: **sharing the weights is an open question requiring a terms-of-service review that has not happened**, and given Marin's own conservative posture in #5745 it should not be assumed to be clear. The same consideration applies to the fast-transformer in #6741 and the calibrated classifier in #7040, since both are trained on the identical Sonnet-derived labels.

Two things I want to be explicit about, because they are the tempting errors here:

- I am **not** asserting that Anthropic's terms forbid this — no such analysis appears in the corpus, and I am not in a position to substitute my own reading of a ToS for one.
- The absence of discussion is **not** evidence that release is fine. It is most consistent with the artifact simply never having been proposed for release: it is an internal v0 pipeline component that is currently known-broken (#6849/#6859/#6860), which is not a checkpoint anyone would be rushing to publish.

If someone needs an actual answer, the corpus suggests the path: raise it with the epic owners (ravwojdyla for the classifier, ihodes for #5360/#6713) and route it the way #5745 was routed — a ToS check before anything moves to `marin-public`. A releasable classifier would most plausibly require relabeling with a permissively-licensed oracle; the cost of a fresh label set is already scoped at roughly $200–400 for ~100k documents ([#6739 comment](https://github.com/marin-community/marin/issues/6739#issuecomment-4833679263)), though that estimate assumes the Sonnet oracle.

## 8. Gaps and caveats

- **No downstream validation exists** for any version of this classifier — the single biggest evidentiary gap, and one the project acknowledges (#6739).
- **PR #7040's body is not in the frozen corpus**; its design is reconstructed from #6849/#6860 and the weekly summary, and its 0.44→0.69 claim is sourced to an issue comment rather than a merged PR.
- **No weight-sharing discussion exists**, as detailed above — reported as a gap, not inferred into an answer.
- The 0.877/0.715 figure occasionally visible in #6739's sweeps is **unreproduced and over budget**; the defensible successor numbers are 0.875/0.703.

---
<!--provenance-->
> *Data: marinmirror — 86367 chunks, built 1h ago · summaries through 2026-07-06_2026-07-12. Frozen eval corpus (`mum-frozen-chunked`); no refresh performed.*
>
> *Query: "What is the data-classifier model and how was it trained and evaluated? (See marin-community/marin#5810.) Can the model weights be shared?"*
>
> *Sub-queries: "#5810 classifier design — oracle model, rubric, label counts, fastText config" · "evaluation — holdout, AUC/Spearman/F1, baselines, downstream ablations" · "weight sharing — licensing, ToS, anti-distillation, HuggingFace release, or an explicit decision not to release" · "deployment — attributes.high_score across Datakit sources, buckets, mixing" · "known limitations and successors — #6849/#6859/#6860, #6739/#6741, #7040 calibrated pooled classifier"*
