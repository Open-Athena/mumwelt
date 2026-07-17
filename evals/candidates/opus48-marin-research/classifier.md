# The Marin data-quality classifier: what it is, how it was trained and evaluated, and whether the weights can be shared

## What the model is

The "data classifier" is the **LLM-oracle fastText quality classifier** tracked in [marin-community/marin#5810](https://github.com/marin-community/marin/issues/5810), a child of the May datakit "quality scores" one-pager [#5360](https://github.com/marin-community/marin/issues/5360). Its job is to attach a continuous document-quality signal — `attributes.high_score = P(high) ∈ [0, 1]` — to every active Datakit source, co-partitioned with normalized parquet so it drops into the existing `classify_fasttext_step` consumer shape alongside the incumbent `dolma3-quality` classifier ([#5810](https://github.com/marin-community/marin/issues/5810)).

Key design choices for v0: a **single rubric across all sources**, a **binary fastText model** (label = "rubric ≥ Average") that emits `P(high)` as a continuous score at inference, and it lands as a *parallel* attribute rather than replacing dolma3-quality in the mixture ([#5810](https://github.com/marin-community/marin/issues/5810)). The shipped v0 run is named **`sonnet46-thr05`** ([#5810 comment](https://github.com/marin-community/marin/issues/5810#issuecomment-4480271122)).

A follow-on experiment, [#6739](https://github.com/marin-community/marin/issues/6739) (PR [#6741](https://github.com/marin-community/marin/pull/6741)), later built a small **fast-transformer** that beats this fastText model under a <1M FLOPs/token budget — trained on the *same* Sonnet-4.6 oracle labels — so "the classifier" now spans two generations (fastText v0, then the pooled fast-transformer).

## How it was trained

The label source is an **LLM oracle**, not human annotation ([#5810](https://github.com/marin-community/marin/issues/5810)):

- **Oracle**: `claude-sonnet-4-6` scoring documents against a **1–5 pretraining-utility rubric**, with a cached system prompt; documents are wrapped in `<document>` tags "to prevent the model from completing instruction-tuned docs instead of scoring them" ([#5810 comment](https://github.com/marin-community/marin/issues/5810#issuecomment-4480271122)).
- **Sampling**: stratified across the **104 active Datakit sources** (carving out `safety_pt/*` and `climblab-ja`), skew-aware via `floor=20 + extra ∝ sqrt(tokens)` so giant sources don't dominate. Target 7,000 docs; after 1,387 API/parse failures (mostly Sonnet rate-limit spikes) → **5,613 usable labels** ([#5810 comment](https://github.com/marin-community/marin/issues/5810#issuecomment-4480271122)).
- **Binary label**: normalized threshold **0.5** (= rubric raw ≥ 3, "Average or better"). Train balance 3,116 / 1,936 (62% / 38%). The fixed 0.5 cutoff was chosen deliberately over a median split (median Sonnet score was raw=2, which would have made ~87% of docs positive) ([#5810 comment](https://github.com/marin-community/marin/issues/5810#issuecomment-4480271122)).
- **fastText hyperparameters**: `dim=100, epoch=5, lr=0.5, wordNgrams=2, minCount=3, bucket=200K`; resulting vocab = 67,934 ([#5810 comment](https://github.com/marin-community/marin/issues/5810#issuecomment-4480271122)).
- **Cost / wall clock**: **$32.73** API spend ($27.92 train + $4.81 eval), under the $50 cap; ~4.5 h total wall clock ([#5810 comment](https://github.com/marin-community/marin/issues/5810#issuecomment-4480271122)).

Code lives at `experiments/datakit/cluster/llm_quality/` (sample → score → train → eval_holdout → eval_vs_dolma3 → all_sources), with a ~$1 smoke test ([#5810 comment](https://github.com/marin-community/marin/issues/5810#issuecomment-4480271122)).

## How it was evaluated

Evaluation is against a **held-out oracle-labeled sample** (n=961 fresh docs, seed=43, unseen by the model) ([#5810 comment](https://github.com/marin-community/marin/issues/5810#issuecomment-4480271122)):

| Metric | Value (fastText v0 `sonnet46-thr05`) |
|---|---|
| AUC | **0.846** |
| Spearman ρ vs LLM score | **0.641** |
| Accuracy / Precision / Recall / F1 @ 0.5 | 0.78 / 0.74 / 0.70 / **0.72** |
| Base rate (positive) | 39% |

**Side-by-side vs the existing `dolma3-fasttext-quality` baseline** on the same 961-doc set: ours ρ=**0.641** vs LLM, dolma3 ρ=**0.168** vs LLM, and ours-vs-dolma3 ρ=**0.210** — i.e. the new model predicts the Sonnet rubric ~4× better than dolma3 and the two largely disagree, so it's adding new signal rather than rediscovering dolma3 ([#5810 comment](https://github.com/marin-community/marin/issues/5810#issuecomment-4480271122)).

Beyond headline metrics, a **per-source spot check** across all 104 sources surfaced calibration caveats worth noting ([#5810 comment](https://github.com/marin-community/marin/issues/5810#issuecomment-4481294989)): multilingual edu content ranks top (Chinese math textbooks at mean 0.968, so *not* English-only); "hard-format" sources like `starcoder2/ir_*` and `massive_function_calling` get literally 0.000 mean (the rubric rejects the surface syntax wholesale); a possible **English bias** inside non-English `finepdfs`; and **template-prefix gaming** where `nemotron-terminal`'s top docs all share an identical instruction-template wrapper. Later issues [#6849](https://github.com/marin-community/marin/issues/6849) and [#6860](https://github.com/marin-community/marin/issues/6860) flag related failure modes (sorting by domain rather than intrinsic quality; near-constant scores within some sources).

The follow-on **fast-transformer** ([#6739](https://github.com/marin-community/marin/issues/6739)) was evaluated apples-to-apples (same Sonnet-4.6 splits, same `eval_holdout` code) and beat fastText on every metric — **AUC 0.875 / ρ 0.703 at 0.41M FLOPs/token** vs fastText's 0.846 / 0.641 ([#6739 final report](https://github.com/marin-community/marin/issues/6739#issuecomment-4826633470)). The team concluded ~0.87 AUC is a **label-quality ceiling** for the 5.6k-doc oracle set — more *diverse oracle-style* labels, not more free web-quality labels, is the bottleneck ([#6739 state comment](https://github.com/marin-community/marin/issues/6739#issuecomment-4833334278)).

## Can the model weights be shared?

**The frozen corpus contains no explicit decision on releasing the classifier weights** — I could not find a thread that rules yes or no on distributing `model.bin`. What the corpus does establish are two facts that bear directly on the question, so I'll flag the tension rather than manufacture an answer:

1. **The model is entirely distilled from Claude.** Both the fastText v0 and the fast-transformer are trained on labels produced by **Anthropic's Claude Sonnet 4.6** oracle ([#5810 comment](https://github.com/marin-community/marin/issues/5810#issuecomment-4480271122); [#6739 report](https://github.com/marin-community/marin/issues/6739#issuecomment-4826633470)). The weights currently live in an **internal GCS bucket**, `gs://marin-eu-west4/datakit/llm-quality-classifier/model/sonnet46-thr05/{model.bin, metadata.json}` — an artifact store, not a public release ([#5810 comment](https://github.com/marin-community/marin/issues/5810#issuecomment-4480271122)).

2. **Marin explicitly positions itself as a "truly open" project that tries to avoid frontier-model distillation.** In a Discord discussion on 2026-07-15, Colin Raffel framed Olmo/Apertus/**Marin** as projects that "try to avoid" distillation attacks, and Percy Liang responded that "the claims of 'it's just distillation' … are kind of overblown, but clearly getting Claude traces is a pretty convenient shortcut," adding that the whole distillation question is "a giant mess" ([#questions Discord, 2026-07-15](https://discord.com/channels/1354881461060243556/1357080963472949428/1527019392821366794)).

Putting these together: a classifier whose labels come wholesale from Claude sits squarely in the middle of the distillation-provenance question Marin is publicly cautious about, and provider terms on using model outputs are the usual constraint on redistributing such artifacts. **On the evidence in this corpus the sharing question is unresolved** — the weights exist and are internally accessible, but nothing here documents a decision (or a legal/policy clearance) to release them openly. Treat "shareable" as an open governance question, not a settled yes.

---
<!--provenance-->
> *Data: marinmirror — 68026 chunks, built 1h ago · summaries through 2026-07-06_2026-07-12 (frozen 2026-07-16 eval corpus; no refresh).*
>
> *Query: "What is the data-classifier model and how was it trained and evaluated? (See marin-community/marin#5810.) Can the model weights be shared?"*
>
> *Sub-queries: "LLM-oracle fastText quality classifier training" · "quality classifier oracle rubric Claude sonnet labels" · "quality classifier evaluation ablation downstream eval accuracy" · "share classifier model weights license open source" · "Anthropic terms distill Claude outputs competing model" · "can we release open weights quality classifier distilled from Claude" · "Marin truly open avoid distillation attacks (Discord)"*
