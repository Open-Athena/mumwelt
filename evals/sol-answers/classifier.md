# The data-classifier model (#5810): training, evaluation, and weight sharing

## What was trained

#5810 built a **fastText document-quality classifier distilled from an LLM oracle**. Claude Sonnet 4.6 scored documents on a written 1–5 “pretraining utility” rubric; documents were wrapped in `<document>` tags to reduce prompt-following/completion artifacts. The binary fastText target was derived from those scores, and the model's output was applied as a continuous `attributes.high_score = P(high)` value across all active Datakit sources ([#5810](https://github.com/marin-community/marin/issues/5810)).

The first recorded pipeline run (`sonnet46-thr05`) completed end-to-end over all **104 active sources**. The issue also records a per-source spot check from one random shard/row-group per source, with sample counts ranging from roughly 500 to 800k rows; that is a deployment sanity check, not a held-out quality evaluation ([#5810 run comment](https://github.com/marin-community/marin/issues/5810#issuecomment-4481294989)).

## How it was evaluated

The strongest explicit evaluation in the frozen corpus comes from the later replacement study, which reused #5810's labels and harness. It split **5,613 oracle-labeled training documents** and **961 held-out documents**, and scored models by threshold-free agreement with the oracle: ROC AUC and Spearman correlation ([#6739](https://github.com/marin-community/marin/issues/6739)). On that ruler:

- deployed v0 fastText: **AUC 0.846, Spearman ρ 0.641**;
- best pooled fast-transformer: **AUC 0.875, ρ 0.703**, at **0.41M FLOPs/token**.

The transformer was a small `embed → pool at 64-token boundaries → transformer layers → regression head` model ([#6739](https://github.com/marin-community/marin/issues/6739), [PR #6741](https://github.com/marin-community/marin/pull/6741)). Capacity/context sweeps, next-token pretraining on 240k free Nemotron documents, weak source-prior pretraining, and FineWeb-Edu-score distillation did not beat the supervised plateau; #6739 therefore closes those as negative follow-ups rather than presenting them as production wins.

Two limitations are load-bearing:

1. The held-out ruler measures agreement with the Sonnet oracle, **not downstream language-model quality**. #6739 says the oracle is education-leaning and explicitly distinguishes oracle validation from downstream validation ([#6739](https://github.com/marin-community/marin/issues/6739)).
2. Production analysis found a construct-validity failure: the v0 fastText model largely separated domains/modalities/languages instead of intrinsic quality within a domain, and its top bucket lacked an absolute “excellent content” anchor ([#6849](https://github.com/marin-community/marin/issues/6849)). Thus good oracle AUC does not establish that thresholding the score improves a pretraining run.

## Can the weights be shared?

The frozen corpus supports **internal reuse**: #5810 produced a model artifact that was applied across all 104 Datakit sources, and the later #6739 harness reused its labels/baseline. But it does **not** record a public Hugging Face release, a public artifact URL, a license for the trained weights, or an explicit approval to redistribute them. Therefore the evidence-backed answer is:

- **Technically:** yes, it is a small ordinary fastText artifact and is already reused inside the pipeline.
- **Publicly / legally:** **not established by the frozen corpus**. Do not promise the weights can be published until the artifact owner confirms the output license and whether weights distilled from Sonnet-generated labels may be redistributed.

The open implementation is not the same thing as released weights. PR #6741 exposes the later fast-transformer implementation and experiment harness, but that also does not by itself grant access to #5810's trained artifact ([PR #6741](https://github.com/marin-community/marin/pull/6741)).

**Verified sources / gap note.** Re-opened #5810 and its deployment spot-check, #6739, PR #6741, and #6849. Every reported metric is from the held-out oracle-agreement study. The public-sharing answer remains an explicit gap because no frozen source states a release/license decision.

---
<!--provenance-->
> *Data: frozen marinmirror — 68026 chunks, built 2h ago · summaries through 2026-07-06_2026-07-12. No refresh (frozen runner only).*
>
> *Query: "the data-classifier model (see #5810): how trained/evaluated, can weights be shared?"*
>
> *Sub-queries: "#5810 oracle and rubric" · "fastText training pipeline" · "holdout metrics" · "fast-transformer replacement" · "downstream/construct validity" · "artifact and weight release" · "license/shareability". HyDE vector retrieval was used for architecture, evaluation, and release facets.*
