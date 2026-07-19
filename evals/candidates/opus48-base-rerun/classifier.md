# The Marin data-quality classifier (#5810): what it is, how it was trained and evaluated, and whether the weights can be shared

**Short answer.** The "data classifier" in [marin#5810](https://github.com/marin-community/marin/issues/5810) is a **binary fastText document-quality classifier distilled from a Claude Sonnet 4.6 oracle**. Run 1 (`sonnet46-thr05`, 2026-05-18) trained it on ~5.6k LLM-labeled documents and scored all 104 active Datakit sources with `attributes.high_score = P(high)`. It was evaluated against a held-out oracle-labeled set (**AUC 0.846, Spearman ρ 0.641**) and beat the `dolma3-fasttext-quality` baseline by ~4× on oracle agreement. It then became the **quality axis of the production data store** — and, as of the corpus freeze, it is still deployed despite three July issues documenting that it is substantially broken. On **sharing the weights: the corpus contains no discussion of this at all** — no release decision, no licensing analysis, no mention of Anthropic terms. That is a genuine gap, not a "no."

---

## 1. What the model is

From the [issue body](https://github.com/marin-community/marin/issues/5810) (`ravwojdyla-agent`, 2026-05-18), filed as a child of the May Datakit epic [#5360](https://github.com/marin-community/marin/issues/5360):

> "Train a fastText quality classifier whose labels come from an LLM oracle scoring documents against a written rubric, and apply it across every active Datakit source as `attributes.high_score = P(high) ∈ [0, 1]` — co-partitioned with normalized parquet so it slots into the existing `classify_fasttext_step` consumer shape (drop-in alongside dolma3-quality)."

So it is deliberately *not* a new inference architecture — it reuses the existing fastText serving path so it can land as a **parallel attribute** next to `dolma3-quality`. The v0 non-goals are explicit: no per-domain rubrics, no multi-class/regression head, no threshold tuning, and "Replacing dolma3-quality in the consolidated mixture — this lands as a parallel attribute first."

Code lives at `experiments/datakit/cluster/llm_quality/` (sample, score, train, eval_holdout, eval_vs_dolma3, all_sources_quality_llm), with a `exp_smoke.py` covering the full chain for ~$1 ([#5810 Run 1](https://github.com/marin-community/marin/issues/5810#issuecomment-4480271122)).

## 2. How it was trained

All figures below are **measured**, from [#5810 comment 4480271122](https://github.com/marin-community/marin/issues/5810#issuecomment-4480271122) (2026-05-18), reporting run `sonnet46-thr05` "complete end-to-end across all 104 active sources."

**Oracle (the label source).**
> "**Oracle**: claude-sonnet-4-6 with a 1-5 pretraining-utility rubric, cached system prompt; documents wrapped in `<document>` tags to prevent the model from completing instruction-tuned docs instead of scoring them."

**Sampling.** 7,000 documents stratified across 104 sources via `floor=20 + extra ∝ sqrt(tokens)` — skew-aware so giant sources don't dominate. `safety_pt/*` and `climblab-ja` were carved out to match existing fan-outs. After "1,387 API/parse failures (mostly Sonnet rate-limit spikes late in the run)", **5,613 usable labels** remained.

**Label thresholding.** Binary at 0.5 normalized (= rubric raw ≥ 3, "Average or better"), giving a train balance of 3116/1936 (62%/38%). The choice was deliberate:
> "Median Sonnet score landed at raw=2, so a median split would have been ~87% positive — the fixed 0.5 cutoff gives a much healthier balance."

**Hyperparameters.** `dim=100, epoch=5, lr=0.5, wordNgrams=2, minCount=3, bucket=200K`; vocab = 67,934.

**Cost and wall clock.** API spend **$32.73** ($27.92 train + $4.81 eval) against a **$50 cap** (the cap is the target, the spend is the measurement). ~4.5 h total wall clock, long pole the all-sources fan-out at 3h 4m.

**Artifacts** are in `gs://marin-eu-west4/datakit/llm-quality-classifier/` — `model/sonnet46-thr05/{model.bin, metadata.json}`, the two eval TSV/summary pairs, per-source inference parquet for all 104 sources (all `SUCCESS`), and the `samples/` + `scored/` splits (`train-n7000-seed42`, `eval-n1000-seed43`).

There is **only one training run** in the corpus. No Run 2, no retrain of this model exists anywhere; successors are different model families or different rubrics (§5).

## 3. How it was evaluated

**Primary eval — held-out oracle agreement.** n=961 fresh oracle-labeled documents, seed=43, "model never saw these" ([#5810](https://github.com/marin-community/marin/issues/5810#issuecomment-4480271122)):

| Metric | Value |
|---|---|
| AUC | 0.846 |
| Spearman ρ vs LLM score | 0.641 |
| Acc / Prec / Rec / F1 @ 0.5 | 0.78 / 0.74 / 0.70 / 0.72 |
| Base rate (positive) | 39% |

**Baseline comparison** on the same n=961 set: ours-vs-LLM ρ = **0.641**, `dolma3-fasttext-quality`-vs-LLM ρ = **0.168**, ours-vs-dolma3 ρ = **0.210**. The reading given: "ours predicts the Sonnet rubric ~4× better than dolma3, and the two largely disagree (ρ=0.21) — adding new signal rather than rediscovering dolma3."

**Secondary eval — per-source spot check.** [#5810 comment 4481294989](https://github.com/marin-community/marin/issues/5810#issuecomment-4481294989) swept mean `P(high)` across all 104 sources. Top: `nemotron_specialized/math_textbooks` 0.968, `nemotron_code_v2/synthetic_student_teacher` 0.914, `cp/peS2o` and `cp/pubmed` 0.857. Bottom: `massive_function_calling` **0.000**, `starcoder2/ir_python` 0.000, `finepdfs/jpn_Jpan` 0.043, `cp/wikiteam` 0.069. The same comment names three failure modes that later prove central — hard-format zeros on LLVM IR and tool-JSON ("valuable for tool-using LLMs, looks like config noise to the rubric"), English bias inside `finepdfs`, and possible template-prefix gaming on `nemotron-terminal` where "every top doc starts `<user> You are an AI assistant tasked with solving command-line tasks…`".

**Third eval — blind LLM intruder study.** [PR #5822](https://github.com/marin-community/marin/pull/5822) (2026-05-18) arbitrated the competing quality and domain arms: quality-v0 (oracle LLM) **0.292 vs 0.214** for quality-dolma3, ~560/kind, "Wilson CIs separated; z=2.99, p≈0.003". **Read this carefully**: these are accuracies on a hard intruder task, and both are low in absolute terms — the domain arms in the same study run 0.710 vs 0.490. This is a *relative* separation reported as a starting point, and rav framed it as such: "This should be enough to start, the goal of June will be to improve these."

**A methodological caveat from within the team.** [#6739 comment 4833511462](https://github.com/marin-community/marin/issues/6739#issuecomment-4833511462) (rjpower, 2026-06-29) argues the whole eval frame is weak: "Our Sonnet oracle is effectively an educational-value classifier", oracle-AUC measures only *annotator fidelity*, and "the honest test of a quality filter is a small-model data ablation … not oracle-AUC." No downstream data ablation of this classifier appears in the corpus.

## 4. It shipped, and it is still the deployed model

The classifier won the quality slot in the production Datakit store — [#5360 comment 4483228852](https://github.com/marin-community/marin/issues/5360#issuecomment-4483228852) (2026-05-18):

> "quality - via fastText classifier trained on llm-oracle - 5 buckets * domain - via luxical-one embedding clustering - 40 buckets. This gives us 200 tokenized buckets, you can find them under `gs://marin-eu-west4/datakit/store/v0.1_20260518`"

Confirmed in live use on 2026-06-02, when hammer asked whether the oracle-LLM scores were the ones being used and ravwojdyla replied "yes, they are used to the ongoing mixing swarm" ([#5812](https://github.com/marin-community/marin/issues/5812#issuecomment-4599394015)). As of 2026-07-03 the reference pipeline still runs `--quality-model gs://marin-eu-west4/datakit/llm-quality-classifier/model/sonnet46-thr05/model.bin` ([#6888](https://github.com/marin-community/marin/issues/6888)).

## 5. The July verdict: the deployed classifier is broken, and nothing has replaced it yet

Three issues filed 2026-07-02 audit it harshly:

- **[#6849](https://github.com/marin-community/marin/issues/6849)** — "The fastText document-quality classifier … scores documents by domain / modality / language rather than intrinsic quality, so the 5 quality buckets sort by source, not quality." Root cause is structural, not fixable by better distillation: "the Sonnet rubric distills a generic 'pretraining value' target that is ~recoverable from source identity (source alone predicts the oracle at AUC 0.852, #6739), so any faithful distillation reproduces the domain bias — **raising oracle-AUC does not fix it**."
- **[#6859](https://github.com/marin-community/marin/issues/6859)** — the rubric caps input at `MAX_TEXT_CHARS = 4000`, so "A 37 MB document is quality-scored on its first ~4 KB (≈ 0.01%)."
- **[#6860](https://github.com/marin-community/marin/issues/6860)** — near-constant within-source scores: "`massive_function_calling`: stddev **0.0** … **12 of 98 sources have score stddev < 0.1**."

**Two successors exist; neither is deployed as of the freeze.** The fast-transformer ([#6739](https://github.com/marin-community/marin/issues/6739) / [PR #6741](https://github.com/marin-community/marin/pull/6741)) measures **AUC 0.875 / Spearman 0.703** on the same oracle splits, beating fastText's 0.846/0.641 — but the thread closed on a pending "Decision point: (a) ship the current pooled model … and/or (b) approve a targeted Sonnet-oracle spend", with rjpower "Holding for direction before any spend", and the diagnosis that ~0.87 is a ceiling set by label quality. The coherence fix on branch `rav/quality-coherence` (Spearman vs coherent-quality 0.44→0.69) surfaced as **open PR #7040**, still in review in the week of 2026-07-06. The most recent weekly-summary framing is blunt: "the store's domain and quality axes are largely uninformative," with rav's own framing that "both domain and quality classifiers are v0 - the goal was to have e2e pipeline. We can definitely do better!" ([#data-mixing, 2026-06-29](https://discord.com/channels/1354881461060243556/1462895580064911522/1521206652320223323)).

## 6. Can the model weights be shared?

**The corpus does not answer this.** Two independent passes — a dedicated retrieval agent and an adversarial skeptic tasked with refuting the first — both came back dry. [#5810](https://github.com/marin-community/marin/issues/5810) contains **zero** occurrences of `releas`, `licen`, `public`, `hugging`, `weight`, `open-sourc`, `terms`, or `distill`-in-a-legal-sense; the only statement about where the model lives is the operational artifact path `gs://marin-eu-west4/datakit/llm-quality-classifier/model/sonnet46-thr05/model.bin`, given with no rationale. Every inbound reference to #5810 (#5812, #6739, #6849, #6859, #6860) is about classifier quality. Literal corpus-wide scans for `commercial terms`, `competing model`, `Anthropic's terms`, `acceptable use policy`, `can we release`, and `legal counsel` all returned nothing.

**So the honest answer is: no decision has been made or recorded.** What the corpus *does* establish, as adjacent context that should not be mistaken for a ruling on this artifact:

- Marin's default posture is aggressively open — "Marin is about open source and open science" (dlwh, [#jobs, 2025-12-04](https://discord.com/channels/1354881461060243556/1446182749299015771/1446182918472077322)); the repo is Apache 2.0 with SPDX headers ([PR #2716](https://github.com/marin-community/marin/pull/2716)); models and datasets do ship to the HF org (`marin-community/marin-8b-base`, `marin-community/medu-science-qa` via [PR #1276](https://github.com/marin-community/marin/pull/1276)).
- That openness is understood to *raise* the compliance bar rather than lower it. dlwh, on NVIDIA's restrictive Nemotron-CC-v2 license ([#marin-32b, 2025-09-07](https://discord.com/channels/1354881461060243556/1367246647062564995/1414124066050281574)): "we do tell people basically everything so like we do have to play by the rules more than some others might."
- The one place the train-then-distill legal question is raised at all is that same thread — HessianFree: "What if a model is trained on it and then you distill? … It's unclear what the legal bindings of that is no?" ([2025-09-07](https://discord.com/channels/1354881461060243556/1367246647062564995/1414236346465255474)), with elie replying it's "a broader question I don't have the answer to". That was eight months before #5810 existed and nobody connected it to the Claude-labeled classifier.
- Related *analysis* is already published world-readably at `storage.googleapis.com/marin-public/rav/quality-score-debugging/`, via the `public_artifacts` mechanism in [PR #6816](https://github.com/marin-community/marin/pull/6816) — precedent for publishing outputs, but not a decision about the weights.

The load-bearing fact for anyone deciding this: the model's **entire training signal is `claude-sonnet-4-6` output** ([#5810](https://github.com/marin-community/marin/issues/5810#issuecomment-4480271122)), so releasability turns on Anthropic's terms governing use of model outputs — a question the corpus shows Marin has raised in the abstract but never resolved, and never applied to this artifact. It needs a human/legal call, not a retrieval result.

## Gaps and caveats

- **No downstream ablation.** Every headline number measures agreement with the Sonnet oracle, not effect on a trained model. The team itself flags this as the wrong ruler ([#6739](https://github.com/marin-community/marin/issues/6739#issuecomment-4833511462)).
- **The deployed model is the broken one.** Both successors (fast-transformer #6741, calibrated pooled #7040) are unmerged at the 2026-07-16 freeze; #6849/#6859/#6860 are open.
- **#5810's open/closed state is inferential.** The corpus stores no issue-state field; #5810 is never referenced as closed, and unlike siblings #5811/#5812 it was not listed as closed in any weekly summary.
- **PR #7040's body is not in the frozen corpus**, so its contents are known only via #6849 and the weekly summary.

---
<!--provenance-->
> *Data: marinmirror — 68026 chunks, built 45h ago · summaries through 2026-07-06_2026-07-12. Frozen eval corpus (2026-07-16); no refresh this run.*
>
> *Query: "What is the data-classifier model and how was it trained and evaluated? (See marin-community/marin#5810.) Can the model weights be shared?"*
>
> *Sub-queries: "how the LLM-oracle fastText quality classifier was trained — oracle, rubric, sampling, thresholding, hyperparameters, cost, artifacts" · "how it was evaluated — held-out oracle set, AUC/Spearman, dolma3 baseline, per-source spot checks, downstream use of high_score" · "can the weights be released — licensing, Anthropic terms, distillation constraints, HuggingFace vs internal GCS" · "parent epic #5360 and the broader Datakit quality-attribute program" · "latest state after Run 1 — retrains, failure-mode fixes, mixture adoption, weekly-summary narrative" · verification pass on all load-bearing numbers and ship/deploy status · adversarial skeptic pass on the weight-sharing dry result.*
