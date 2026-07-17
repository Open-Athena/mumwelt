# Data ablations so far: mixes, classifiers, scales, and test sets

The corpus does not contain one clean, closed matrix covering every data experiment. The defensible picture is several generations of experiments with different rulers; combining their numbers as if they were one leaderboard would be misleading.

## 1. Early dataset and classifier baselines

- The original dataset baseline program called for matched 1B-model runs on FineWeb, FineWeb-Edu, Dolma 1.7, DCLM-Baseline, and SlimPajama ([#72](https://github.com/marin-community/marin/issues/72)). This is the broad “which source corpus?” comparison, separate from later classifier work.
- The DCLM replication trained fastText classifiers on OpenHermes-2.5 + ELI5 positives against web negatives. Because Marin did not have DCLM's exact ELI5/negative data, seed and mixture variance mattered: classifiers from different generated-data seeds agreed at roughly 90% F1, while an earlier Marin-vs-DCLM comparison was only 80% F1 ([#102](https://github.com/marin-community/marin/issues/102)). Those are **classifier agreement** numbers, not downstream LM quality.
- A later positive-data comparison trained Llama-1.4B language models for 10k steps / about 42B tokens after filtering with several bigram fastText classifiers. Reported examples include OpenHermes 200k + RefinedWeb 200k and ELI5 100k + OpenHermes 100k + RefinedWeb 200k; evaluation was bits-per-byte overall and on C4-en ([#164](https://github.com/marin-community/marin/issues/164#issuecomment-2465607666)).
- The FineWeb-Edu line explicitly distilled Llama-3 education annotations into fastText ([#390](https://github.com/marin-community/marin/issues/390)). The older Dolmino classifier proposal was later closed because the team had moved to MEDU, so it should not be counted as a completed head-to-head ([#605](https://github.com/marin-community/marin/issues/605)).

## 2. Cooldown / high-quality-mixture tests

The cooldown methodology converged on taking a common pretrained checkpoint and annealing alternative datasets, with “works” defined against a 70% DCLM / 30% FLAN control ([#722](https://github.com/marin-community/marin/issues/722#issuecomment-2702715304)). High-quality data generally improved Paloma macro loss relative to ordinary DCLM—with MegaWika called out as an exception—but that improvement did not consistently appear in downstream benchmarks ([#820](https://github.com/marin-community/marin/issues/820#issuecomment-2683490987)). This is why Paloma alone is not a sufficient ruler for deciding a production mix.

The more recent raw-web/extraction program used a common Nemotron 1.3B checkpoint for cooldown comparisons, and separately compared standard Resiliparse extraction with LLM extraction on domain data; its code experiment used 47 code-heavy URL patterns from 10 Common Crawl snapshots and evaluated HumanEval pass@1 ([#2351](https://github.com/marin-community/marin/issues/2351)). Its later scaling suite also used DCLM COREv2 and Uncheatable loss plots, while explicitly flagging token-count/quality mismatch as a fairness problem. Those extraction results are not interchangeable with the old 1B source-corpus runs.

## 3. Current curation/mix campaign

The current tracking issue is #6713: curate/tune the pretraining mixture and productionize the CoreWeave data pipeline ([#6713](https://github.com/marin-community/marin/issues/6713)). The frozen weekly primary-linked record shows five matched d1536, 2e21-FLOP runs comparing **DCLM, Nemotron, FineWeb-CC, Resiliparse-extracted web, and FineWeb-Edu**. At the recorded snapshot DCLM had the best aggregate Paloma bits-per-byte (~0.988), Resiliparse was strongest on code, and FineWeb-Edu was worst in aggregate (~1.441), largely from code. These are in-flight/snapshot observations, not a closed final leaderboard; the run family and scope are anchored by [#6713](https://github.com/marin-community/marin/issues/6713).

A contemporaneous focused-crawl A/B offers a cleaner closed comparison at the smallest rung: on v5p-8 d512, CC-SUPPLEMENTAL-2026-22 scored 4.712 Paloma macro loss versus 4.402 for CC-MAIN-2024-18, so the main crawl was better by 0.310 at that rung ([#6570](https://github.com/marin-community/marin/issues/6570#issuecomment-4776449650)). It is one rung and one test family, not proof about larger-scale downstream behavior.

## 4. Current classifier comparison and its ruler

The deployed v0 classifier is the Sonnet-oracle-distilled fastText system from #5810. A later apples-to-apples replacement study trained on the same 5,613 oracle-labeled documents and evaluated on the same 961-document holdout. Its pooled fast-transformer (`embed → pool@64 → transformer → regression head`) achieved AUC 0.875 and Spearman ρ 0.703 at 0.41M FLOPs/token, versus fastText's 0.846 / 0.641 ([#6739](https://github.com/marin-community/marin/issues/6739), [PR #6741](https://github.com/marin-community/marin/pull/6741)). NTP pretraining and weak-source-prior/FineWeb-Edu distillation did not beat that plateau ([#6739](https://github.com/marin-community/marin/issues/6739)).

That comparison is against the **Sonnet oracle**, not downstream LM quality, and #6739 warns that the oracle itself is education-leaning. Worse, production inspection found that the v0 fastText score largely sorted domain/modality/language rather than intrinsic within-domain quality and lacked an absolute quality anchor ([#6849](https://github.com/marin-community/marin/issues/6849)). Therefore these classifier AUC/ρ numbers cannot be treated as proof that filtering improves Paloma, CORE, or downstream tasks.

## What can safely be compared

| Experiment family | Compared | Scale / data | Evaluation |
|---|---|---|---|
| Prior-corpus baselines | FineWeb, FineWeb-Edu, Dolma 1.7, DCLM, SlimPajama | 1B models | published/baseline LM eval program (#72) |
| fastText positive-mix study | OpenHermes, ELI5, web-negative mixtures | Llama-1.4B; 10k steps; ~42B tokens | BPB, C4-en BPB (#164) |
| cooldown HQ study | candidate HQ sets vs 70% DCLM/30% FLAN | common checkpoint + anneal | Paloma plus downstream suite (#722/#820) |
| current source curation | DCLM, Nemotron, FineWeb-CC, Resiliparse, FineWeb-Edu | d1536; 2e21 FLOPs | Paloma aggregate/domains; snapshot (#6713) |
| current classifier model | fastText vs pooled fast-transformer | 5,613 train / 961 holdout oracle docs | oracle-label AUC and Spearman; **not downstream** (#6739) |

**Verified sources / gap note.** Re-opened #72, #102, #164, #390, #605, #722, #820, #2351, #6570, #6713, #6739, PR #6741, and #6849 or their exact frozen comments. The #6713 run-family result is not closed in the issue itself; it is retained only as a dated frozen-corpus snapshot and labeled accordingly. No claim is made that all early proposed baselines completed.

---
<!--provenance-->
> *Data: frozen marinmirror — 68026 chunks, built 2h ago · summaries through 2026-07-06_2026-07-12. No refresh (frozen runner only).*
>
> *Query: "data ablations so far — which mixes / classifiers compared, on what data sizes and test sets?"*
>
> *Sub-queries: "source-corpus baselines" · "fastText positive/negative mixtures" · "cooldown HQ mixes" · "current DCLM/Nemotron/FineWeb/Resiliparse campaign" · "oracle fastText vs fast-transformer" · "Paloma vs Uncheatable/CORE/downstream rulers" · "closed vs in-flight results". HyDE vector retrieval was used for current source and classifier facets.*
