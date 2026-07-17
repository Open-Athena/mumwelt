# Target benchmarks versus development proxies

Marin's emerging practice is to keep **targets** and **proxies** separate. Targets are the behaviors ultimately wanted—knowledge/reasoning, math, coding, multilingual ability, instruction following, long-context use, and agentic task success—measured with downstream or execution-based evaluations. Proxies are cheaper, smoother signals used repeatedly during pretraining, mixture selection, scaling fits, and checkpoint triage. A proxy can guide development without being claimed as the target itself [#5005](https://github.com/marin-community/marin/issues/5005), [#5819](https://github.com/marin-community/marin/issues/5819).

## What counts as a target

The target side includes benchmark performance such as MMLU-style knowledge/MCQA, math reasoning, coding, long-context tasks, and especially execution-based agentic evaluations such as SWE-bench or Terminal-Bench. These are closer to the desired behavior, but they can be expensive, noisy, post-training-dependent, or vulnerable to contamination. The Artificial Analysis proxy project therefore reframed benchmark names into **competencies**: exact GPQA/HLE/AIME/MATH/MMLU-Pro/IFBench/LiveCodeBench/SciCode items should not be placed in routine PPL tracking; HLE in particular was explicitly excluded. Exact held-out targets remain held out [#5819](https://github.com/marin-community/marin/issues/5819).

## The main development-proxy portfolio

There is intentionally no single “model goodness” scalar. The checkpoint-confidence work calls for a portfolio that catches broad failures before expensive downstream evaluation [#5005](https://github.com/marin-community/marin/issues/5005):

- **Paloma and broad perplexity/BPB slices** for smooth pretraining progress and preregistered loss forecasts.
- **Uncheatable/fresh-data and raw-PPL sets** to reduce memorization and expose gaps in technical text, multilingual data, code, formatting, OCR, long context, and other under-covered formats.
- **Benchmark-aligned training/dev proxies**, such as normalized multiple-choice probabilities or held-out generated reasoning/text sets, rather than repeatedly optimizing on the final test items.
- **Scaling-law and checkpoint correlations** that ask whether early/base metrics predict the eventual downstream score.

These tools answer different questions. Paloma can be a stable loss ruler without being a sufficient measure of code or agent ability; raw-PPL gap analysis is a coverage/regression detector, not proof that fixing the gap causally improves the downstream target [#5005](https://github.com/marin-community/marin/issues/5005).

## What the experiments say about proxy validity

The strongest lesson is that **metric design matters more than the benchmark label**. On a 300M mixture matrix with 240 rows, ordinary MMLU accuracy had essentially no useful monotonic relation to BPB (Spearman −0.069), while normalized choice probability correlated strongly with accuracy (0.866). The interpretation in the thread is that accuracy itself is noisy; use the smoother, high-SNR proxy, but keep a held-out benchmark split for confirmation [#5247](https://github.com/marin-community/marin/issues/5247). Those values are specific to that 300M matrix, not universal constants.

Agentic proxies have been harder. The proposed cheap signals—loss on successful traces and the failed-minus-successful loss gap—had weak or non-monotonic evidence on MATH, and a matched-base OT-Agent attempt still suggested fine-tuning breaks straightforward log-probability comparability. Predicting “CORRECT” versus “INCORRECT” from complete traces was suggested, but related MATH variants had also worked poorly [#4389](https://github.com/marin-community/marin/issues/4389). The issue was later closed because the work moved elsewhere, not because a faithful proxy had been found.

A newer math study gives a narrower positive result: across ten SimpleRL-Zoo base models, base-model reasoning features can help predict **post-RL level**, but predicting the **RL gain** itself was weak (reported R² = 0.33). The close-out interpretation was that pre-RL pass@K is the main predictor of post-RL pass@K when other conditions are held fixed [#6096](https://github.com/marin-community/marin/issues/6096). That does not license carrying the result to coding/SWE-bench; the issue explicitly lists that as the next domain.

## Operational rule

Use proxies for rapid iteration, mixture ranking, regression detection, and preregistered forecasts. Validate proxy choice on many interventions and model scales, prefer smooth/high-SNR metrics, and hold tokenizer, prompt, data, and checkpoint conditions fixed. Then make release or capability claims only on the held-out target benchmarks. If proxy and target disagree, report the disagreement rather than promoting the proxy to the target.

The contamination guard is equally important: do not turn public target questions into frequently optimized PPL slices. Build competency-matched validation sets from separate or generated data, audit overlap, and reserve the real target tests for confirmation [#5819](https://github.com/marin-community/marin/issues/5819). In short: **Paloma/PPL and related diagnostics are development instruments; downstream and execution-based evaluations remain the scoreboards.**

### Source/gap notes

The frozen corpus documents a portfolio and several domain-specific studies, not a finalized canonical benchmark list. Proxy evidence is strongest for matched experimental panels and should not be generalized across base/instruct/post-RL models without a new correlation study.

---
<!--provenance-->
> *Data: marinmirror frozen 2026-07-16 — 68,026 chunks · summaries through 2026-07-06_2026-07-12 (no refresh).*
>
> *Query: “target benchmarks vs development proxies?”*
>
> *Sub-queries: “target competencies and held-out benchmarks” · “checkpoint-confidence PPL portfolio” · “Paloma versus downstream rankings” · “MMLU smooth proxy/SNR study” · “agentic soft proxies and negative results” · “Artificial Analysis competency surrogates and leak guard” · “base PPL/pass@K to post-RL performance” · “skeptic pass on domain transfer and causality”*
