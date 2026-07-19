# Target benchmarks and development proxies

**Short version.** Marin runs a two-tier evaluation system, and the split is deliberate and explicitly theorized. The **targets** are expensive, generative, near-floor-at-small-scale benchmarks — the Artificial-Analysis-style competency basket, the Open-LLM-Leaderboard-hard suite for released instruct models, and agentic benchmarks (SWE-bench Verified, Terminal-Bench). The **development proxies** are cheap, dense, high-SNR likelihood metrics — Paloma macro loss and bits-per-byte, the Uncheatable Eval slices, and a 200+-entry perplexity-gap suite — plus preregistered scaling-law loss targets and, on the post-training side, a small fixed MATH-500/AIME24/GSM8K screening set.

The important caveat up front: the target list is **not settled** as of the corpus edge (2026-07-16). [#6703 "Eval selection (and implementation)"](https://github.com/marin-community/marin/issues/6703) is a one-line open question ("Which evals will we run and are they ready to run in [#6503](https://github.com/marin-community/marin/issues/6503)"), and the core team was still assembling a list in July. Several claims below are best read as *the current working answer*, not a ratified spec.

---

## 1. Target benchmarks

### 1a. The Artificial Analysis competency frame (the organizing target)

The most complete statement of what Marin wants the model to be good at is [#5819 "Experiment: Find pretraining proxies for Artificial Analysis competencies"](https://github.com/marin-community/marin/issues/5819) (dlwh, 2026-05-18). It takes the [AA Intelligence Index v4.0.4 methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking) as the enumeration of target capabilities:

> "Artificial Analysis' text-only index emphasizes agents, coding, general knowledge, scientific reasoning, long-context use, professional work, and instruction following, with multilingual tracked separately."

The AA component tasks it maps against ([comment 4482697610](https://github.com/marin-community/marin/issues/5819#issuecomment-4482697610)) are: **GDPval-AA, tau2-Bench Telecom, Terminal-Bench Hard, SciCode, AA-LCR, AA-Omniscience, IFBench, HLE, GPQA Diamond, CritPt**; plus AA "extra text evals" outside the main index: **MMLU-Pro, MMLU, AIME 2025, LiveCodeBench, HLE text-only, Global-MMLU-Lite**.

**Fidelity guard — do not overstate this.** The corpus contains **no committed AA Index score target** and no named competitor model to beat on it. #5819 is careful that AA is a framing device for *competencies*, not a scoreboard to optimize; the issue is explicitly framed as "finding pretraining proxies for AA-measured competencies, not claiming every AA behavior has a faithful PPL surrogate" ([comment 4523472318](https://github.com/marin-community/marin/issues/5819#issuecomment-4523472318)). I searched for an explicit "beat model X on the AA index" commitment and did not find one.

### 1b. Headline benchmarks for a released instruct model

The clearest declared headline suite is from the Marin 8B Instruct release ([willheld, #evals, 2025-05-17](https://discord.com/channels/1354881461060243556/1356487738840318002/1373406790301257899)):

> "Our benchmarks come primarily from the Open LLM Leaderboard's hard evaluation set, which includes IFEval, BBH, MATH-Hard, GPQA, MuSR, and MMLU-Pro… We also incorporate key benchmarks from the OLMo 2 technical report for instruction tuned models, which includes GSM8K-CoT, MMLU, and AlpacaEval. We exclude DROP… For code generation, we use HumanEval"

Harness policy from the same message: EleutherAI lm-evaluation-harness with chat templates for everything except AlpacaEval (official implementation on vLLM). AlpacaEval was later effectively dropped as a headline number — [#1343](https://github.com/marin-community/marin/issues/1343#issuecomment-2932676539): "AlpacaEval was 24.83. This was much lower than we hoped, likely due to EOS issues. I think this is getting closed in favor of our internal RL." **No refreshed headline list for the next instruct release appears in the corpus.**

### 1c. Agentic targets

Run through the Marin Harbor fork ([#2536](https://github.com/marin-community/marin/issues/2536), backported to Python 3.11 for the cluster): **Terminal-Bench 2 (TB2), TB-Lite, and SWE-bench Verified (random-100)**. The OT-Agent 32K SFT reproduction in [#3896](https://github.com/marin-community/marin/issues/3896) reports against released reference numbers — SWE-bench 13/100 = 13.0% (released 14.0%), TB-Lite mean 12.0% over 3 trials (released 18.0%), TB2 7.9% (released ~8.1%). Marin-8B-Instruct's own TB2 baseline was **0/89 = 0%** ([#4420](https://github.com/marin-community/marin/issues/4420)) — these targets are genuinely out of reach for the current model, which is exactly why proxies matter.

### 1d. The current, in-flight target list (July 2026)

Benjamin Feuer, [#evals, 2026-07-10](https://discord.com/channels/1354881461060243556/1356487738840318002/1526324939743957022): "The Marin core team has started to put together a list of evals we are particularly interested in seeing as part of Marin development. You can check out the GH issue [here](https://github.com/marin-community/marin/issues/7090)." *(Note: #7090 itself is not ingested in this frozen corpus — I can cite the announcement but not the list's contents.)*

Percy Liang's framing of how that list fits the two-tier system is the single best summary of the whole architecture ([#evals, 2026-07-13](https://discord.com/channels/1354881461060243556/1356487738840318002/1526324939743957022)):

> "these will be the evals which we will add to our basket of our 200+ perplexity evals and standard generative evals that we will monitor at the end of training (which includes pretraining + midtraining + SFT) before RL, in addition to using hint-mediated forecasting of the harder generative evals to get smoother proxies?"

He also asked, unanswered in the corpus: "Also, do we think Harbor index is useful as a curated set to look at?"

### 1e. Post-training harness stack

`marin-community/evalchemy` (forked from `teetone/evalchemy`, lm-eval-harness rebased to upstream main) + `marin-community/harbor` — Benjamin Feuer, [2026-06-11](https://discord.com/channels/1354881461060243556/1356487738840318002/1514733965070827641): "this combined with `marin-community/harbor` should cover the evals we need for post-training (we may need to add a few benchmarks though)". Evalchemy's task set landed via [PR #2779](https://github.com/marin-community/marin/pull/2779) (AIME24/25, AMC23, HMMT, MATH500, HumanEval+, MBPP+, LiveCodeBench, GPQA Diamond). **Status: target, not achieved** — [#6863 "[Epic] July Eval tasks"](https://github.com/marin-community/marin/issues/6863) still lists as its DoD "Both Evalchemy and Harbor can be easily triggered from Marin on TPUs."

---

## 2. Development proxies

### 2a. Paloma macro loss — the primary pretraining ruler

Paloma macro-average loss (and its bits-per-byte form) is the metric hero runs are steered and scored on. It is "just the standard paloma macro loss as implemented… it's just the code run in exp1337" ([willheld, #evals, 2026-06-08](https://discord.com/channels/1354881461060243556/1356487738840318002/1513661253602508941)). Live example: the 2T intermediate cooldown of the 67B-A2B run closed at **2.2772 Paloma macro loss, bpb 0.8242** ([#6811](https://github.com/marin-community/marin/issues/6811)).

Known artifact, worth knowing: the `twitterAAE_HELM` split moves the wrong way. dlwh, same thread: "Tons of symbols and stuff. Weird separators… Not worth worrying about. Means you're fitting the training distribution better (and we need to change the training data distribution)."

### 2b. Preregistered loss targets — proxies as commitment devices

Marin fits a Chinchilla-form scaling law on a small-model ladder with the irreducible-loss asymptote **pinned by hand**, then publishes the predicted Paloma macro loss *before* the run. Pinning matters: with free `L∞`, "at 1e23 the prediction moved from 2.157 to 2.226 (≈0.07)"; with `L∞` fixed at 1.6, "the 1e23 prediction changes by only 0.013 across the four fits" ([#4447](https://github.com/marin-community/marin/issues/4447), [comment 4225593039](https://github.com/marin-community/marin/issues/4447#issuecomment-4225593039)).

| Run | Preregistered | Achieved |
|---|---|---|
| 1e23 MoE (d5120, 129B/16B) | 2.252 | **2.234** ([#4697](https://github.com/marin-community/marin/issues/4697#issuecomment-4498921338)) |
| 67B-A2B 10T, stage 1 (first 8T) | **2.269** | *not yet reached* ([#6044](https://github.com/marin-community/marin/issues/6044#issuecomment-4820008980)) |

The stage-1 registration is quoted exactly: *"Preregistered loss target for stage 1 of the run (first 8T tokens): 2.269 paloma macro loss when evaled at seqlen 8192 on 1024 sequences per eval."* The eval protocol is registered alongside the number, and five falsifiable assumptions with it (irreducible loss 1.4; z-loss added after the small runs; batch-size increase assumed to cancel that; only 3 extrapolation points vs 20+ for the prior run; 24% peak LR at stage-1 end). ClassicLarry states the purpose plainly: *"I anticipate we will be able to use the actual achieved loss to update our knowledge of training dynamics and which assumptions to revisit."* The process issue is [#6046](https://github.com/marin-community/marin/issues/6046).

**Do not report 2.269 as achieved.** The 2.2772 cooldown result is a *different, non-comparable* measurement — a 2T checkpoint, fully decayed LR, phase-2 mix, evaluated at 8× context (65,536) rather than seqlen 8,192 ([#6811](https://github.com/marin-community/marin/issues/6811)). It is a favorable early signal, not a scored target.

### 2c. Uncheatable Eval and the perplexity-gap suite

Alongside Paloma: **Uncheatable Eval** slices (`wikipedia_english`, `bbc_news`, `arxiv_physics`, `arxiv_computer_science`, `github_python`, `github_cpp`), FineWeb2 multilingual, Stack v2 code slices, structured tables (`totto`, `gittables`, `wikitablequestions`), formal methods, and LM-eval bridge canaries (`mmlu_auxiliary_train`, `gsm8k_train`) — inventory per [#5819](https://github.com/marin-community/marin/issues/5819) and the `perplexity_gap_registry.py` bundles `base_raw`, `multilingual_raw`, `runnable_long_tail`, `bio_chem`.

Code coverage was deliberately widened in [#5254](https://github.com/marin-community/marin/issues/5254) after finding "at least ~3/4 of the reachable uncovered tokens in the interesting middle of the tail are programming / API / config / build / framework tokens" — a `CODE_ECOSYSTEM` family of per-language Stack v2 held-out slices, **shipped as a v0 cut of 105 of 136 slices** after the materialization job stalled ([comment 4454871475](https://github.com/marin-community/marin/issues/5254#issuecomment-4454871475)), closed 2026-05-18 "pending us creating an IID validation set from our actual code."

**The two rulers disagree, and that is a live finding.** In the H100 validation-run thread, re-scoring the same runs on the shared Uncheatable cache *flipped the ranking* in datakit's favor, because Paloma scores code-and-math-heavy mixtures comparatively low ([#6716](https://github.com/marin-community/marin/issues/6716), week of 2026-07-06). Read mixture comparisons on both rulers.

### 2d. How a proxy earns the right to be used — SNR, not intuition

[#5247 "Identify useful set of evals"](https://github.com/marin-community/marin/issues/5247) measures signal-to-noise directly: 10 replicate noise-baseline rows at 300M against a 242-run × 1073-metric signal matrix ([comment 4368082302](https://github.com/marin-community/marin/issues/5247#issuecomment-4368082302)). Result: 43 metrics clear SNR ≥ 10, 76 clear ≥ 5, 148 clear ≥ 2 of 1060. The family breakdown is the punchline:

> "By family, all Paloma and uncheatable BPB/loss metrics clear SNR >= 2. Task/lm-eval metrics are much noisier: 38/120 `lm_eval` metrics and 40/842 MMLU-subject metrics clear SNR >= 2."

Top SNR: `uncheatable_eval/arxiv_physics/bpb` at 44.64, `paloma/falcon-refinedweb/bpb` at 25.58. The adopted rules from the same comment: *"Raw hard accuracy is too noisy for direct optimization at 300M; use it for validation/reporting"*, and — relayed from Will Held ([comment 4368119915](https://github.com/marin-community/marin/issues/5247#issuecomment-4368119915)) — *"correlation w/ accuracy is misleading as accuracy is itself noise; just use the smooth proxy w/ the highest SNR."*

Normalization is not a detail: on 300M rows (n=240), MMLU accuracy vs BPB Spearman was **−0.069**, vs `choice_logprob_norm` **−0.026**, but vs `choice_prob_norm` **+0.866** ([comment 4339766407](https://github.com/marin-community/marin/issues/5247#issuecomment-4339766407)).

The qualification criterion adopted for downstream prediction is decision-relevance, not fit quality — [#4550](https://github.com/marin-community/marin/issues/4550) motivates "**selection regret > raw R²**", and [#6665](https://github.com/marin-community/marin/issues/6665#issuecomment-4805081871) makes it operational: *"Require decision-relevant improvements: regret@1, lower-tail RMSE/optimism, and top-k predicted observed rows. Spearman-only gains are not enough to justify a mixture submission."* **No numeric pass threshold (regret ceiling, SNR floor) is ratified anywhere in the corpus** — SNR ≥ 2 / ≥ 5 appear as reporting bands, not gates.

### 2e. The data-mixture objective

[#5362 "Select Metric for Data Mix Optimization"](https://github.com/marin-community/marin/issues/5362) opens with "Needs to be high SNR." Factor analysis found a low-dimensional aggregate but over-optimized the easy factors: *"The factors that are more clearly explainable based on the data mix (e.g. math and coding capabilities) seem to end up a bit over-optimized because they are the easiest to optimize!"* The decision landed 2026-06-22 ([comment 4773354017](https://github.com/marin-community/marin/issues/5362#issuecomment-4773354017)): `L(theta) = \sum(hinge(task_loss, prop_loss-\eps)) + \sum_{code+math} task_loss + L_2(theta, theta_prop)` — a hinge against a proportional-reference baseline with unhinged linear terms for code and math.

### 2f. Smoothing hard generative evals — pass@k and tunably-easier tasks

Because target benchmarks floor out at development scale, [#4549](https://github.com/marin-community/marin/issues/4549) builds scaffolds "with tunable knobs that make the target task considerably easier", aiming to "predict the pass@k performance for extremely large k (e.g., k > 100k) without actually having to sample k times." The concrete instantiation is **Masked GSM8K** (`mask_fraction`, tasks `mask_00`…`mask_08`). On a 36-cell Delphi midtrain matrix ([comment 4583759889](https://github.com/marin-community/marin/issues/4549#issuecomment-4583759889)), `mask_00` pass@1 rises monotonically with compute — 0.6173 (3e20) → 0.6981 (1e21) → 0.8473 (1e22) — while pass@32 saturates. The write-up is scrupulous about what it is: `mask_00` is "solution-primed → this is answer-*extraction* with the CoT visible, **not** from-scratch GSM8K reasoning. Read as a **relative** signal at matched compute." This is what Percy meant by "hint-mediated forecasting."

### 2g. Post-training development proxies

During SFT/RL iteration the screening set is small and fixed, governed by a protocol document called `EVAL_CONVENTION`: **MATH-500 (1 seed), AIME24 (10 seed), GSM8K (strict/flex)** — used to rank 54 cells of the Delphi SFT grid ([#6279](https://github.com/marin-community/marin/issues/6279#issuecomment-4747832779)). Expensive characterization passes use pass@{1,8,32,128}, N=128, temp 0.7 ([comment 4754886989](https://github.com/marin-community/marin/issues/6279#issuecomment-4754886989)). A classic-NLP grid (MMLU, HellaSwag, ARC, PIQA, WinoGrande, OBQA, BoolQ, TruthfulQA, LAMBADA, TriviaQA, NQ-open, DROP) runs as a regression check ([comment 4754913165](https://github.com/marin-community/marin/issues/6279#issuecomment-4754913165)).

Two protocol traps documented there: GSM8K needs **flexible-extract** ("Strict-match floors to ~0 for every no-chat-template model"), and pass@1 in #6279 is the sampled unbiased estimator at temp 0.7, **not** greedy decoding.

---

## 3. The discipline connecting the two tiers

Three rules make the target/proxy split more than a convenience.

**No benchmark-maxxing.** The core rule from [#5819](https://github.com/marin-community/marin/issues/5819#issuecomment-4483211940): exact benchmark items from AA components — GDPval-AA, tau2-Bench, Terminal-Bench, SciCode, AA-LCR, AA-Omniscience, IFBench, HLE, GPQA Diamond, CritPt — plus MMLU-Pro, AIME, MATH-500, LiveCodeBench, HumanEval, MBPP exact tasks and near-duplicates, "should not be regular PPL tracking data. If exact benchmark-derived PPL is ever run, it should be opt-in diagnostic only, clearly labeled, and excluded from training-run core dashboards." Allowed as small, clearly-labeled canaries: `lm_eval/mmlu_auxiliary_train` and `lm_eval/gsm8k_train`. Proxies are built from *non-benchmark distributions that support the same competency* — CI logs and failing-test-to-patch traces for Terminal-Bench-like debugging, long public reports for AA-LCR-like retrieval, Wikipedia/Wikidata cloze for Omniscience-like factuality, generated constraint transformations for IFBench-like formatting.

**Fidelity is labeled, not assumed.** `ppl_fidelity` attaches "to a competency/metric/rendering claim, not to a dataset in isolation" — factual recall is high-fidelity under raw PPL; instruction following, abstention, long-context retrieval and agentic state tracking are low-fidelity and need paired/discrimination or non-PPL companion metrics. Truthfulness, hallucination robustness and abstention are **explicitly deferred** to post-training. Each tracked slice records a role: `headline`, `canary`, or `diagnostic`.

**A portfolio, not a scalar.** [#5005](https://github.com/marin-community/marin/issues/5005) (dlwh): "The goal is not to define one scalar for whether a base model is 'good'; it is to build a portfolio of evidence that catches broad failures before expensive post-training, RL, and agent evaluations." Its DoD requires validating agentic soft proxies against real SWE-bench/Terminal-Bench outcomes "and **label unvalidated proxies as such**."

---

## 4. Where this is weakest — gaps and disagreements

These are load-bearing and I'd rather flag them than paper over them.

- **No agentic proxy has been validated.** [#4389](https://github.com/marin-community/marin/issues/4389) tested trace log-likelihood proxies and rejected them. On the OT-agent leaderboard replication (top-25 Qwen3-8B finetunes on SWE-bench Verified, 50 candidate proxies): *"the results were not great (highest was ~0.3)"* rank correlation, and *"the best model actually got \*higher\* loss (1) on all 25 sets of trajectories than the worst model"* ([comment 4209804794](https://github.com/marin-community/marin/issues/4389#issuecomment-4209804794)). Masking to tool-calls-only or masking user text didn't help. On MATH, *"success-failure gap generally did worse than just loss"* ([comment 4183999166](https://github.com/marin-community/marin/issues/4389#issuecomment-4183999166)). [PR #4539](https://github.com/marin-community/marin/pull/4539) was auto-closed stale 2026-05-11; the issue closed 2026-06-10, folded into #4550. #5005's "validate agentic proxies" DoD item remains open.
- **A direct, unresolved disagreement about whether PPL leads accuracy at all.** dlwh: "I'm hoping that agentic RL will work the same as MCQA, where you see smoothly improving PPL long before you get any gains in accuracy for e.g. MMLU." rohithck: "this doesn't seem to be the case so far" — and sharper: *"even among BPB metrics sometimes BPB on task traces is actually less correlated with performance than BPB on some more generic dataset (e.g., arxiv-cs and arxiv-math is more correlated with MATH than BPB on correct rollouts)."* dlwh separately raises a **temperature confound**: deep cooldowns shift PPL without changing token ordering.
- **The strongest positive proxy result partly self-refutes.** [#6096](https://github.com/marin-community/marin/issues/6096) found base-model BPB on OpenMathReasoning traces predicts post-SimpleRL accuracy at r = −0.89 to −0.92 (n=10) — but the [close-out](https://github.com/marin-community/marin/issues/6096#issuecomment-4836443867) notes RL adds a near-constant +15.8 ± 4.1 points to everyone, so base and post-RL accuracy are 0.97 correlated anyway, and *"Predicting the **RL gain** (post − base) **fails: R² = 0.33**."* It forecasts capability *level*, not who benefits from RL.
- **Much of the reliable-scaling program is proposed, not executed.** [#4548](https://github.com/marin-community/marin/issues/4548) and [#4551](https://github.com/marin-community/marin/issues/4551) both went stale 2026-07-01 with no posted results.
- **Target lists are unfinished.** [#6703](https://github.com/marin-community/marin/issues/6703) is an open one-liner; [#6054 "How can we inform pre-training decisions and which evals we're hill-climbing with post-training goals / evals"](https://github.com/marin-community/marin/issues/6054) is a bare title with no body or comments; and the promotion criterion for the intermediate cooldown checkpoint in [#6811](https://github.com/marin-community/marin/issues/6811) is *literally* unfilled — "Criteria is pass@256 on **X Y Z evals**".

---
<!--provenance-->
> *Data: marinmirror — 68026 chunks, built 45h ago · summaries: 15 weeks, latest 2026-07-06_2026-07-12. Frozen eval corpus (MARIN_EVAL_FREEZE=2026-07-16); no refresh performed.*
>
> *Query: "What are our target benchmarks and our development proxies?"*
>
> *Sub-queries: "target/headline benchmarks the 2026 contender aims at (Artificial Analysis Intelligence Index, named competitors)" · "loss-based development proxies — Paloma macro loss, bits-per-byte, Uncheatable Eval, perplexity-gap suite, and their disagreements" · "soft proxies for agentic/coding benchmarks in data-mixture studies (#4389, #4539, #5254)" · "pretraining proxies for Artificial Analysis competencies (#5819) and the competency taxonomy" · "how a metric qualifies as a trustworthy proxy — reliable scaling, SNR, selection regret, preregistered loss targets (#4550, #5247, #6044)" · "post-training benchmarks and harnesses — evalchemy, harbor, EVAL_CONVENTION, pass@k gating"*
