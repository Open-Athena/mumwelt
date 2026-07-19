# Target benchmarks and development proxies

## The short version

Marin runs a **two-tier evaluation stack**, and the gap between the tiers is deliberate.

- **Target benchmarks** — what we ultimately want to score on — are anchored on the **Artificial Analysis Intelligence Index** for the base/pretrained model, plus a small fixed **MATH-500 / AIME24 / GSM8K** convention for post-training, and **SWE-bench Verified / Terminal-Bench** for agentic work.
- **Development proxies** — what we actually steer on day to day — are **Paloma and `uncheatable_eval` bits-per-byte**, a preregistered **Paloma macro-loss** number for hero runs, and a set of **smooth, high-SNR task proxies** (`choice_logprob_norm`, teacher-forced NLL, pass@k) that stand in for noisy accuracy.

The load-bearing rule connecting them is that **target benchmark items are explicitly forbidden as tracking data**. Proxies must be benchmark-*adjacent* and distributional, never benchmark-derived ([#5819](https://github.com/marin-community/marin/issues/5819#issuecomment-4483211940)).

An important caveat up front: **the corpus contains no numeric score target for any benchmark.** There is no "we aim for AA-II ≥ N" anywhere. The target list was still being assembled at the corpus edge.

---

## 1. Target benchmarks

### 1.1 Artificial Analysis Intelligence Index is the external ruler

AA-II is the framing used when the team reasons about where capability gains must come from. The clearest evidence is that data-mix decisions are argued *against* it — the close-out recommendation of the curated-vs-proportional study is literally titled "**Recommendation — best data mix for AA-II**," and concludes:

> "for AA-II, take **curated** as a weak default — but the honest result is that at this scale the mix is not where AA-II gains will come from."
> — [#6757](https://github.com/marin-community/marin/issues/6757#issuecomment-4847875950), 2026-06-30

Adding index support is also scoped into the eval-serving roadmap: "Map out adding support for the Artificial Analysis Intelligence Index" ([#5368](https://github.com/marin-community/marin/issues/5368), 2026-05-01).

### 1.2 The component list (the most authoritative enumeration)

The decontamination scoping issue enumerates the index as **16 evals, 8 configured and 8 skipped**, with exact item counts ([#5519](https://github.com/marin-community/marin/issues/5519#issuecomment-4464748971), 2026-05-16):

**Configured (8 of 16):** HLE (2,158 text-only, `cais/hle`), AA-Omniscience (6,000, `ArtificialAnalysis/AA-Omniscience-Public`), GPQA Diamond (198, `Idavidrein/gpqa`), MMLU-Pro (12,032, `TIGER-Lab/MMLU-Pro`), LiveCodeBench (~315), SciCode (288), IFBench (294, `allenai/IFBench_test`), GDPval (220, `openai/gdpval`).

**Skipped (8 of 16), with stated reasons:** AA-LCR (100 — "private/unreleased AA dataset"), MMMU Pro (1,730 — "visual reasoning — text decon irrelevant"), τ²-Bench Telecom (114 — "programmatically generated (no stable text)"), Terminal-Bench Hard (44), CritPt (70 — "physics challenges on custom grading server"), AIME 2025 (60 — "no canonical HF mirror at scrape time"), APEX-Agents-AA (452), Global-MMLU-Lite (~6,000).

### 1.3 Post-training: a deliberately narrow fixed suite

Post-training standardized on three benchmarks under a written convention, not a broad suite:

> "**Eval protocol** — fixed by `EVAL_CONVENTION.md`, run via evalchemy → the `stanford-crfm/lm-evaluation-harness` fork: 0-shot, `temperature=0.7`, **MATH-500** pass@1 (seed 42, boxed grader), **AIME24** pass@1 (mean ± se over 10 seeds 42–51), **gsm8k** exact-match. **Raw** = mean(MATH-500, AIME24, gsm8k-strict)"
> — [#6279](https://github.com/marin-community/marin/issues/6279#issuecomment-4674542747), 2026-06-10

For instruction following, **IFBench is the stated primary target and IFEval the continuity baseline**: "IFEval is the continuity baseline. IFBench is the primary target because it is the broader, more adversarial instruction-following benchmark… North star: improve IFBench and IFEval without creating a model that only overfits the IFBench verifier/spec" ([#5244](https://github.com/marin-community/marin/issues/5244), 2026-04-28).

For agentic work the live targets are **SWE-bench Verified** and **Terminal-Bench 2.0**, run through Harbor with the terminus-2 harness ([#6959](https://github.com/marin-community/marin/issues/6959), [#4420](https://github.com/marin-community/marin/issues/4420#issuecomment-4190111890)).

### 1.4 The end-of-training basket

Percy Liang described how the tiers combine at the end of training:

> "these will be the evals which we will add to our basket of our **200+ perplexity evals and standard generative evals that we will monitor at the end of training** (which includes pretraining + midtraining + SFT) before RL, in addition to using **hint-mediated forecasting of the harder generative evals to get smoother proxies**"
> — [#evals Discord](https://discord.com/channels/1354881461060243556/1356487738840318002/1523362280136380708), 2026-07-13

---

## 2. Development proxies

### 2.1 Why proxies at all: the signal-to-noise case

This is the empirical backbone of the whole approach. On a 242-row 300M swarm with 1,060 shared metrics, measured against a variable-subset noise baseline ([#5247](https://github.com/marin-community/marin/issues/5247#issuecomment-4382603100), 2026-05-04):

> "High-SNR smooth eval signals are mostly BPB/loss metrics."

| metric | variable-subset SNR |
|---|---:|
| `eval/uncheatable_eval/arxiv_physics/bpb` | 44.64 |
| `eval/uncheatable_eval/github_python/bpb` | 26.80 |
| `eval/paloma/falcon-refinedweb/bpb` | 25.58 |
| `eval/uncheatable_eval/bpb` | 25.04 |

> "By family, **all Paloma and uncheatable BPB/loss metrics clear SNR ≥ 2**. Task/lm-eval metrics are much noisier: **38/120 `lm_eval` metrics and 40/842 MMLU-subject metrics clear SNR ≥ 2**."

The resulting verdict on task metrics was **keep 12, downweight 89, report only 48**, with the blunt operating rule: "**Raw hard accuracy is too noisy for direct optimization at 300M; use it for validation/reporting.**" Will Held's sharper version: "correlation w/ accuracy is misleading as accuracy is itself noise; just use the smooth proxy w/ the highest SNR."

Corroborated independently in data mixing: "This time, we tried optimizing MMLU rather than a perplexity metric; unfortunately we have found it **too noisy at this swarm scale**" ([#data-mixing](https://discord.com/channels/1354881461060243556/1462895580064911522/1485030242413183130), 2026-03-21).

### 2.2 The smooth task proxies actually selected

Where a task survives, it is tracked through a *smoothed rendering*, not raw accuracy ([#5247](https://github.com/marin-community/marin/issues/5247#issuecomment-4382603100)):

| task | selected smooth proxy | SNR | Spearman vs hard acc |
|---|---|---:|---:|
| `swag_0shot` | `choice_logprob_norm` | 6.18 | 0.973 |
| `hellaswag_0shot` | `choice_logprob_norm` | 5.17 | 0.834 |
| `lambada_0shot` | `perplexity` | 4.65 | −0.897 |
| `piqa_5shot` | `choice_prob_norm` | 2.47 | 0.658 |
| `humaneval` | teacher-forced NLL on canonical solution | 9.21 | −0.491 |

Two cautions stated in the same thread: HumanEval's teacher-forced NLL is high-SNR but "pass@1 is near floor; treat it as useful evidence but **not a standalone optimization target yet**," and generation tasks "are the noisy long-pole; **never let them drive selection**." MMLU is only usable through the right transform — Spearman vs standard MMLU accuracy is −0.069 for BPB but **+0.866 for `choice_prob_norm`**.

### 2.3 Paloma macro loss: the headline hero-run proxy

Hero runs preregister a loss number before launch:

> "Preregistered loss target for stage 1 of the run (first 8T tokens): **2.269 paloma macro loss when evaled at seqlen 8192 on 1024 sequences per eval.**"
> — ClassicLarry, [#6044](https://github.com/marin-community/marin/issues/6044#issuecomment-4820008980), 2026-06-27 — **this is a target, not a result**

The first intermediate cooldown at ~2.1T tokens landed at **2.2772 Paloma macro loss / 0.8242 BPB**, a 0.109 (−4.6%) drop across ~211B cooldown tokens (2026-07-06 weekly summary, reporting [#6811](https://github.com/marin-community/marin/issues/6811)). This is an **achieved** number, but it is explicitly **not apples-to-apples** with the 2.269 target: the target is measured at seqlen 8,192 before the final LR cooldown, whereas the 2T cut fully decays LR, switches to the phase-2 mix, and evaluates at 8× context (65,536). It is a strongly favorable early signal that does **not** retire the 8T preregistration. Marin's lowest-ever loss for reference is 2.202, from the 32B model.

*Provenance caveat:* the 2.2772 figure is carried by the weekly summary; the primary comment stating it is not indexed in this frozen corpus. The 2.269 preregistration is primary-sourced.

### 2.4 The perplexity-gap registry

The proxy suite is assembled in `perplexity_gap_registry.py`, which defines four bundles — `base_raw` (Paloma + Uncheatable English/code), `multilingual_raw` (FineWeb2 top-50 + Indic), `runnable_long_tail` (SVG-Stack, VerilogEval), and `bio_chem` — plus structured-table and formal-methods slice families ([#5819](https://github.com/marin-community/marin/issues/5819#issuecomment-4482092952), 2026-05-18). A long-context tier was designed with a 32K default and 64K opt-in over five slices (`pg19`, `govreport`, SCROLLS `qasper`/`narrative_qa`/`quality`) ([#5825](https://github.com/marin-community/marin/issues/5825#issuecomment-4483260810), 2026-05-21).

### 2.5 The AA-competency proxy program — and its "no benchmark-maxxing" rule

[#5819](https://github.com/marin-community/marin/issues/5819) is the central design document: build pretraining-side proxies for AA-measured *competencies* without importing AA's items. It began as a direct AA-eval → PPL-surrogate matrix, then **corrected course after a benchmark-leak audit**:

> "Constraint update: core tracking should avoid benchmark-maxxing. We should **not** add the actual AA component questions, or close public benchmark train/test questions, as core perplexity evals. That includes GPQA/HLE/AIME/MATH/MMLU-Pro/IFBench/LiveCodeBench/SciCode exact items, **even where a train split exists**."
> — dlwh, [#5819](https://github.com/marin-community/marin/issues/5819#issuecomment-4483211940), 2026-05-18

The approved substitution pattern is distributional: Terminal-Bench Hard → "CI logs, shell transcripts, manpages, failing-test-to-patch traces, unified diffs"; τ²-Bench → "synthetic/helpdesk/CRM/service transcripts… tool-call logs from non-benchmark sources"; LiveCodeBench → "Stack v2, GitHub issue→patch, docstring→implementation"; AIME/math → "generated arithmetic/algebra/competition-style problems from templates."

Only small **canaries** from train splits survive in core tracking: `lm_eval/mmlu_auxiliary_train` for broad MCQA and `lm_eval/gsm8k_train` for grade-school math format — "with modest weight and clear labeling."

The honest self-assessment in the same thread:

> "current Marin PPL is useful as a broad data-fit dashboard, but **it is not yet a decent AA surrogate**."

The thread was later narrowed further (2026-05-23): it now claims only to find proxies for AA-measured competencies, "not claiming every AA behavior has a faithful PPL surrogate," and **truthfulness, hallucination robustness, and abstention are explicitly deferred** as post-training or non-PPL behaviors. Per-metric `ppl_fidelity` tagging was introduced because fidelity varies sharply — high for factual recall, "**very low**" for abstention/calibration, where "raw PPL on an 'I don't know' string is dominated by token frequency."

Work was split into competency sub-issues **#5823–#5829** (math, instruction-following, long-context, factuality/abstention, professional-document, MCQA, scientific reasoning).

### 2.6 Emerging proxies: pass@k and forecasting

**pass@k** is being adopted as a proxy modality that reveals capability masked by formatting failure — pass@{1,8,32,128} at N=128/problem showed "base models aren't empty — they're **format-floored**" ([#6279](https://github.com/marin-community/marin/issues/6279#issuecomment-4747832779)). The July intermediate-cooldown hero run states its acceptance criterion as "**pass@256** on X Y Z evals" ([#6811](https://github.com/marin-community/marin/issues/6811), 2026-07-01) — note the eval names are still a literal placeholder. pass@K forecasting has even influenced tokenizer choice, since matching tokenizers make the method easier (Weekly Digest, 2026-07-14).

### 2.7 Scaling ladders as the decision proxy

Mix and architecture decisions are made on iso-FLOP scaling ladders (d512–d1280, 3e17–3e19 FLOPs) and reported as **compute-equivalent speedup** with lack-of-fit confidence intervals, since each speedup is an extrapolation through a 4-point scaling fit ([#6757](https://github.com/marin-community/marin/issues/6757#issuecomment-4847875950)). A 111-eval side-by-side speedup table backs the current 10T mix ([#data-mixing](https://discord.com/channels/1354881461060243556/1462895580064911522/1521595789783076985), 2026-06-30).

**Proxies do not always transfer.** The SuperBPE tokenizer edge reversed at scale, and the curation win proved not Pareto-dominant (2026-07-06 weekly summary) — which is precisely why decisions are re-checked on a ladder rather than a single small point.

---

## 3. Why the separation is enforced: contamination

The proxy/target firewall is backed by an actual decontamination pipeline. `marin.datakit.decon` matches 13-word n-grams per paragraph and flags at ≥50% paragraph overlap, covering 600 benchmark families including MMLU, GSM8K, HumanEval, HellaSwag, GPQA, and MATH. It is **deliberately precision-favoring**: it flags ~0.01% of documents and intentionally ignores paragraphs under 13 words, after a whole-paragraph fallback was removed in #5656 for producing "~18% phantom contamination from trivial stubs" ([#6852](https://github.com/marin-community/marin/issues/6852), 2026-07-02). Synthetic-injection recall is "54% verbatim, 100% on prefix/suffix variants" ([#5519](https://github.com/marin-community/marin/issues/5519), 2026-05-18).

Real contamination was found and is not hypothetical: an at-scale MMLU run over 45,096,087 records flagged 2,605 documents, with AP test-prep books embedded verbatim — one document producing 514 n-gram matches at `max_overlap=1.0` ([#5519](https://github.com/marin-community/marin/issues/5519), ravwojdyla-agent, 2026-05-12).

---

## 4. Harness stack (what runs what)

| Layer | Harness | Role |
|---|---|---|
| Pretraining | lm-evaluation-harness | logprob + generation evals; Marin is migrating off its fork onto upstream main |
| Reasoning / post-training | `marin-community/evalchemy` | MATH500, AIME24/25/26, GPQA-D, HLE, LiveCodeBench v5/v6 |
| Agentic | `marin-community/harbor` | SWE-bench Verified, Terminal-Bench via terminus-2 |

Agentic evals reached **TPU parity** in July: an 8B agentic-SWE model on SWE-bench-Verified (random-100, N=300) ran end-to-end on a single preemptible **v6e-4** ([#6958](https://github.com/marin-community/marin/issues/6958), 2026-07-05). Automation of the checkpoint→leaderboard path is specified but **not shipped** ([#6503](https://github.com/marin-community/marin/issues/6503), 2026-06-18).

Where competitor comparisons are made, the discipline is **same-harness**, not published numbers:

> "we ran Llama-3.x and Gemma-2 instruct models — pre-reasoning, non-thinking peers — under the **identical `EVAL_CONVENTION` suite**, clamped to Delphi's exact 4k boxed-grader budget… **Running them under Delphi's exact protocol removes every harness confound.**"
> — [#6279](https://github.com/marin-community/marin/issues/6279#issuecomment-4750276628), 2026-06-19

And across frameworks, only deltas count: "**absolute numbers across harnesses aren't comparable — deltas are**" ([#6915](https://github.com/marin-community/marin/issues/6915), 2026-07-03).

---

## 5. Where we actually stand against the targets

Measured results, labeled as such:

- **Delphi-1e22 (9.7B)**, after midtraining + SFT + RL on MarinSkyRL: MATH500 **1.6 → 53.4**, AIME24 **0 → 6.6**, GSM8K **11.0 → 68.6** ([Discord close-out](https://discord.com/channels/1354881461060243556/1521285492602048542/1521288546697085150), 2026-06-29). Every large RL run "improved substantially… but they all also regressed on at least one of the benchmarks."
- **Marin-8B-Instruct on Terminal-Bench 2.0: 0/89 = 0%** ([#4420](https://github.com/marin-community/marin/issues/4420#issuecomment-4190111890), 2026-04-06).
- **Marin 8B SFT vs Qwen** on AIME2024/2025: 20.0%/13.3% vs Qwen2.5-7B-Instruct's 53.3%/53.3% — "Marin results currently lag behind Qwen"; issue auto-closed unresolved ([#2199](https://github.com/marin-community/marin/issues/2199)).

Internal read, May 2026: "We did SFT our 8B model and released an instruct version, which wasn't anything to write home about" (Kevin Xiang Li); "I think we need to really improve our basic mid-training + SFT before doing more advanced post-training" (Percy Liang).

---

## 6. Conflicts, gaps, and cautions

**Unresolved conflicts**

1. **Index version.** [#5519](https://github.com/marin-community/marin/issues/5519#issuecomment-4464748971) scopes against **v4.0** (2026-05-16); [#5819](https://github.com/marin-community/marin/issues/5819#issuecomment-4482092952) uses **v4.0.4** (2026-05-18). No comment pins a version.
2. **Index membership.** #5519 treats MMLU-Pro and AIME 2025 as in-scope index members; dlwh explicitly disagrees — "AA lists MMLU-Pro as an *additional* text eval rather than a main Intelligence Index component" ([#5819](https://github.com/marin-community/marin/issues/5819#issuecomment-4482686586)).
3. **Taxonomy.** A 2026-05-21 review argued the competency list is "roughly 60% capability and 40% surface mixed together," proposing a (capability × domain × surface) matrix. Partially accepted; not fully refactored.

**Gaps in this corpus**

- **No numeric score targets exist.** No AA-II threshold, no per-benchmark bar, no competitor-parity number. Any claim of the form "Marin targets AA-II ≥ N" is unsupported here.
- **[#7090](https://github.com/marin-community/marin/issues/7090)** — the July core-team eval wishlist, the most current statement of which evals we care about — **is not indexed**. This is the single largest gap.
- dlwh answered this exact question in Discord (2026-06-12) with a pointer titled "**Marin target metrics**"; its contents are not in the corpus.
- The July hero-run eval issues are stubs at freeze: [#6703](https://github.com/marin-community/marin/issues/6703) ("Which evals will we run") is one line; [#6811](https://github.com/marin-community/marin/issues/6811)'s criteria still read "X Y Z evals."
- `text_comparison_core` — the proposed core AA-proxy bundle — has **no evidence of merging to main**; #5819's program is design-plus-subissues, with slice families filed separately (#5863 FEVER, #6067 format-sensitivity, #5096 paraphrase robustness, #5618 surface-form probes).
- No 111-eval win/loss tally; the mix's own verdict was still open, with early 67B-A2B gains attributed "to Larry's architecture changes rather than the data mix."
- **τ²-Bench has never been run** — it appears only as a competency axis rated "Insufficient." No LiveCodeBench, HumanEval, or SWE-bench post-training score exists for a *Marin* model; the #6959 SWE-bench numbers are on a GLM-4.7-swesmith base.
- No IFEval/IFBench number for any Marin model — #5244's results are on Tülu-3-8B-DPO.

**Terminology correction:** "same ruler" is not Marin's phrase; the idiom in use is **"same-harness"** / "identical `EVAL_CONVENTION` suite." Relatedly, the RL work is on **Delphi-1e22 ≈ 9.7B**, not a "Delphi-25B," and true **Zero-RL (RL with no SFT) was planned but never reported** — do not attribute Zero-RL results to Marin.

---
<!--provenance-->
> *Data: marinmirror — 86367 chunks, built 0h ago · summaries through 2026-07-06_2026-07-12 (frozen eval corpus `2026-07-18-chunked`; refresh disabled, none triggered).*
>
> *Query: "What are our target benchmarks and our development proxies?"*
>
> *Sub-queries: "Artificial Analysis Intelligence Index component evals as stated targets for the 2026 contender / hero runs" · "perplexity-gap suite, uncheatable_eval, Paloma, BPB — the pretraining proxy suite and #5819/#5254 conclusions" · "Paloma macro-loss preregistration: the 2.269 stage-1 target, the 2.277 cooldown, #6046 loss registration" · "downstream eval harnesses actually run on checkpoints — lm-eval-harness, evalchemy, harbor, the 111-eval verdict, which evals are decision-making vs noisy" · "small-scale / scaling-ladder proxies that must transfer to the hero run — #4550, iso-FLOP speedup, mixture surrogates, proxies that break at scale" · "post-training benchmark targets and their proxies — MATH-500/AIME/GSM8K, IFBench, SWE-bench, Terminal-Bench, same-harness comparisons"*
