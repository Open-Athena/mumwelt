# Target benchmarks and development proxies

Marin's evaluation stack is **two-tiered and has evolved over time**: a set of external
**target benchmarks** we ultimately want to move, and a much larger tier of cheap,
benchmark-leak-safe **development proxies** — perplexity / bits-per-byte (BPB) slices and
pre-registered scaling-law targets — that are read *during* pretraining/midtraining, long
before the target benchmarks are meaningful. Crucially, **what Marin actually executes in
its eval pipeline is not the same as the aspirational target list**, and the objectives
have shifted across the project's life. Answering this well requires separating four
things: (1) what actually ran on training jobs, (2) the pre-registered scaling proxies,
(3) the current live mixture-selection objective, and (4) the external target frameworks
we are trying to approximate.

Ownership note: the evals area is led/populated primarily by **willheld (William Held)**
(#evals, Mega-Evals, 32B), **Calvin-Xu** (OLMix / Table-9 scaling), **ClassicLarry / Larry
Dial** (67B-A2B pre-registration), **Helw150 + kothasuhas + RohithKuditipudi**
(downstream-scaling proxies / Delphi), **nikil-ravi** (`default_eval` / CORE), and
**dlwh / Kevin Xiang Li / ihodes** across infra, the AA-surrogate design, and the hero
runs. Attributions below follow the corpus.

---

## 1. The load-bearing during-training eval tier (what actually ran for most of training)

For most of Marin's training history the canonical during-training evals settled into three
co-existing layers wired into `default_train` / `default_eval` / `default_anneal`:

**(a) Paloma PPL / validation-loss sets — always-on.** The first real during-training eval.
`#448` "Add default ppl evals to default training" (dlwh, 2024-10-18) was closed by
`#482`, which wired **Paloma** in as a default validation set
([#448](https://github.com/marin-community/marin/issues/448),
[#482](https://github.com/marin-community/marin/pull/482)). Paloma macro/micro loss remains
the headline pretraining number to this day.

**(b) DCLM-CORE-style MCQ tasks — the canonical `default_eval`.** Requested in `#553` "Add
CORE eval from DCLM paper" (nikil-ravi, arxiv 2406.11794), delivered by
**[PR #574](https://github.com/marin-community/marin/pull/574)** "Add default_eval with CORE
tasks" (nikil-ravi, 2024-11-18): makes evals a `list[EvalTaskConfig]` and "Adds tasks from
the CORE benchmark in the DCLM paper. **It has 14/22 — some of them require remote code, use
different numbers of shots** etc." (the 8 omitted tasks are not enumerated in-corpus).
A homegrown `internal_eval` framework (`#468`/`#594`) was the abandoned predecessor: its
numbers "ended up being garbage somehow" and the code was removed, superseded by
lm-eval-harness ([#553](https://github.com/marin-community/marin/issues/553),
[#594](https://github.com/marin-community/marin/issues/594)). The default_eval task set (from
W&B run names) was ~12 tasks: `agieval_lsat_ar, arc_easy, arc_challenge, boolq,
commonsense_qa, copa, hellaswag, lambada_openai, openbookqa, piqa, winograd, winogrande`
(with a `bigbench_language_identification_multiple_choice` variant).

**(c) MMLU — opt-in, later pushed toward anneal.** `#658`/`#672`: "By default, MMLU
evaluation is not run — this adds an option to run 0-shot and 5-shot MMLU for larger scale
runs" ([#658](https://github.com/marin-community/marin/issues/658),
[#672](https://github.com/marin-community/marin/pull/672)). Later proposed as the *sole*
`default_anneal` eval to dodge CORE-download flakiness that was killing anneal runs (merge
status a gap in-corpus).

**Harness lineage — the lm-eval-harness fork with "soft metrics."** CORE was routed through
a lightly-modified fork of EleutherAI's lm-evaluation-harness, not internal_eval.
**[PR #817](https://github.com/marin-community/marin/pull/817)** (nikil-ravi, 2025-02-18)
"Switch to lm-eval-harness fork with soft metrics" made
`stanford-crfm/lm-evaluation-harness` the default "and will also log the soft metrics …
It currently does not fully work with MMLU due to a bug." "Soft metrics" = **bits-per-byte
(BPB) + per-choice logprob metrics**, logged alongside accuracies. The fork exists mainly to
"make lm_eval_harness not fully dependent on Torch," and the stated rationale is "it's the
closest thing we have to a standard, particularly for non-frontier models." Debug noise:
`#781` (MMLU on Levanter), `#1125` (MMLU soft-metrics tracking). By early 2026 willheld had
grown this to **"71 evals in lm-eval-harness for pretraining"** with a 100-task target
([2026-01-21](https://discord.com/channels/1354881461060243556/1356487738840318002/1463620749356175519)),
later **"181 evals … as our candidate set"**
([2026-01-26](https://discord.com/channels/1354881461060243556/1356487738840318002/1465481663411912824)).
The fork is being retired toward EleutherAI mainline mid-2026 ("Seems like we can move off our
fork"), so single-harness attribution is a snapshot.

**"Mega-Evals" — a lead, but never the default set.**
**[PR #2663](https://github.com/marin-community/marin/pull/2663)** "Mega-Evals" (Helw150,
2026-02-05) "Runs ~200 Evals using Lm-Eval-Harness" — but it was flagged for a
cleanup-masks-failure bug and **auto-closed for inactivity (2026-05-04, re-closed
2026-06-12); it never merged.** Treat "~200 Mega-Evals" as an aspirational harness, not the
load-bearing default_eval. (Successor hardening: [PR #6362](https://github.com/marin-community/marin/pull/6362)
adds DCLM Core v2 scoring helpers, Calvin-Xu, 2026-06.)

---

## 2. Pre-registered Paloma macro-loss scaling targets (the core development-proxy mechanism)

The distinctive Marin development proxy is not a benchmark score at all — it is a
**pre-registered Paloma macro-average-loss vs FLOPs scaling law**, of functional form
`macro(C) = L∞ + A·C^(-α)` (irreducible asymptote + power law in compute), registered
*before* a hero/ladder run finishes so the run can be judged against a "smooth scaling"
acceptance bar.

**Delphi scaling suite ([#1337](https://github.com/marin-community/marin/issues/1337)).**
"Delphi: Create a modern scaling suite ('modernized Pythia')" (percyliang, 2025-05-22): a
reusable open suite of checkpoints going beyond Pythia (up to ~32–70B, more tokens,
Nemotron-CC/CommonPile rather than The Pile). "getting the best accuracy isn't a goal." The
**pre-registration (Helw150, 2026-03-03, explicitly "for @ihodes")**:
> "We should expect the following Paloma Macro Losses at the end of these Nemotron runs **if
> we have smooth scaling** (assuming a loss asymptote of ~1.9 from @kothasuhas work): **1e21:
> 2.75, 1e22: 2.55, 1e23: 2.40**"
([#1337 comment-3992593231](https://github.com/marin-community/marin/issues/1337#issuecomment-3992593231)).

Achieved-vs-target (prefer close-out; all Delphi-Nemotron ladder):
| Budget | Pre-registered target (1.9 asymptote) | Achieved | Verdict |
|---|---|---|---|
| 1e21 | 2.75 (projected) | **2.7581** (measured) | within forecast (~.0002 off) |
| 1e22 | 2.55 (projected) | **2.53** (measured) | **beat** forecast despite spikes ([comment-4016705345](https://github.com/marin-community/marin/issues/1337#issuecomment-4016705345)) |
| 1e23 | 2.40 (projected) | **no final number in corpus** | run was spiky, required optimizer fix; last comment (2026-04-24) shows downstream forecasts explicitly "Not pre-registered" |

The 1e23 run underwent a **stability crisis** (untuned weight decay → norm growth →
divergence), fixed with AdamH + WhenWen's norm-constrained optimizers; it was allowed to
finish "to update our priors about the dangers of spiking." The 1.9 asymptote is itself
**contested** — the 1e22 over-performance is read as evidence "the asymptote at 1.9 is
wrong." Engineering that operationalized this lives in `#3292` "Delphi Scaling Setup"
(modular isoflop recipes: `c_adamc`, `completed_adamh`).

**Sibling MoE isoflop ladder — cleaner close-out, same methodology.**
[#4447](https://github.com/marin-community/marin/issues/4447) fits the same
`L∞+A·C^(-α)` law via leave-future-out CV;
**[#4697](https://github.com/marin-community/marin/issues/4697)** the 1e23 MoE (d5120, 129B
total / 16B active, ~1T tokens, v4-1024→v4-2048) **predicted 2.252, final Paloma macro loss
2.234 — <1% off pre-registration** ([#4697 comment-4498921338](https://github.com/marin-community/marin/issues/4697#issuecomment-4498921338)).
**Do not conflate ladders**: Delphi-Nemotron 1e21 = 2.7581, MoE isoflop 1e21 = 2.599, MoE
1e23 = 2.234 — separate ladders/datasets/recipes sharing the same pre-registration discipline.

**The #downstream-scaling "proxy evals project" (stem
[#4550](https://github.com/marin-community/marin/issues/4550)).** "Reliable scaling for
downstream evals/post-training" (RohithKuditipudi, 2026-04-09; teetone, ahmeda14960,
AlienKevin): predict how an expensive run performs on downstream benchmarks from cheaper
runs. Off-policy trajectory log-likelihood proved "a somewhat noisy indicator," so they
pivoted to **predicting task performance across a scaling ladder (holding out the largest
model)** using the Marin isoflops + Delphi ladders, plus token-scaling-down (#4548) and
pass@k for large k (#4549). Example: fitting **GSM8K loss/BPB for the Delphi models**,
extrapolating <1e21→1e23 (off by ~0.1 nats, under-predicting). The precursor
**[#4389](https://github.com/marin-community/marin/issues/4389)** "soft proxy for agentic
benchmarks" returned a **largely negative result** (no monotonic proxy→downstream
relationship on MATH across ~15 models; success-failure gap did worse than plain validation
loss) and was **closed 2026-06-10, superseded by #4550.**

---

## 3. Current live mixture/DSP-selection objective: OLMoBaseEval Easy "Table-9" BPB macro

The objective that actually drives **data-mixture and DSP (domain-selection-policy)
selection at freeze** is the **OLMoBaseEval Easy Table-9 unweighted 51-component BPB macro**
(`table9_macro_bpb`), driven by **Calvin-Xu** (2026-06-23→07-06). This is the paper-faithful
OLMix Table-9 target: the unweighted mean of 51 BPB components (Minerva/ARC/MT-MBPP/Basic
Skills leaves standalone; MMLU collapsed to 4 OLMix size-weighted buckets; top-level
QA/code/math aggregates excluded), from 104 scored tasks / 88,592 instances
([#6611 comment-4804956805](https://github.com/marin-community/marin/issues/6611#issuecomment-4804956805)).

**How it is computed:**
[PR #6726](https://github.com/marin-community/marin/pull/6726) "Add Marin-native OLMoBaseEval
Easy Table 9 BPB evaluator" (Calvin-Xu, 2026-06-27) scores the suite **natively in Levanter
on TPU** (region-locally, replacing the "convoluted external Stanford-SC eval path"):
continuation-masked **fp32 log-probs over a UTF-8-byte denominator**, unweighted
per-instance → per-component → macro. Parity vs the SC oracle on a us-east5 v6e-8: macro
absolute diff **1.9e-8**, all 51 components within 1e-3. A silent bf16-drift bug in the Iris
executor path was caught and fixed in review (`olmo_base_eval_step` now defaults `f32`,
matmul precision "highest").

**It superseded the Uncheatable-Eval-only objective — but only partly.** The prior objective
was `eval/uncheatable_eval/bpb` (single-target "Uncheatable Eval"), against which the OLMix
baseline `olmix_d001_kl005_cap4` (two-phase, Huber δ0.01, KL 0.05, cap 4) was selected in
[#6608](https://github.com/marin-community/marin/issues/6608). The explicit supersession
event (2026-06-26): the older uncheatable OLMix scaling parent "was stopped because it was
**superseded by the current Table-9 macro objective** and was repeatedly preempting the
Table-9 optimized scaling rows"
([#6608 comment-4811794926](https://github.com/marin-community/marin/issues/6608#issuecomment-4811794926)).
**Nuance:** the supersession is scoped, not total — at freeze, `eval/uncheatable_eval/bpb`
**remained a live scaling-validation target** (`dsp_effexp_kl01`/`olmix_d001_kl005_cap4`),
coexisting with the new primary fit objective
([#6602 comment-4804956790](https://github.com/marin-community/marin/issues/6602#issuecomment-4804956790)).
There was also an interim generation between them: `olmo_base_easy_top3_macro_bpb`
(equal-weight mean of the three top-level aggregates).

**The proxy in action.** Effective-exposure DSP fit directly on `table9_macro_bpb` on a
280-row 300M panel: **best model = effective-exposure DSP, `linear_reg=1e-4`, OOF Spearman
0.918 / OOF RMSE 0.01284**, beating both OLMix variants (paper-faithful `single_tied`
0.380; Marin-extension `two_phase_adapted` 0.891 — the latter must be reported separately as
it is *not* the OLMix paper baseline). The Table-9-optimized scaling ladder (Iris) at freeze:
both 3e18 and both 2e19 arms **SUCCEEDED**; **3e20/1e21 still running** (incomplete). The
objective is being *refined not frozen*: [#6665](https://github.com/marin-community/marin/issues/6665)
rejected reliability-weighting the aggregate as an estimand shift ("Keep the headline
estimand fixed: the unweighted 51-component Table-9 macro BPB").

---

## 4. Perplexity-gap / AA-competency PPL surrogate suite (the May-2026 proxy redesign)

Layered on top of the above is the **AA-competency perplexity-gap suite**
([#5819](https://github.com/marin-community/marin/issues/5819), dlwh, 2026-05-18): a
competency-organized portfolio of raw-text PPL/BPB slices, consolidated into
`experiments/evals/model_perplexity_gap_suite.py`
([PR #5836](https://github.com/marin-community/marin/pull/5836), which explicitly composes
raw-PPL providers "for Marin 32B vs Qwen3 32B scoring").

**Core design rule — avoid "benchmark-maxxing."** The proxies deliberately do **not** contain
the actual target-benchmark items: "We should not add the actual AA component questions …
That includes GPQA/HLE/AIME/MATH/MMLU-Pro/IFBench/LiveCodeBench/SciCode exact items, even
where a train split exists." Each target competency gets a *distributional surrogate* tagged
with a **`ppl_fidelity`** rating (faithful for factual recall / sci-prose familiarity; weak
for abstention, agentic state-tracking, long-context retrieval, which are explicitly deferred
to post-training / non-PPL evaluation). Coverage bundles: Paloma slices, Uncheatable Eval
(arxiv/wikipedia/bbc/github), FineWeb2 multilingual, Stack v2 / formal-methods / bio-chem /
structured-table slices, and an LM-eval bridge (with `mmlu_auxiliary_train`, `gsm8k_train`
allowed only as small canaries). Competency gaps became subissues #5823–#5829.
Proxy quality is **still being validated**: the success criterion is rank-correlation between
a PPL proxy and downstream evals ([#6712](https://github.com/marin-community/marin/issues/6712)),
and early reads (#5005) reported low SNR on many slices.

---

## 5. Target benchmarks (what we ultimately want to move)

**Artificial Analysis (AA) Intelligence Index v4.0.4 — the headline target framework, but
mostly aspirational competencies, not executed evals.** From
[#5819 comment-4482092952](https://github.com/marin-community/marin/issues/5819#issuecomment-4482092952)
(dlwh), the v4.0.4 index components are: **GDPval-AA · tau2-Bench Telecom · Terminal-Bench
Hard · SciCode · AA-LCR · AA-Omniscience · IFBench · HLE (text-only) · GPQA Diamond ·
CritPt**; standalone AA text evals *not* in the index: **MMLU-Pro · AIME 2025 ·
LiveCodeBench · Global-MMLU-Lite**. **Critically, #5819 treats these as competency labels to
approximate via PPL surrogates, not as evals in the pipeline** — the agentic/long-context/
document components (tau2, Terminal-Bench, GDPval, AA-LCR, CritPt, AA-Omniscience, HLE,
IFBench) are **not executed**. There is **no achieved AA-Intelligence-Index score in the
corpus**; AA-II is an optimization *target* proxied by BPB/downstream deltas (e.g.
[#6757](https://github.com/marin-community/marin/issues/6757): curated vs proportional mixes
"statistically indistinguishable" on macro-uncheatable-BPB).

**What is actually executed as generative evals: a reasoning subset via Evalchemy (TPU/vLLM).**
[PR #2779](https://github.com/marin-community/marin/pull/2779) (moojink, 2026-02-13) added
`EvalchemyEvaluator` running **AIME24/25 (+AIME26, OlympiadBench, LiveCodeBench v6 via
[PR #3690](https://github.com/marin-community/marin/pull/3690)), AMC23, HMMT, MATH500,
HumanEval+, MBPP+, LiveCodeBench, GPQA Diamond** as Marin eval steps on TPU via vLLM. So the
**actually-run generative set overlaps AA only partially** (GPQA Diamond, LiveCodeBench,
AIME25; plus non-index tasks). SciCode is in the PPL map and decon but **not** in the
executed Evalchemy list — a documented gap. Agentic AA components are routed to **Harbor**
where "agentic eval parity on TPU" remained an open goal as of July 2026. (Decontamination
scope ≠ execution scope: [#5519](https://github.com/marin-community/marin/issues/5519)
decontaminates against **8 AA + 849 lm-eval-harness leaves**, but that is a datakit pipeline,
not evals we run.) Percy Liang's framing (2026-07-13): curated evals feed "our basket of
200+ perplexity evals and standard generative evals … monitored at the end of training
(pretraining + midtraining + SFT) before RL."

**Comparator target — "beat Qwen."** On the pretraining side the concrete bar is Qwen (2.5/3
32B). On the released 32B, the real story is a **downstream generative-benchmark comparison
with harness caveats**, not just a perplexity gap: on the lm-eval fork pretraining suite,
**marin-32b-base 61.96 vs Qwen2.5-32B 64.66, Olmo-3-32B 63.20**
([ziqingh, 2025-11-21](https://discord.com/channels/1354881461060243556/1356487738840318002/1441540620865769554)),
with gaps concentrated in "math, code and reasoning" (dlwh). willheld reported the 32B "strong
on the large majority of benchmarks" but had to exclude **generation benchmarks** because of
OLMES 8-shot-COT-vs-lm-eval harness differences and "our Qwen must be mildly different"
([#marin-32b, 2025-10-01](https://discord.com/channels/1354881461060243556/1367246647062564995/1423016207300300912)).
Including generation tasks the mean-rank ordering had **Qwen 2.5 32B Base 2.74 ahead of
Marin 32B Base 3.00**.

**Pretraining loss "targets" (a distinct kind of target).** For hero runs the headline is
**Paloma macro-loss** against pre-registered targets. For the **July 67B-A2B run** (d=2560,
67B total / 2.01B active, MuonH, v4-2048, ~10.07T tokens ≈ 1.2e23 FLOPs), ClassicLarry
pre-registered a **stage-1 (first 8T tokens) target of 2.269 Paloma macro_loss @ seqlen 8192
/ 1024 sequences** — a **projection, not achieved**, and it lives in
**[#6044](https://github.com/marin-community/marin/issues/6044#issuecomment-4820008980)** (not
#6811). The separate pass@k acceptance bar in
**[#6811](https://github.com/marin-community/marin/issues/6811)** (ihodes) is a **placeholder**:
"Criteria is pass@256 on X Y Z evals" — "X Y Z" is literal, never concretized. The measured
2T early-cooldown reading of **2.277** (from the finished W&B run `…cooldown_step39k`) is
**apples-to-oranges** with 2.269 (2.8T vs 8T tokens, seqlen 65536 vs 8192, phase-2 vs phase-1
mix) — its closeness is coincidental. Reference bars in-thread: 32B lowest-ever Paloma 2.202;
prior 1e23 MoE final 2.234.

---

## 6. Where each tier is computed (harness / platform map)

- **`stanford-crfm/lm-evaluation-harness` fork** (Levanter, TPU) — pretraining log-prob +
  MCQA/generation; DCLM CORE + MMLU + soft/BPB metrics; being retired toward EleutherAI
  mainline. W&B logging under `marin-community/marin` (`lmeval_*` runs), metric keys like
  `eval/uncheatable_eval/bpb`, `eval/paloma/macro_loss`.
- **Evalchemy** (`mlfoundations` → `teetone` → `marin-community` fork; TPU via vLLM) —
  post-training reasoning/generative evals (the executed AA-adjacent subset).
- **HELM** (`stanford-crfm/helm`) — parallel, partly sidelined for reasoning/coding.
- **Harbor** — intended home for agentic AA components; **not at parity** on TPU at freeze.
- **Iris** (iris.oa.dev) job ladder — the scaling-validation ladders (Delphi, Table-9-optimized).
- **Delphi scaling ladder** — the pre-registered macro-loss proxy runs (Nemotron/CommonPile).
- **Marin-native Levanter evaluator** ([PR #6726](https://github.com/marin-community/marin/pull/6726))
  — the Table-9 BPB macro (W&B project `marin-eval`, run family `olmo_base_eval_table9`).
- **OLMES** — the 8-shot-COT harness whose differences changed marin-32b generation-benchmark numbers.

---

## 7. Conflicts, superseded states, and gaps

- **Objective evolved:** Uncheatable-Eval-only → interim top-3 macro → **Table-9 51-component
  macro** for mixture/DSP selection (Uncheatable remained a coexisting validation target).
- **Superseded:** homegrown `internal_eval` → lm-eval-harness fork; #4389 soft-agentic-proxy
  → #4550; Mega-Evals (#2663) never merged.
- **Target ≠ achieved / target ≠ executed:** AA v4.0.4 components are competencies to
  approximate; only a reasoning subset (Evalchemy) actually runs. No achieved AA-II score
  in corpus.
- **Citation precision:** the 2.269 stage-1 target is in **#6044** (not #6811); #6811's
  pass@256 bar is a placeholder; the 2.277 cooldown reading pins to the W&B run, not an issue.
- **Gaps:** no final 1e23 Delphi Paloma macro-loss number; the 1.9 asymptote is contested;
  the 8 omitted DCLM-CORE tasks (#574) aren't enumerated; 3e20/1e21 Table-9 rungs still
  running; SciCode not executed despite being an index component.

### Proxy-validity, quantified (why soft metrics, and where proxies fail)

Two studies anchor *why* Marin trusts soft / normalized-probability metrics over raw accuracy. [#5247](https://github.com/marin-community/marin/issues/5247) (Calvin-Xu, 2026-04-28): on 300M qsplit core rows (n=240), raw **MMLU accuracy has ~no monotonic relation to BPB (Spearman −0.069)** and near-zero vs choice_logprob_norm (−0.026), while MMLU-acc vs **choice_prob_norm correlates +0.866** — close-out: *"accuracy is itself noise; use the smooth proxy with the highest SNR."* [#6096](https://github.com/marin-community/marin/issues/6096) (Helw150): across 10 SimpleRL-Zoo base models, base-model reasoning features predict post-RL *level*, but **predicting the RL gain fails (R²=0.33)** — pre-RL pass@K is the main predictor of post-RL pass@K; coding/SWE-bench is flagged as the untested next domain.

---
<!--provenance-->
> *Gold reference answer. Data: marinmirror frozen 2026-07-16 eval corpus, accessed via
> `evals/runners/mum-frozen` (refresh disabled; single-process). Every load-bearing number
> and citation re-verified against the frozen corpus via `mum-frozen show <url>`. Numbers
> labeled achieved / measured / target / projected; hardware/config attached where in-corpus;
> close-out state preferred over mid-thread proposals; unciteable claims omitted.*
>
> *Question (id=benchmarks): "What are our target benchmarks and our development proxies?"*
