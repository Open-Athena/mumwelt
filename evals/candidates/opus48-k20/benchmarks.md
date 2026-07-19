# Target benchmarks and development proxies

**Short answer.** Marin runs a deliberately two-tier evaluation stack. The **targets** — what a finished model is graded on — are downstream capability and agentic benchmarks, anchored to the **Artificial Analysis (AA) Intelligence Index** as a reference taxonomy and run through **Harbor / Evalchemy / lm-evaluation-harness**. The **development proxies** — what actually steers day-to-day decisions — are almost entirely **perplexity / bits-per-byte (BPB)** metrics: **Paloma macro loss** for hero-run loss preregistration, **Uncheatable Eval macro BPB** and (since 2026-06-26) **OLMoBaseEval Table-9 macro BPB** for data-mixture selection, **feBPB** for tokenizer choice, and a **200+-slice perplexity-gap suite** for coverage diagnostics.

The bridge between the two tiers — *does the proxy actually predict the target?* — is an explicit, still-open investment area, and the honest state is: **BPB is an excellent coverage and signal-to-noise metric and a much weaker capability metric.**

---

## 1. The governing philosophy: "perplexity-pilled," with correlation as the contract

The strategy was stated plainly by David Hall in #evals: *"we're gonna be perplexity-pilled so we just want to measure correlation between task and ppl and then stop looking at the tasks once we get good correlation"* — to which Will Held replied *"that is my pitch for the long term direction... Many evals -> a few PC's. And then have a single PPL eval that is correlated with each PC"* ([discord, 2025-11-20](https://discord.com/channels/1354881461060243556/1356487738840318002/1461766966040596644)).

That target/proxy split is institutionalized in the **Eval Manager** role epic [#6499](https://github.com/marin-community/marin/issues/6499) (penfever, 2026-06-18), which owns *"the eval suite composition per stage"* and *"the per-stage go/no-go gate metric"*, with the explicit rule:

> **"Nothing advances on perplexity alone past Stage 3; Stages 4–10 gate on capability / agentic / safety benchmarks."**

So: proxies gate early stages, target benchmarks gate late ones.

Percy Liang's most recent framing (2026-07-13) describes the assembled whole as *"our basket of our 200+ perplexity evals and standard generative evals that we will monitor at the end of training (which includes pretraining + midtraining + SFT) before RL"* ([discord](https://discord.com/channels/1354881461060243556/1356487738840318002/1526324939743957022)).

---

## 2. Target benchmarks

### 2a. Artificial Analysis Intelligence Index — the reference taxonomy

The clearest statement of "what we want to be good at" is dlwh's mapping in [#5819](https://github.com/marin-community/marin/issues/5819) ("Experiment: Find pretraining proxies for Artificial Analysis competencies", opened 2026-05-18). His matrix is *"based on the current Artificial Analysis methodology page for Intelligence Index v4.0.4"* ([comment](https://github.com/marin-community/marin/issues/5819#issuecomment-4482092952)), with index components:

**GDPval-AA, τ²-Bench Telecom, Terminal-Bench Hard, SciCode, AA-LCR, AA-Omniscience, IFBench, HLE (text-only), GPQA Diamond, CritPt** — plus standalone AA text evals not in the Index but treated as useful surrogates: **MMLU-Pro, AIME 2025, LiveCodeBench, Global-MMLU-Lite**. (dlwh notes specifically that *"AA lists MMLU-Pro as an additional text eval rather than a main Intelligence Index component."*)

The same 16-eval v4.0 list was independently enumerated with item counts on the **decontamination** side by ravwojdyla-agent in [#5519](https://github.com/marin-community/marin/issues/5519#issuecomment-4464748971) (2026-05-16) — 8 configured (HLE 2,158 text-only; AA-Omniscience 6,000; GPQA Diamond 198; MMLU-Pro 12,032; LiveCodeBench ~315; SciCode 288; IFBench 294; GDPval 220) and 8 skipped as unusable for decon (AA-LCR — *"private/unreleased AA dataset"*; τ²-Bench Telecom — *"programmatically generated (no stable text)"*; Terminal-Bench Hard 44; CritPt 70; AIME 2025 60; MMMU Pro; APEX-Agents-AA; Global-MMLU-Lite). That scope came from dlwh's instruction *"imho we should just throw agent against everything in AA and lm eval harness"* ([discord, 2026-05-11](https://discord.com/channels/1354881461060243556/1441211384279994529/1503517600091734169)).

**Important caveat — AA is a taxonomy, not a runnable suite.** On ihodes' enumeration issue [#6050](https://github.com/marin-community/marin/issues/6050) ("TBD: Enumerate / hook up evals of interest — AA II evals + any others?"), Will Held replied 2026-06-01: *"FWIW we can't set exactly set up AA II because it's not open source [which is good for an eval, no cheating :)]. They have their own harnesses for the things which are public and a few of their own datasets which is non-public."* **No numeric AA Intelligence Index target for a Marin model appears anywhere in the corpus.**

A second, deliberate constraint: AA component items are **banned from the proxy suite**. dlwh, [#5819](https://github.com/marin-community/marin/issues/5819#issuecomment-4482112279): *"We should not add the actual AA component questions, or close public benchmark train/test questions, as core perplexity evals. That includes GPQA/HLE/AIME/MATH/MMLU-Pro/IFBench/LiveCodeBench/SciCode exact items, even where a train split exists."* He later retracted HLE text-only specifically and issued a full [benchmark-leak audit](https://github.com/marin-community/marin/issues/5819#issuecomment-4483211940). Targets and proxies are kept strictly disjoint.

### 2b. The harnesses that actually run the targets

| Layer | Harness | Evidence |
|---|---|---|
| Pretraining, log-prob + generative | **lm-evaluation-harness** (Marin fork) | Percy Liang, [2026-01-16](https://discord.com/channels/1354881461060243556/1356487738840318002/1461766966040596644): *"For pre-training, we are using lm-evaluation-harness (for log probs and generation evals via vllm)"*; willheld, 2026-01-21: *"I'm up to 71 evals in lm-eval-harness for pretraining"* and *"to get us to the 100 task target"* |
| Post-training, static benchmarks | **Evalchemy** (`marin-community/evalchemy` fork, lm-eval-harness rebased) | Benjamin Feuer, [2026-06-11](https://discord.com/channels/1354881461060243556/1356487738840318002/1514733965070827641): *"this combined with `marin-community/harbor` should cover the evals we need for post-training"*; Evalchemy invocation tracked in [#6866](https://github.com/marin-community/marin/issues/6866) |
| Post-training, agentic | **Harbor** (+ Daytona sandboxes) | [#6865](https://github.com/marin-community/marin/issues/6865) (yonromai, 2026-07-02): *"DoD: Harbor evals can be invoked, **Qwen3-32B parity on TBench2**. In scope: Model: Qwen32B; Cluster: TPUs; Sandboxes: Daytona"* |
| Agentic SWE on TPU | terminus-2 agent on **SWE-bench-Verified** | [#6958](https://github.com/marin-community/marin/issues/6958) (penfever, 2026-07-05): agentic SWE evals *"now run end-to-end on Iris"* v6e TPU |

Note the acceptance criterion pattern: the target-benchmark work is gated on **reproducing a known model's published score** (Qwen3-32B on TBench2), not on Marin's own number.

A live, curated **list of evals the core team wants** was opened as **#7090** on 2026-07-10 (Benjamin Feuer, [discord](https://discord.com/channels/1354881461060243556/1356487738840318002/1525123185756864583): *"The Marin core team has started to put together a list of evals we are particularly interested in seeing as part of Marin development"*). **That issue is not in this frozen corpus**, so its contents can't be reported here — it is the most likely current canonical answer to "what are the target benchmarks," and is a known gap in this write-up. Percy's reply above situates it: those evals get **added to** the 200+ PPL basket, not substituted for it.

### 2c. Where Marin actually stands on target benchmarks (measured)

On a pretraining eval-task average, ziqingh reported ([discord, 2025-11-21](https://discord.com/channels/1354881461060243556/1356487738840318002/1461766966040596644)): Qwen2.5-32B 64.66, Olmo-3-1125-32B 63.20, Meta-Llama-3-70B 62.88, **marin-32b-base 61.96**, Qwen3-8B-Base 61.82, OLMo-2-32B 60.97, gemma-3-27b-pt 60.81. dlwh's read in the same thread: *"we're not great at math, code and reasoning"*, with persistent gaps versus Qwen2.5 in *"MMLU, MMLU Pro, gpqa, commonsenseqa, BBH, Coding: human eval, Alignment: toxigen, thruthful_qa, Reasoning: ANLI, MUSR"* — adding the caveat *"We generally suspect some benchmaxxing from qwen."*

**Data-hygiene warning on recent numbers:** rohithck flagged 2026-07-16 that Delphi HumanEval results reported as 10-shot were *"for 0-shot not 10-shot"* ([#7229](https://github.com/marin-community/marin/issues/7229), [discord](https://discord.com/channels/1354881461060243556/1356487738840318002/1526324939743957022)), with willheld confirming from the W&B log (`num_fewshot has been set to 0 for humaneval in its config. Manual configuration will be ignored.`) and rohithck adding *"the fewshot prompt construction for humaneval is also broken."* Treat recent HumanEval figures as under review.

---

## 3. Development proxies

### 3a. Paloma macro loss — the hero-run preregistration metric

Paloma `macro_loss` is what large runs are **preregistered and scored against**, via a Chinchilla-form scaling fit.

- **1e23 MoE (129B total / 16B active):** target *"predicting final paloma macro loss of 2.25"* ([#4697](https://github.com/marin-community/marin/issues/4697#issuecomment-4271187922), 2026-04-17). **ACHIEVED: "Final paloma/macro_loss was 2.234, 1% better than the initial prediction of 2.252"** ([#4697](https://github.com/marin-community/marin/issues/4697#issuecomment-4498921338), 2026-05-20).
- **July 67B-A2B hero run (10.07T tokens, TPU v4-2048):** **TARGET** — *"Preregistered loss target for stage 1 of the run (first 8T tokens): 2.269 paloma macro loss when evaled at seqlen 8192 on 1024 sequences per eval"* ([#6044](https://github.com/marin-community/marin/issues/6044#issuecomment-4819900707), 2026-06-27), from a 3-anchor fit at E=1.4.
- The **2T intermediate cooldown** landed at **2.2772 Paloma macro loss** (BPB 0.8242) at step 42,149 ([#6811](https://github.com/marin-community/marin/issues/6811)). **This does not retire the 8T target** — it is a favorable early signal only: the cooldown fully decays LR, switches to the phase-2 mix, and evaluates at 8× context (65,536) rather than the preregistered seqlen 8,192. No achieved 8T number exists in this corpus.

**Paloma is explicitly *not* the mixture-selection metric.** willheld, [2026-05-28](https://discord.com/channels/1354881461060243556/1462895580064911522/1509706849459372212): *"Both Hellaswag and Paloma Macro BPB are maxed (within the swarm at least) by proportional mixing"* — i.e. it gives no optimization gradient. Earlier: *"Paloma fails (b) rather than (a)… Too contaminated since most of it comes from sources that we (and everyone else) either directly or indirectly sample from"* ([2025-09-16](https://discord.com/channels/1354881461060243556/1356487738840318002/1417578522360418476)), against his three-part rubric for a north-star metric: (a) correlated with what we care about, (b) hard to cheat, (c) cheap and low-noise.

### 3b. Uncheatable Eval macro BPB → OLMoBaseEval Table-9 — the mixture-selection metric, and its handover

The literal optimization target in the mixture-scaling issues was **`eval/uncheatable_eval/bpb`, lower is better** ([#6602](https://github.com/marin-community/marin/issues/6602), [#6608](https://github.com/marin-community/marin/issues/6608), Calvin-Xu, 2026-06-23). Its candidacy came from Will Held: *"Uncheatable eval seems likely to be (b) and (c), so if we establish (a) I would be willing to advocate for it being our main north star metric."*

**It was superseded on 2026-06-26.** Calvin-Xu, [#6608](https://github.com/marin-community/marin/issues/6608#issuecomment-4804956822): the older uncheatable parent run was stopped because it was *"superseded by the current Table-9 macro objective."* The new objective is the **OLMoBaseEval Easy Table-9 51-component macro BPB** ([#6611](https://github.com/marin-community/marin/issues/6611#issuecomment-4809543112)), with a Marin-native evaluator landed in [PR #6726](https://github.com/marin-community/marin/pull/6726) reproducing the external oracle to **macro abs diff 1.9e-8** across 104 tasks / 88,592 instances.

**The proxies disagree, and the disagreement is documented:**

- **Best hyperparameter is target-specific.** KL=0.1 selected against `uncheatable_eval/bpb` ([#6602](https://github.com/marin-community/marin/issues/6602#issuecomment-4798297812)); KL=0.025 against the Table-9 macro ([#6611](https://github.com/marin-community/marin/issues/6611#issuecomment-4809543112)).
- **Surrogate rankings flip between targets.** The paper-faithful OLMix `single_tied` surrogate scores OOF Spearman **0.813** against uncheatable BPB but collapses to **0.380** against the Table-9 macro, where effective-exposure DSP reaches **0.918** ([#6602](https://github.com/marin-community/marin/issues/6602#issuecomment-4797047344), [#6608](https://github.com/marin-community/marin/issues/6608#issuecomment-4804956822)).
- **Aggregate BPB wins can coexist with downstream regressions.** In the curated-vs-nemotron iso-FLOP verdict ([#6757](https://github.com/marin-community/marin/issues/6757#issuecomment-4847173767), Helw150, 2026-06-30, MoE d512–d1280 at 3e17–3e19 FLOPs), curated wins macro uncheatable BPB **1.78× [1.45, 2.24]** iso-FLOP speedup and wins HumanEval/GSM8K/MBPP/all 8 math subjects — while **significantly losing commonsense**: hellaswag [0.48, 0.80], winogrande [0.55, 0.94], piqa [0.59, 0.75]. Curated vs *proportional* on the same metric is **1.14× [0.93, 1.41] — not significant**, and is recommended only as a tie-breaker, *"not a robust win."*
- **Different proxies have different sensitivity to the same intervention.** On the datakit fuzzy-dedup arm, `paloma/macro_loss` moved −0.0129 (2.4σ) while `uncheatable/macro_loss` moved −0.0335 (4.6σ) ([#5309](https://github.com/marin-community/marin/issues/5309)).

**Latest stated form of the mixture objective** is neither of the single scalars above. [#5362](https://github.com/marin-community/marin/issues/5362#issuecomment-4773354017) ("Select Metric for Data Mix Optimization", close-out 2026-06-22): *"Landed on `L(theta) = \sum(hinge(task_loss, prop_loss-\eps)) + \sum_{code+math} task_loss + L_2(theta, theta_prop)`"* — a per-task hinge against proportional, plus explicit code+math terms, plus an L2 anchor to proportional. The motivation is recorded earlier in the same issue: with a single aggregate, *"the factors that are more clearly explainable based on the data mix (e.g. math and coding capabilities) seem to end up a bit over-optimized because they are the easiest to optimize!"* ([comment](https://github.com/marin-community/marin/issues/5362#issuecomment-4453698111)).

### 3c. feBPB — the tokenizer/architecture proxy

**FLOP-equivalent BPB**, introduced 2026-07-03 in [PR #6916](https://github.com/marin-community/marin/pull/6916) / [#6796](https://github.com/marin-community/marin/issues/6796). Rationale: cross-entropy is per-token so a tokenizer emitting fewer tokens looks better for free; BPB makes quality tokenizer-agnostic, and inference-FLOPs-per-byte prices cost — crucially **at the ~250B-total / ~20B-active deployment target, not at the small proxy**, because the vocab-dependent LM head is 30–50% of proxy FLOPs but only ~2–3% at deployment. *"Pricing at the wrong scale over-penalizes large vocab and flips verdicts"* ([#6796](https://github.com/marin-community/marin/issues/6796#issuecomment-4878397410)). Best measured arm: trained-superbpe-64k-t32k + n-gram at **1.1532 feBPB, −6.8%** vs the Llama-3 `marin-128k` reference of 1.2376 (rising to −10.0% at serving/training FLOP ratio ρ=4.5).

### 3d. The perplexity-gap suite — the coverage proxy

[#5005](https://github.com/marin-community/marin/issues/5005) / [#4934](https://github.com/marin-community/marin/issues/4934) (evaluator landed in [PR #4962](https://github.com/marin-community/marin/pull/4962)), public dashboard at <https://marin.community/analysis/perplexity-gap/>. dlwh's description of the base suite ([discord, 2026-04-27](https://discord.com/channels/1354881461060243556/1356487738840318002/1498186469578375269)): *"We already have a standard held-out loss suite built largely around Paloma, plus Uncheatable Eval, a public benchmark with recent news, code, fiction, and other harder-to-contaminate slices."*

Measured gaps (Marin 8B vs peers, [#5005](https://github.com/marin-community/marin/issues/5005#issuecomment-4299449950)): **Paloma gap only +0.0029 bpb vs Llama and uncheatable +0.0049 — but FineWeb2 multilingual +0.2699.** dlwh's summary: *"We continue to be broadly good at edited English prose… We're very bad at non-English prose. So bad that we made a button to hide multilingual evals… 'Messy' data is a blind spot."* This is precisely the argument for the suite: the headline proxies looked fine while real weaknesses hid outside them. Coverage was then expanded to 136 programming-language slices under [#5254](https://github.com/marin-community/marin/issues/5254).

---

## 4. The bridge: how good are the proxies at predicting the targets?

This is its own investment area — [#6712](https://github.com/marin-community/marin/issues/6712) (ihodes, 2026-06-26), *"Strengthen the evals / PPL / diagnostic signals that predict downstream quality"*, whose single "metric to improve" row reads **"Rank correlation between PPL proxy and downstream eval."** **Fidelity note: #6712 records no baseline and no target number, and had zero comments as of the freeze.** The evidence lives in its children and neighbors, and it cuts both ways:

**Where proxies work:**
- **Math capability level.** [#6096](https://github.com/marin-community/marin/issues/6096) (Helw150, 2026-06-01): base-model `cot` BPB on real math traces vs post-SimpleRL-Zoo accuracy across 10 models — Pearson **r = −0.89**, and **r = −0.97** within the 7–8B cohort (removing the size confound). Real traces beat templated synthetic (−0.89 vs −0.68…−0.72): *"The predictor is specific to the target distribution, not 'reasoning-shaped' text."*
- **Mixture regression.** OOF Spearman **0.918–0.980** predicting BPB targets from mixture weights ([#6611](https://github.com/marin-community/marin/issues/6611#issuecomment-4804306911)).
- **Signal-to-noise.** *"By family, all Paloma and uncheatable BPB/loss metrics clear SNR ≥ 2. Task/lm-eval metrics are much noisier: 38/120 lm_eval metrics and 40/842 MMLU-subject metrics clear SNR ≥ 2"* ([#5247](https://github.com/marin-community/marin/issues/5247#issuecomment-4368082302)). This is the core practical case for proxies: raw benchmark accuracy is too noisy to optimize at small scale.

**Where proxies fail:**
- **MCQA accuracy.** Same 300M panel, n=240: Spearman(standard MMLU accuracy, BPB) = **−0.069** — essentially uncorrelated. The normalized choice-probability proxy works instead (**+0.866**) ([#5247](https://github.com/marin-community/marin/issues/5247#issuecomment-4339766407)). Calvin-Xu's conclusion: *"generic uncheatable/Paloma BPB is not enough by itself for MCQ/task accuracy."*
- **Agentic benchmarks — tested and refuted.** [#4389](https://github.com/marin-community/marin/issues/4389) tried positive-trace loss and the success−failure gap as proxies. On MATH: *"there was not a monotonic relationship for any of these (rank correlation was generally < 0.8)… success-failure gap generally did worse than just loss"* ([2026-04-03](https://github.com/marin-community/marin/issues/4389#issuecomment-4183999166)). On SWE-bench Verified with 25 Qwen3-8B finetunes: *"the results were not great (highest was ~0.3)"*, and pathologically *"the best model actually got higher loss on all 25 sets of trajectories than the worst model"* ([2026-04-08](https://github.com/marin-community/marin/issues/4389#issuecomment-4209804794)). Helw150's read: *"strong evidence that something in the fine-tuning procedure probably breaks the comparability of the logprobs."* **Issue closed 2026-06-10** (*"Rohith is working on this elsewhere"*, [comment](https://github.com/marin-community/marin/issues/4389#issuecomment-4666807165)); PR [#4539](https://github.com/marin-community/marin/pull/4539) auto-closed unmerged. Successor [#4550](https://github.com/marin-community/marin/issues/4550) has no comment after 2026-04-21 in this corpus. An independent SNR measurement corroborates the ordering: success-only agentic trace BPB scores SNR **4.98**, the success−failure gap only **1.46** ([#5401](https://github.com/marin-community/marin/issues/5401#issuecomment-4368778080)).
- **RL gain (as opposed to level).** [#6096](https://github.com/marin-community/marin/issues/6096#issuecomment-4836443867) close-out: *"RL adds a near-constant +15.8 ± 4.1 pts to everyone… Predicting the RL gain (post − base) fails: R² = 0.33."* The proxy forecasts capability level, which RL preserves — **not** which weak base benefits most from RL.
- **Extrapolation across scale.** rohithck, [2026-05-19](https://discord.com/channels/1354881461060243556/1493722964644990996/1506401125471879298): *"I am off by 0.1 nats when trying to extrapolate [GSM8k loss] from < 1e21 to 1e23"* — with the sign clarified (*"the models do better than expected"*) and the magnitude partly self-questioned after willheld showed a smaller same-direction error. The robust claim is *consistent under-prediction*, not a settled 0.1-nat figure.

The unresolved question is stated best by rohithck ([2026-04-27](https://discord.com/channels/1354881461060243556/1356487738840318002/1498463568352378961)): *"there are two things… 1) for any task, in principle does BPB on some 'right' validation corpus predict performance; 2) if yes, how do we identify this validation set. I'd guess the answer to (1) is yes. But the answer to (2) is still idk."* And a standing methodological objection from dlwh: BPB has a *"temperature problem… Deep cooldowns decrease temperature but don't really fundamentally shift the ordering of the tokens"* ([2026-04-27](https://discord.com/channels/1354881461060243556/1356487738840318002/1498466558412849233)).

Anti-benchmaxxing discipline is explicit: *"treat large BPB wins on known slices as red flags for leakage, not green flags"*, publish absolute BPB and baseline deltas rather than improvement-over-self ([#5005](https://github.com/marin-community/marin/issues/5005#issuecomment-4300444399)). And dlwh's position on proxy selection: *"i am very against only picking datasets that correlate with downstream evals, but i am very much in favor of including datasets that correlate with downstream evals"* ([2026-04-28](https://discord.com/channels/1354881461060243556/1462895580064911522/1498811239927906405)).

---

## 5. Gaps and caveats in this answer

- **#7090** — the core team's current curated eval list, the most likely canonical target-benchmark answer — **is not in this frozen corpus**. Only its existence and framing are citable.
- **No numeric AA Intelligence Index target** for any Marin model exists in the corpus, and AA II is **not runnable** as published ([#6050](https://github.com/marin-community/marin/issues/6050)).
- **"The big three" pretraining evals** are referred to by name (willheld, [2026-06-26](https://discord.com/channels/1354881461060243556/1462895580064911522/1520176931876638861): *"Scaling for what I think of as 'the big three' pretraining evals"*) but **are never enumerated in any source I found** — I have deliberately not guessed which three.
- **#6712's headline metric has no baseline or target value**, so "how good are our proxies" has no single agreed number.
- The **mixture-selection metric changed twice within four weeks** (uncheatable BPB → Table-9 macro on 2026-06-26; and #5362's hinge-based composite as of 2026-06-22). Any statement of "the" proxy is time-stamped, not permanent.

---
<!--provenance-->
> *Data: marinmirror — 68026 chunks, built 48h ago · summaries through 2026-07-06_2026-07-12. Frozen eval corpus (`MARIN_EVAL_FREEZE=2026-07-16`, retrieval pool pinned to k=20); no refresh triggered.*
>
> *Query: "What are our target benchmarks and our development proxies?"*
>
> *Sub-queries: "Artificial Analysis Intelligence Index as target benchmark suite, and pretraining proxies for AA competencies (#5819)" · "pretraining development proxy metrics — Paloma macro loss, Uncheatable Eval macro BPB, OLMoBaseEval Table-9, feBPB: which is canonical for which decision" · "data-selection diagnostics (#6712), downstream scaling laws (#4550), SimpleRL-Zoo proxy correlation (#6096)" · "soft proxies for agentic benchmarks (#4389 / PR #4539) and the ICL/MATES influence proxy (#6521)" · "post-training benchmark harnesses — Harbor, Evalchemy, lm-evaluation-harness — and their parity targets" · "the standard pretraining eval suite and the 'big three' pretraining evals" (dry: never enumerated in corpus).*
