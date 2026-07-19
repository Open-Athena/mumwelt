# The July 2026 milestone

## The one-line version

The milestone is titled **"July milestone: complete 67B-A2B MoE; start XB-AYB MoE on B200s; start post-training Marin MoEs"** ([weekly summary, Jul 6–12](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).

It **runs to mid-August and is explicitly a preparation month, not a finish line**. The throughline it serves is the **Marin 2026 contender**: when the twelve NVL72 racks land around **September 15**, the team kicks off its largest hero run — a **~512B-A16B model on 20T tokens**. July's job is to be ready for that, and for the **~120B-A8B B200 run at the end of the month**, by finishing and cooling down the two runs in flight, getting Blackwell MFU up, locking the data mix and architecture, and pricing post-training ([Jul 6–12 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).

**Four workstreams carry it** ([same source](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)):

1. NVL72 pretraining MFU toward **25%** for the September run;
2. a **pretraining recipe** that sets up post-training well;
3. **post-training infrastructure that is fast enough** — including a working definition of "fast enough";
4. a **post-training recipe** for the best model the available FLOPs allow.

The milestone is organized on GitHub into three tiers: **Hero Runs** (the concrete use of compute), **Commitments** ("what we must land this milestone to be ready for the runs ahead"), and **Areas of Investment** ("where time goes as-needed around the commitments; in practice the bulk of the month's work") ([Jul 6–12 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).

**Two structural notes before the detail.** First, there is **no single "July milestone" tracking issue** — it is a GitHub *Milestone*, whose scaffolding is a batch of ~20 epic stubs opened by @ihodes on **2026-06-26** ([#6689](https://github.com/marin-community/marin/issues/6689), [#6700](https://github.com/marin-community/marin/issues/6700)–[#6716](https://github.com/marin-community/marin/issues/6716)); the milestone statement itself appears only in the weekly summaries' header. It succeeded the June milestone, *"June: kick off overtrained 67B-A2B 10T"* ([Jun 22–28 summary](https://mws.oa.dev/summaries/summary-2026-06-22_2026-06-28.html)). Second, **the "four workstreams" are a narrative framing and do not map 1:1 onto the three-tier issue structure** — no issue enumerates them, and there is **no milestone-level definition of done**; DoD exists only per-epic (quoted below where it exists).

One telling detail: the epics were **renamed mid-milestone from "July run" to "Aug 1 run"** — [#6706](https://github.com/marin-community/marin/issues/6706)'s title moved from *"Get B200 MFUs above X in advance of July run"* to *"Get B200 MFUs above 20% in advance of Aug 1 run"*. The B200 kickoff slipped to the milestone's edge, and the bar hardened from a question mark into a number.

---

## Tier 1 — Hero runs (in flight)

### 67B-A2B Grug MoE, 10T tokens, TPU v4-2048 — [#6704](https://github.com/marin-community/marin/issues/6704) / [#6044](https://github.com/marin-community/marin/issues/6044)

The centerpiece. [#6704](https://github.com/marin-community/marin/issues/6704) is the tracking/planning issue ("This is a tracking / planning issue for #6044 … This should likely land in August"); [#6044](https://github.com/marin-community/marin/issues/6044) is the day-to-day log.

- **Model/hardware**: 67.1B total / 2.01B active, d=2560, 256 experts top-4, seq 8192, sliding window 2048, MuonH — on **TPU v4-2048**, ~10.07T token horizon ([#6044](https://github.com/marin-community/marin/issues/6044)).
- **Progress**: passed roughly **2.3T tokens**, holding **~18.6% MFU** on the resumed run ([Jul 6–12 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)). Completion is **targeted around August 30** — i.e. *after* the July milestone closes, which is why the milestone leans on intermediate cuts.
- **First intermediate cooldown — [#6811](https://github.com/marin-community/marin/issues/6811) — CLOSED, and the week's most-cited result.** From step 39,000, Larry Dial ran a 3,150-step LR decay that finished at **2.277 Paloma macro-loss** (bits-per-byte 0.824) from a checkpoint at only ~2.1T tokens.
- **Read that number carefully.** It compares against a **preregistered stage-1 target of 2.269** Paloma macro-loss, but that target is set at the **8T-token (80%-completion)** mark, evaluated at **seqlen 8,192** and **before** the final cooldown, whereas the 2T cut fully decays LR, switches to the phase-2 mix, and evaluates at **8× context (65,536)**. The summary's own framing: **"strongly favorable early read … though not apples-to-apples"** — it does **not** retire the 8T preregistration ([#6811](https://github.com/marin-community/marin/issues/6811), [Jul 6–12 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).
- **Why the cooldown exists at all** — this is the milestone logic in miniature. Its stated criteria is *"pass@256 on X Y Z evals, which we believe gives us enough to do mid+SFT+RL and exercise our inference and post-training stack early (so we're ready to run when the full 10T [#6704](https://github.com/marin-community/marin/issues/6704) run lands)"* ([#6811](https://github.com/marin-community/marin/issues/6811)). The step-42150 checkpoint was verified and copied to CoreWeave S3 and a v6e inference zone by Romain Yon, so mid-training, SFT, RL and inference can start against a usable model well before the full run lands.
- **A second intermediate cooldown at 5T is planned for early August** ([Jul 6–12 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).

### Marin 11B-A1.5B hardware-and-loss validation on H100 — [#6716](https://github.com/marin-community/marin/issues/6716)

Epic title: *"Hardware and Loss Validation runs on H100s"*. A d2048 MoE (24 layers, 64 experts top-4, GQA 4:1, ~1.53B active / 10.6B total) on **64×H100 on CoreWeave**, deliberately pushed to 500B tokens — *well past* compute-optimal — because the artifact is not the point: as Isaac Hodes put it, *"this is more about exercising the hardware instead of delivering an artifact."* It **held ~23.8% MFU** (~1.37M tok/s), completion expected ~July 13–14 ([#6716](https://github.com/marin-community/marin/issues/6716)).

One honest caveat recorded with it: the FLOP counter treats attention as fully quadratic and ignores the 2048-token sliding window, so **the reported 23.8% is slightly optimistic** on the attention term ([#6716](https://github.com/marin-community/marin/issues/6716)).

---

## Tier 2 — Commitments (the must-lands)

These three are the gate on the end-of-month B200 run.

### 1. Get B200 MFU above 20% — [#6706](https://github.com/marin-community/marin/issues/6706)

Epic title: *"Get B200 MFUs above 20% in advance of Aug 1 run"*; the retrievable issue body is one line — *"Need to be at 20%+?"*. The sibling investment epic [#6710](https://github.com/marin-community/marin/issues/6710) states the metric formally: *"Metric to improve | Training MFU % (≥20% per #6706)"*.

**Status: not there yet, but closing.** B200 MFU reached **17.8% whole-model, single-node on 8×B200 at d5120**, by wiring Tri Dao's QuACK SM100 grouped GEMM (the SonicMoE kernel) into the expert MLPs via a torch-free CUTLASS shim. The climb was **12.5%** (H100 config reproduced as-is) → **16.2%** (width sweep) → **17.8%** (QuACK), which established that **batch is a dead lever** — the workload is arithmetic-intensity-starved, not batch-starved — while **width is live**. The QuACK gated GEMM alone measured ~1,175 TFLOP/s, ~52% of peak ([#6706](https://github.com/marin-community/marin/issues/6706), detail under [#6710](https://github.com/marin-community/marin/issues/6710)).

Worth stating plainly: **17.8% is measured; 20% is the bar; 25% is the September target.** Those are three different numbers and only the first is an achievement.

Beware one easy conflation: [#6693](https://github.com/marin-community/marin/issues/6693) *"[tracking] >20 MFU at 128 GPUs workstream plan"* is a **separate H100/128-GPU** target with its own rows (FA4 SM90, PGLE, fp8 expert GMM [#6699](https://github.com/marin-community/marin/issues/6699)) — not the B200 bar.

### 2. Price post-training — [#7074](https://github.com/marin-community/marin/issues/7074)

Epic title: *"Approximate tokens (upper bound) needed for post-training"*. **Status: answered this milestone.** Benjamin Feuer costed the pipeline — RL over 10 experts, then multi-teacher on-policy distillation (MOPD) into the student on the Nemotron path — in **hardware FLOPs** deliberately (the pretraining plan's currency, so RL's low ~1% aggregate MFU is counted rather than hidden). The result ([#7074](https://github.com/marin-community/marin/issues/7074)):

- Post-training a small 67B-A2B hero costs roughly **0.25–0.9× its 10T pretraining in HW-FLOPs** (about 1.2–3.9× the cheaper 2T-cut), and only **~5–20% of the 120B big run**.
- The cost is **dominated by the RL leg**, whose cost scales with the experts' hosting footprint — so a **~5× environment speedup divides all of it by five**.
- In literal tokens it is small — order **10⁹–10¹⁰, under 0.5% of a 10T pretrain** — though that undercounts, since each post-training token is far costlier to produce.

This is the number that makes workstreams 3 and 4 tractable: it tells you how much of the FLOP budget post-training may claim.

### 3. Lock the shape of the B200 run — [#7073](https://github.com/marin-community/marin/issues/7073)

Epic title: *"Shape of model (arch + tokens etc) for Aug 1 run"*. **Status: still open — 4/14 sub-issues closed, and no decision was recorded in the Jul 6–12 window** ([#7073](https://github.com/marin-community/marin/issues/7073)).

This commitment is to lock the **data mix**, the **architecture**, and a **preregistered loss target** for the ~120B-A8B B200 run. Its inputs are the other two commitments ([#6706](https://github.com/marin-community/marin/issues/6706), [#7074](https://github.com/marin-community/marin/issues/7074)) plus the architecture/data-mix investment epics. The summary names the piece that matters: *"Preregistering the loss target, as was done for the 67B-A2B run's stage 1 before it launched, is the piece that turns the decision into a testable prediction"* ([#7073](https://github.com/marin-community/marin/issues/7073)).

**This is the single biggest open risk in the milestone** — the run is meant to start at end of month, and its architecture, data and prediction are all still unfixed.

Also tracked adjacent to the commitments: [#6705](https://github.com/marin-community/marin/issues/6705) *"[Hero run] Post training on 67B-A2B 10T"*, which is candid that *"realistically the final hero run training on the 67B-A2B on full 10T tokens won't begin until early (or mid?) August"*, and [#6053](https://github.com/marin-community/marin/issues/6053) *"What is our post-training plan to attain EOY goal"* — whose only substantive comment (penfever) says there are *"multiple plans in-flight"* living in Google Docs **outside this corpus**, so the EOY post-training plan is not readable here.

### A note on workstream 3: "fast enough" is still literally undefined

The milestone commits to *"post-training infrastructure that is fast enough (and a working definition of fast enough)"* — and the second half of that clause is not rhetorical. Both places the bar should live are open TODOs:

- [#6709](https://github.com/marin-community/marin/issues/6709) (Inference speed for RL rollouts) names its metric as rollout throughput / generation latency but leaves the target open: *"What are we targeting — rollout tok/s, end-to-end RL step time, or cost/token?"*, with *"First task: measure and document the current baseline(s) before setting a target."*
- [#6870](https://github.com/marin-community/marin/issues/6870) is the literal bar and is literally blank: *"Fast enough: **TODO: Within X% of Jax on GPUs vs. within Y% of reference model (e.g. Qwen?)**"*.

What *is* measured is the demand signal. Benjamin Feuer reported MarinSkyRL running **131k-context RL on MoE models on CoreWeave**: Qwen3-Coder-30B-A3B and Qwen3.6-35B-A3B each fit on **64×H100 (8 nodes) at roughly 2.5 hours per step** — *"slower than I would like but not insane"* — at about **10M generated tokens per step**; Qwen3-Next-80B-A3B, the closest public analog to the incoming Marin 67B-A2B, is expected to need **96–128×H100**; and 131k is judged *"probably the limit of what we can achieve on this hardware without additional cleverness"* ([#7052](https://github.com/marin-community/marin/issues/7052), reported in [#reinforcement-learning](https://discord.com/channels/1354881461060243556/1524748716190339122/1524749076355088485)). That ~2.5 h/step is precisely the lever [#6709](https://github.com/marin-community/marin/issues/6709) exists to pull — and recall from [#7074](https://github.com/marin-community/marin/issues/7074) that a ~5× environment speedup divides the whole post-training bill by five.

---

## Tier 3 — Areas of investment (where the bulk of the work actually goes)

A contiguous block of epics [#6707](https://github.com/marin-community/marin/issues/6707)–[#6715](https://github.com/marin-community/marin/issues/6715), plus three July-specific epics. All carry the same charter line: *"Ongoing area of investment, picking up after the July Commitment and Hero Run work in this milestone"* ([Jul 6–12 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).

| Epic | Area | Where it stood in the Jul 6–12 window |
|---|---|---|
| [#6715](https://github.com/marin-community/marin/issues/6715) | Training & cluster infra / reliability | **Iris cross-cluster federation went live** — a job submitted to marin/marin-dev now runs *whole* on CoreWeave ([#7064](https://github.com/marin-community/marin/issues/7064)), with identity/authz/trust ([#7034](https://github.com/marin-community/marin/issues/7034)) and capacity-aware placement ([#7108](https://github.com/marin-community/marin/issues/7108)). Separately **the Executor is gone**: the eager import-time DAG retired for lazy `ArtifactStep`s, a net ~−14k lines across 276 files ([#6649](https://github.com/marin-community/marin/pull/6649)). |
| [#6714](https://github.com/marin-community/marin/issues/6714) | SFT data curation | **Axolotl landed as a first-class SFT backend** ([#6839](https://github.com/marin-community/marin/issues/6839)), closing a chat-template save footgun; Levanter "SFT with confidence" opened ([#7045](https://github.com/marin-community/marin/issues/7045)). |
| [#6713](https://github.com/marin-community/marin/issues/6713) | Pretraining data curation & mix | **SuperBPE's tokenizer edge reverses at scale** — 128k arms start level with Llama-3 and fall further behind as budget grows ([#6796](https://github.com/marin-community/marin/issues/6796)); and Marin's `high_quality` curation is **not Pareto-dominant** at compute-matched DCLM CORE ([#2351](https://github.com/marin-community/marin/issues/2351)). Both are negative results, recorded as such. |
| [#6712](https://github.com/marin-community/marin/issues/6712) | Data-selection diagnostics | A **preregistered content-embedding mixture surrogate priced a never-swept bucket and beat the sweep's own best-ever mixture** from a single embedding job ([#6969](https://github.com/marin-community/marin/issues/6969)). |
| [#6711](https://github.com/marin-community/marin/issues/6711) | Model architecture & scaling recipe (MoE) | GPU upskilling plan opened; H100 MoE weak-scales flat to 64 chips. |
| [#6710](https://github.com/marin-community/marin/issues/6710) | B200 training MFU & perf | The engine room behind commitment [#6706](https://github.com/marin-community/marin/issues/6706) — QuACK/SonicMoE, fp8, and pipeline parallelism, pushing toward **25% for September**. |
| [#6709](https://github.com/marin-community/marin/issues/6709) | Inference speed (for RL rollouts) | Rollout-speed epic opened, still at the **baseline-establishing** stage. A reported vLLM "wedge" turned out to be **a 4096-token NaN/HTTP-400 bug**, not a broker leak — the original diagnosis was withdrawn ([#6983](https://github.com/marin-community/marin/issues/6983)). |
| [#6708](https://github.com/marin-community/marin/issues/6708) | RL framework of the future | The **xorl reproduction was walked back under a same-ruler fidelity audit** (see caveat below); MarinSkyRL's 131k-context MoE RL mapped on CoreWeave ([#7052](https://github.com/marin-community/marin/issues/7052)). |
| [#6707](https://github.com/marin-community/marin/issues/6707) | RL data curation, experiments & ablations | Zero-RL on Delphi-25B: format beats plumbing; RL-data breakout kicked off. |
| [#6867](https://github.com/marin-community/marin/issues/6867) | July Grug Inference | DoD: *"Support full size GrugMoE model on both TPUs and GPUs. Stretch: Inference is fast enough on GPUs."* **0/3 sub-issues closed.** EP8 GrugMoE serving passed on CoreWeave/H100 ([#6042](https://github.com/marin-community/marin/issues/6042)); TPU-vLLM fork pins consolidated ([#7025](https://github.com/marin-community/marin/issues/7025)). |
| [#6863](https://github.com/marin-community/marin/issues/6863) | July Eval tasks | DoD: both Evalchemy and Harbor easily triggerable from Marin on TPUs. Agentic SWE evals validated on v6e TPUs at parity with the H100 reference (0.240 vs 0.237 on swebench-verified-random-100) ([#6958](https://github.com/marin-community/marin/issues/6958)); CI proposed to stop the harbor/evalchemy forks rotting ([#7044](https://github.com/marin-community/marin/issues/7044)). |
| [#6037](https://github.com/marin-community/marin/issues/6037) | datakit: July-hero release | New code sources onboarded; GHALogs ingest sharded. |

---

## Conflicts, caveats and what is *not* settled

These are the places where a confident summary would be wrong.

**1. The B200 run's size and cluster changed between sources — trust the later one.** The tracking issue [#6689](https://github.com/marin-community/marin/issues/6689) *"[Hero Run] nB-AmB XT on B200s"* (filed 2026-06-26) still carries a deliberately placeholder name (`nB-AmB`), lists **288 B200 (4×NVL72)**, and leaves FLOP budget, tokens, data mix, W&B and preregistered loss all as `?`. The Jul 6–12 summary describes the same run as **~120B-A8B kicking off on 2 NVL72** ([Jul 6–12 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)). The later framing is the current one, but note that per [#7073](https://github.com/marin-community/marin/issues/7073) the shape is **formally still open** — so "~120B-A8B" should be read as the working assumption, not a locked decision.

**2. The September and EOY run sizes are not reconciled with each other.** The Jul 6–12 summary says the September contender is **~512B-A16B on 20T tokens**; [#6689](https://github.com/marin-community/marin/issues/6689) describes the destination as the **"EOY 256–500B-AYB run"** on "the full 800+ B200 cluster". Nothing in the corpus reconciles the two, and `AYB` is an explicitly unresolved active-parameter count. Treat **512B-A16B / 20T as the latest and most specific** statement and 256–500B-AYB as the older, wider bracket — but do not present either as fixed.

**3. 2.277 vs 2.269 is not a passed test.** See the Tier-1 detail above. The 2T cooldown number is a favorable *early* signal measured under different conditions than the target; the 8T stage-1 preregistration remains live and unretired ([#6811](https://github.com/marin-community/marin/issues/6811)).

**4. Every MFU number belongs to its hardware.** ~18.6% is the **TPU v4-2048** 67B-A2B run; ~23.8% is **64×H100** on the 11B-A1.5B validation run; **17.8%** is **single-node 8×B200** at d5120. 20% and 25% are **targets**, not results. Do not carry any of these across platforms.

**5. An RL result was audited and substantially revised mid-milestone.** The cross-framework xorl reproduction of MarinSkyRL, which last week read as matching/exceeding reference math lifts (+12.3 MATH500 / +8.8 gsm8k) on a fraction of the hardware, was corrected on July 10 by Ashwinee Panda's fidelity audit: the original was **not a same-ruler comparison** (Muon vs AdamW, different prompt/grader pair, more permissive grader). Under the parity contract it lands at roughly **+0.8 to +1.4 MATH500** against the reference's +8.4, and +7.6 to +8.4 gsm8k against +4.4, with AIME regressing — total transfer broadly comparable (~+9–10 vs +12.8) but **allocated differently** ([#6915](https://github.com/marin-community/marin/issues/6915), [Jul 6–12 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).

**6. Corpus boundary.** The frozen GitHub thread corpus effectively ends around **#6967 / 2026-07-05**; issues numbered ~#7000+ are described in the Jul 6–12 weekly summary but are largely not retrievable as full threads. Claims here about #7012, #7024, #7034, #7044, #7052, #7064, #7073, #7074, #7108, #7116 and #7117 rest on the **weekly summary's narrative**, not on the underlying threads. Anything decided after **2026-07-16** is outside this corpus entirely — including, notably, whether [#7073](https://github.com/marin-community/marin/issues/7073) has since closed and whether B200 MFU cleared 20%.

---

## How to read the milestone

The shape is coherent once you see it: **two runs in flight that will not finish inside July**, so the milestone extracts value from them early via **intermediate cooldowns** ([#6811](https://github.com/marin-community/marin/issues/6811)) that hand a usable checkpoint to the post-training and inference stacks. Meanwhile **three commitments** de-risk the *next* run — can Blackwell go fast enough ([#6706](https://github.com/marin-community/marin/issues/6706)), what will post-training cost ([#7074](https://github.com/marin-community/marin/issues/7074)), and what exactly are we training ([#7073](https://github.com/marin-community/marin/issues/7073)) — and the investment epics keep the substrate (data, kernels, RL, inference, evals, cluster) moving underneath.

July is the month the team buys the right to start the September ~512B-A16B run on schedule.

---
<!--provenance-->
> *Data: marinmirror — 68026 chunks, built 45h ago · summaries through 2026-07-06_2026-07-12 (frozen eval corpus, pinned 2026-07-16; no refresh this run).*
>
> *Query: "explain our july 2026 plan"*
>
> *Sub-queries: "July 2026 milestone — stated goals, definition of done, dates, four workstreams" · "67B-A2B Grug 10T hero run on TPU v4-2048: progress, MFU, 2T/5T cooldowns, Paloma vs preregistered target" · "end-of-July B200 run (nB-AmB / ~120B-A8B on NVL72): architecture, data mix, preregistration, open decisions" · "July post-training plan: FLOPs budget, RL over experts, SFT data, rollout throughput" · "Blackwell B200/NVL72 MFU workstream: 17.8% measured, 20% bar, 25% September target, QuACK/SonicMoE, fp8, pipeline parallelism" · "why July matters for the rest of 2026: the ~512B-A16B September contender and the EOY 256–500B-AYB run" · "July supporting infrastructure and data investment areas: Iris federation, ArtifactSteps, mixture surrogate, SuperBPE, Axolotl SFT, TPU-vLLM, evals/CI"*
