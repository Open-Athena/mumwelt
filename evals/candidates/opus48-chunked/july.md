# The July 2026 plan

## The frame

The milestone's own title states the three things July is for:

> **"July milestone: complete 67B-A2B MoE; start XB-AYB MoE on B200s; start post-training Marin MoEs"**
> — carried in the header of both the [week of June 29](https://mws.oa.dev/summaries/summary-2026-06-29_2026-07-05.html) and [week of July 6](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html) summaries

Two framing facts matter more than the title. First, **July is not a July-shaped month**: "The July milestone runs to **mid-August** and is a **preparation month, not a finish line**." Second, it is scoped by what comes *after* it ([week of July 6](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)):

> "The throughline for the rest of the year is the Marin 2026 contender: when the twelve NVL72 racks land around September 15, the team kicks off its largest hero run — a ~512B-A16B model on 20T tokens. **July's job is to be ready for that**, and for the ~120B-A8B B200 run that kicks off on 2 NVL72 at the end of the month, by finishing and cooling down the two ongoing runs, getting Blackwell model FLOPs utilization (MFU) up, locking in the data mix and architecture, and pricing post-training."

*Sourcing caveat:* the September-15 date, the twelve-rack count, and the 512B-A16B / 20T pairing appear **only in the weekly summary** — no GitHub or Discord primary in the corpus carries them. The nearest primary is [#6689](https://github.com/marin-community/marin/issues/6689), which says only "the path to our **EOY 256–500B-AYB run**" and "the last run before we train on the **full 800+ B200 cluster**." Treat the September specifics as plan-of-record narrative, not preregistered commitment.

**Four workstreams carry the milestone** (verbatim, [week of July 6](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)): "NVL72 pretraining MFU toward 25% for the September run; a pretraining recipe that sets up post-training well; post-training infrastructure that is fast enough (and a working definition of fast enough); and a post-training recipe for the best model the available FLOPs allow."

The work is organized in three tiers, each with its own charter: **Hero Runs** ("the concrete use of compute"), **Commitments** ("what we must land this milestone to be ready for the runs ahead"), and **Areas of Investment** ("in practice the bulk of the month's work").

---

## 1. Hero runs — two in flight

### 67B-A2B Grug MoE, 10T tokens, TPU v4-2048 ([#6704](https://github.com/marin-community/marin/issues/6704), logged on [#6044](https://github.com/marin-community/marin/issues/6044))

The centerpiece, inherited from the June milestone. **Achieved**, as of the week of July 6: crossed from ~1.47T to step 39,000 (~2.114T tokens, about 21% of the 10.07T horizon), holding **~18.6% MFU on TPU v4-2048** — up from ~13.5% after Larry Dial relaunched July 1 from the step-15k checkpoint with a planned batch doubling to 67.1M tokens ([week of June 29](https://mws.oa.dev/summaries/summary-2026-06-29_2026-07-05.html)).

**The 2T intermediate cooldown ([#6811](https://github.com/marin-community/marin/issues/6811)) is the month's most-cited result — and the one most easily misreported.** A 3,150-step LR decay branched from step 39,000 finished at step 42,149 (33.9h wall-clock) at **Paloma macro-loss 2.2772, bits-per-byte 0.8242** — down 0.109 (−4.6%) over ~211B cooldown tokens. Confirmed in the W&B run summary (`..._muon_cooldown_step39k`, state finished, TPU v4-2048) and by Will Held on July 12: *"Final loss - first cooldown is done!"*

The preregistered target it is compared against, quoted from Larry Dial on [#6044](https://github.com/marin-community/marin/issues/6044#issuecomment-4820008980):

> "Preregistered loss target for stage 1 of the run (first 8T tokens): **2.269 paloma macro loss when evaled at seqlen 8192 on 1024 sequences per eval.**"

**2.2772 does not retire that target, and the team says so explicitly.** The comparison is deliberately not apples-to-apples on three axes: the target is at **8T tokens** (this cut is at ~2.1T), at **seqlen 8,192** (the cooldown evaluates at 8× context, 65,536), and **before** the final LR cooldown (this one fully decays LR and switches to the phase-2 mix, which Larry notes "tends to have higher loss"). The summary's own section headline — "2T cooldown hits the stage-1 target" — is contradicted by its body: "So **it doesn't retire the 8T preregistration; it is a strongly favorable early signal.**" Read it as the latter.

Two further caveats from the participants themselves. Larry flagged that "the code eval is sensitive to the exact eval token count since its ordered by languages, so different `max_eval_batches` may make this not exactly apples to apples." And Will Held noted Marin's lowest-ever loss is **2.202 from the 32B model**, asking whether the cooldown could catch it in dramatically fewer FLOPs; Larry: "haha probably not. fun target tho." It did not.

**What the cooldown buys** is the point of the exercise, per [#6811](https://github.com/marin-community/marin/issues/6811): "enough to do **mid+SFT+RL and exercise our inference and post-training stack early** (so we're ready to run when the full 10T [#6704](https://github.com/marin-community/marin/issues/6704) run lands)." The step-42150 checkpoint was verified and copied to CoreWeave S3 and a v6e inference zone by Romain Yon, so post-training can start against a usable model months before the full run finishes. Note the acceptance criteria are still literal placeholders — "pass@256 on X Y Z evals."

**Timeline — weaker evidence than it looks.** The summary states completion is "targeted for around **August 30**" with "a second intermediate cooldown at **5T** planned for early August." Neither the August-30 date nor the 5T cooldown appears in any GitHub or Discord primary in the corpus. The primaries give: [#6704](https://github.com/marin-community/marin/issues/6704) — "This should likely land in August"; and Larry on June 29 — "It will likely take about **50 days to complete**" (a ~June 28 start + 50 days ≈ mid-August). A slip toward end-August is consistent with the July crashes and the ~30% tokens/s cost of the 65k-seqlen extension, but "August 30" is summary-attested only.

The context extension itself rode on a YaRN attention-scale probe: because the model sets `disable_long_rope` with 2,048-token sliding windows, the real lever at 8× context is a temperature scale on the query·key product (`qk_mult`). A 6-arm sweep picked all-layers coef=0.1. Larry flagged the cost plainly: longer context "helps on the evals immediately" but costs **~30% in tokens/second** from quadratic attention.

### Marin 11B-A1.5B validation run, 64×H100 ([#6716](https://github.com/marin-community/marin/issues/6716))

**Achieved: ~23.8% MFU** (~1.37M tokens/s, ~1.53 s/step at batch 512 × seq 4096) on 64×H100 on CoreWeave `cw-us-east-02a`, with completion expected July 13–14. A d2048 MoE, 24 layers, 64 experts top-4, GQA 4:1, ~1.53B active / 10.6B total.

The artifact is explicitly not the point — the run is pushed to 500B tokens, well past compute-optimal, to shake out the GPU stack. Isaac Hodes: *"this is more about exercising the hardware instead of delivering an artifact."* **Two cautions:** the FLOP counter treats attention as fully quadratic and ignores the 2048-token sliding window, so 23.8% is "slightly optimistic on the attention term"; and this is an **H100** number — it is *higher* than the B200 figure below and says nothing about Blackwell readiness.

---

## 2. Commitments — "what we must land to be ready for the runs ahead"

### B200 MFU above 20% ([#6706](https://github.com/marin-community/marin/issues/6706)) — NOT MET

This is the gate on the end-of-month run: "B200 model FLOPs utilization (MFU) must clear 20% before it kicks off on 2 NVL72. **It is not there yet**, but the gap is closing."

**Achieved: 17.8% single-node on 8×B200 at d5120** (14.9% at d2560), reached by wiring Tri Dao's QuACK/SonicMoE grouped GEMM into the expert MLPs via a torch-free CUTLASS shim — the QuACK gated GEMM alone at ~1,175 TFLOP/s, ~52% of peak ([#7012](https://github.com/marin-community/marin/issues/7012), via summary). The diagnostic that unlocked it: **batch is a dead lever** (the workload is arithmetic-intensity-starved, not batch-starved) while **width is live**. Quote the width qualifier — "17.8%" bare overstates the general case.

**20% and 25% are both targets.** 25% is the September bar for the ~512B-A16B run, tracked in [#6710](https://github.com/marin-community/marin/issues/6710), whose "Metric to improve" is literally "Training MFU % (≥20% per #6706)". **No NVL72-measured MFU exists anywhere in the corpus** — every Blackwell number is single-node 8×B200.

Two levers in flight, both still short:
- **fp8** — at the production d2560/E256/K4 shape the full MoE layer measures **1.53× vs bf16 at 2-node EP16 over InfiniBand** — but that is an **H100** measurement, and "**defaults stay bf16**" pending loss-curve validation on a real trajectory ([#6911](https://github.com/marin-community/marin/issues/6911#issuecomment-4878649652)).
- **Pipeline parallelism** — JaxPP PP MoE training across four 8×H100 nodes reached **mean MFU 18.26% at ~414k tokens/s**, "about 1.74 points below the 20 bar" — again **H100, not B200**, followed by a week of negative results.

### Post-training cost ([#7074](https://github.com/marin-community/marin/issues/7074)) — ANSWERED

Benjamin Feuer filled this in — the one commitment that moved to an answer this month. Costed in hardware FLOPs (so RL's low ~1% aggregate MFU is counted rather than hidden), the pipeline is **RL over the experts, then multi-teacher on-policy distillation (MOPD) into the student**, and is dominated by the RL leg, whose cost scales with the experts' hosting footprint:

> "post-training a small 67B-A2B hero costs roughly **0.25–0.9× its 10T pretraining in HW-FLOPs** (about 1.2–3.9× the cheaper 2T-cut), and **only ~5–20% of the 120B big run**; a ~5× environment speedup divides all of it by five."

In literal tokens it is tiny — order 10⁹–10¹⁰, under 0.5% of a 10T pretrain — "though that undercounts, since each post-training token is far costlier to produce."

**This is an estimate, not a measurement**: the epic is titled "Approximate tokens (**upper bound**) needed for post-training," the summary calls it "penciled," and the ~1% RL MFU and 5× environment speedup are assumptions feeding the model. Note the scope split — **0.25–0.9× applies to the 67B-A2B, 5–20% to the 120B run**; conflating them would be wrong by an order of magnitude.

### Shape, data mix and preregistration for the end-of-month run ([#7073](https://github.com/marin-community/marin/issues/7073)) — STILL OPEN

> "This commitment is to lock in the data mix, the architecture, and a preregistered loss target for the ~120B-A8B B200 run that kicks off at the end of the month on 2 NVL72. **It is still open — no decision was recorded this week** — but its inputs are actively in motion: the B200 MFU bar [#6706](https://github.com/marin-community/marin/issues/6706), the post-training cost estimate [#7074](https://github.com/marin-community/marin/issues/7074), and the architecture and data-mix work under the investment epics."

4 of 14 sub-issues closed; 11 comments, 0 PRs in the latest week. "Still open" is right; "no progress" would not be.

The hero-run stub [#6689](https://github.com/marin-community/marin/issues/6689) shows how open: FLOP budget, tokens, data mix, and pre-registered loss are all literally `?`, and the title is still the placeholder "nB-AmB XT". Its four sub-issues — [#6700](https://github.com/marin-community/marin/issues/6700) (data mix), [#6701](https://github.com/marin-community/marin/issues/6701) (architecture), [#6702](https://github.com/marin-community/marin/issues/6702) (preregister loss), [#6703](https://github.com/marin-community/marin/issues/6703) (eval selection) — are bare stubs.

Where the inputs stand:
- **Architecture**: the working call is **GQA**, after FLOP accounting went against Multi-head Latent Attention — MLA on every layer costs +17–20% training FLOPs at 4k seqlen and balloons at long context ([#6522](https://github.com/marin-community/marin/issues/6522)). Explicitly not finalized: [#6889](https://github.com/marin-community/marin/issues/6889) was opened to measure MLA's MFU at the d5260 target size first.
- **Data mix**: gated operationally on the **datakit July-hero release** ([#6037](https://github.com/marin-community/marin/issues/6037), 4/13 sub-issues closed) — #7073's sub-issue set is essentially the datakit cluster. That cluster carries nine open, well-characterized defects filed July 2, including quality scoring judging long documents on a truncated 4KB lead ([#6859](https://github.com/marin-community/marin/issues/6859)), dedup wiping near-all of many sources — "median of ~76% of documents per source, with 15 of 113 sources over 95% removed" ([#6854](https://github.com/marin-community/marin/issues/6854)) — and decontam flagging only ~0.01% of documents ([#6852](https://github.com/marin-community/marin/issues/6852)).
- **Tokenizer**: no decision. The SuperBPE-128k recommendation was **walked back at scale** — [#6796](https://github.com/marin-community/marin/issues/6796) closed after "the win is budget-dependent and **reverses at scale**", leaving only a trained 64k-fixed arm ahead, and only at low budget.
- **A validated new tool, not yet adopted**: the preregistered embedding-based mixture surrogate ([#6969](https://github.com/marin-community/marin/issues/6969)) priced a never-swept bucket and beat the sweep's own best-ever run — 0.9410 vs 0.9554 uncheatable BPB, roughly 8σ — from one embedding job instead of a re-sweep. Its own preregistered caveat: the optimized point was optimistic by ~0.031 BPB (winner's curse).

---

## 3. Areas of investment — "in practice the bulk of the month's work"

Nine standing epics, each filed June 26 with a named metric — among them cluster infra/reliability ([#6715](https://github.com/marin-community/marin/issues/6715), "Goodput % / interruption-recovery time"), inference speed for RL rollouts ([#6709](https://github.com/marin-community/marin/issues/6709), "Rollout throughput (tok/s) / generation latency"), RL framework selection ([#6708](https://github.com/marin-community/marin/issues/6708), "Decision, not a metric"), and SFT data curation ([#6714](https://github.com/marin-community/marin/issues/6714), "IFEval lift"). The headline items:

- **Iris cross-cluster federation went live** — a job submitted to marin or marin-dev now runs whole on CoreWeave and is watched from where it was launched ([#7064](https://github.com/marin-community/marin/issues/7064)), on identity/authorization/trust layers ([#7034](https://github.com/marin-community/marin/issues/7034)) with placement moved off the submit path so federated jobs queue until a peer reports free capacity ([#7108](https://github.com/marin-community/marin/issues/7108)). The mechanism landed; I found **no attestation of a production hero run actually federating**, and cross-cluster spend caps are a known open hole ([#6843](https://github.com/marin-community/marin/issues/6843)) — a user "can therefore exceed their intended cap by fanning work out to peers."
- **The Executor is gone** ([#6649](https://github.com/marin-community/marin/pull/6649)) — a net ~−14k lines across 276 files, replacing eager import-time `ExecutorStep` globals with lazy, explicitly versioned `ArtifactStep`s, so a code edit no longer silently forks an output path.
- **Rollout speed is at the baseline-establishing stage, by its own admission.** [#6709](https://github.com/marin-community/marin/issues/6709) says "**First establish the baseline(s), then optimize against an agreed target**" and its open question is still "What are we targeting". The GrugMoE inference DoD is literally unfilled: "Fast enough: **TODO: Within X% of Jax on GPUs vs. within Y% of reference model**" ([#6870](https://github.com/marin-community/marin/issues/6870)). Against that, Feuer measured Qwen3 30B-A3B / Qwen3.5 35B-A3B fitting on 64×H100 at **~2.5hr/step** — "slower than I would like but not insane."
- **An honest walk-back**: the xorl repro's headline gains were re-audited on a same ruler and fell from "+12.3 MATH500" to **+0.8 to +1.4** ([#6915](https://github.com/marin-community/marin/issues/6915), via summary).

---

## What to watch / where this could slip

1. **The B200 MFU gate is the binding constraint.** 17.8% vs a 20% bar, with the two big levers (fp8, pipeline parallelism) both still measured on H100 and both below bar. If it doesn't clear, the end-of-month run either slips or goes ahead off-target.
2. **#7073 is the long pole and it is stalled on datakit.** Locking data mix + architecture + preregistration is the stated commitment, no decision is recorded, and its sub-issue set is the datakit defect cluster.
3. **An unreconciled 2× discrepancy in the run's compute.** The summary says the end-of-month run kicks off on **2 NVL72**; [#6689](https://github.com/marin-community/marin/issues/6689)'s key-facts table says **"288 B200 (4xNVL72) on CoreWeave cluster"**. Nothing in the corpus reconciles them. #6689 is the weaker source (a June stub with blank fields, and the "~120B-A8B" size appears only in the summary), but this is inference, not a recorded decision.
4. **Date framing collision**: the epic titles for #6706 and #7073 both say "**Aug 1** run" while the prose says "end of the month" run. Same run.
5. **The most likely thing to be misreported**: the 2T cooldown's 2.2772 as "hit the preregistered target." It didn't — different token count, context length, mix, and LR state.

## Gaps in the corpus

The frozen mirror's GitHub leg thins out around **2026-07-04**, so **#7073, #7074, #7012, #6969, #7124** and others return "not found in corpus" and are attested by the weekly summary alone. The EOY post-training plan lives in Google Docs linked from [#6053](https://github.com/marin-community/marin/issues/6053), outside the corpus. And the September-15 / twelve-rack / 512B-A16B / 20T specifics have no primary corroboration — they should be repeated as plan narrative, not as commitment.

---
<!--provenance-->
> *Data: marinmirror — 86367 chunks, built 0h ago · summaries through 2026-07-06_2026-07-12 (frozen eval corpus `2026-07-18-chunked`; no refresh this run).*
>
> *Query: "explain our july 2026 plan"*
>
> *Sub-queries: "July 2026 milestone definition, scope, goals and dates" · "67B-A2B Grug MoE 10T hero run status, 2T intermediate cooldown, Aug 30 completion" · "B200 MFU above 20% bar, QuACK/SonicMoE, fp8, pipeline parallelism, NVL72 toward 25%" · "post-training plan: RL over experts, MOPD, HW-FLOPs cost, rollout speed, SFT data" · "shape of model, data mix and preregistered loss for the ~120B-A8B B200 run; datakit July-hero release" · "September 512B-A16B / 20T hero run, twelve NVL72 racks, EOY 256–500B-AYB goal" · "areas of investment: Iris federation, Executor retirement, cluster reliability, evals" · adversarial verification pass on nine load-bearing claims.*
