# The July 2026 Plan

Marin's work is organized around dated GitHub **milestones**. The July plan lives in milestone #10, **"July milestone: complete 67B-A2B MoE; start XB-AYB MoE on B200s; start post-training Marin MoEs"** (due 2026-07-31, 18 open / 2 closed issues). That title is the plan in one line, and it sits between the June milestone (#9, "kick off overtrained 67B-A2B 10T") and the August milestone (#12). The whole thing is framed as being on the path to an **end-of-year 256–500B-AYB run**.

## The three headline goals

The milestone title names July's three deliverables:

1. **Complete the 67B-A2B MoE** — the June "hero run" training a 67B-total / ~2B-active MoE on 10T tokens.
2. **Start the next XB-AYB MoE on B200s** — the ~120B-A8B run that is the last checkpoint before training on the full 800+ B200 cluster.
3. **Start post-training Marin MoEs** — the first RL/SFT on a Marin-trained MoE.

## Structure: a Commitment, Hero Runs, then ongoing workstreams

The milestone is deliberately structured. Nearly every standing workstream issue says it "picks up **after the July Commitment and Hero Run work** in this milestone," giving the shape:

### A. The "Commitment" (the one must-hit bar)

- **#6706 — Get B200 MFUs above 20% in advance of the Aug 1 run.** The single hard bar for the month; the continuing perf investment (#6710) is explicitly gated behind it.

### B. Hero runs (the flagship training runs)

- **#6704 — Land June 67B-A2B run on TPUs** (owner @ClassicLarry). Tracking issue for the 10T-token run; realistically lands in August, with intermediate cuts.
  - **#6811 — Land July 67B-A2B intermediate cooldown on 2T tokens** (closed) — an intermediate cut of the 10T run.
- **#6689 — [Hero Run] ~120B-A8B XT on B200s** (owner @ClassicLarry). Stated purpose: "Prepare the next best model for post-training on the path to our EOY 256–500B-AYB run… **This is the last run before we train on the full 800+ B200 cluster.**" Runs on 288 B200 (4×NVL72); deliberately a de-risking run (expect scary MFUs, 1–3 node failures/day, rollback-from-checkpoint). Its July planning sub-issues are the key deliverables:
  - **#6700 — Data mix** for the pre–mid-training hero run
  - **#6701 — Architecture** for the July hero run (includes the long-context plan)
  - **#6702 — Preregister** the hero-run loss
  - **#6703 — Select/implement the Marin AA-II validation (eval) suite**
- **#6716 — Hardware & Loss Validation runs on H100s (11B-A1.5B @ 500B)** — a multinode-GPU proof-of-concept.
- **#6705 — Post-training on 67B-A2B 10T** (owner @penfever, 512+ H100 on CoreWeave). The first post-trained Marin MoE; realistically starts early/mid-August, listed here for July planning.
  - **#7170 — Post-training on the 67B-A2B 2T intermediate cut** — the earlier, smaller-cut post-training used to shake out bugs against the #6811 checkpoint.

### C. Ongoing investment workstreams (continue after the commitment/hero work)

Standing tracking issues, each with a "metric to improve":

- **#6707 — RL data curation, experiments & ablations** (coordinating with the "OT collective")
- **#6708 — RL framework of the future** — evaluate candidates, close parity gaps (#6341), pick the standard (guide: #6162)
- **#6709 — Inference speed for RL rollouts** — make RL not inference-bound, focused on H100s/vLLM
- **#6710 — B200 training MFU & perf** — continues past the #6706 bar
- **#6711 — Model architecture & scaling recipe (MoE)** — MoE + long-context research and the isoflop scaling recipe feeding #6701/#6702
- **#6712 — Data-selection diagnostics** — PPL/eval proxies that predict downstream quality
- **#6713 — Pretraining data curation & mix** — tune the mix, productionize the CoreWeave data pipeline
- **#6715 — Training & cluster infra / reliability** — goodput, orchestration, checkpointing

### D. July "tasks" epics (definition-of-done oriented)

- **#6863 — [Epic] July Eval tasks.** DoD: both Evalchemy and Harbor can be easily triggered from Marin on TPUs.
- **#6867 — [Epic] July Grug Inference tasks.** DoD: support the full-size GrugMoE model on both TPUs and GPUs; stretch: fast enough inference on GPUs.
- **#6037 — datakit: July-hero release.** DoD: decide new dataset inclusions (add CC POC crawl, more code data), fix known child issues, evaluate and produce the new mix.
- **#7273 — Canonical end-to-end reference pipeline** (raw sample → datakit → pretrain → eval) as a single change-validation harness.

### E. Sizing the next run (hand-off to August)

- **#7073 — Shape of model (arch + tokens) for the ~120B-A10B Aug 1 run** and **#7074 — approximate token upper bound needed for post-training** are the late-July sizing decisions that feed August.

## How to read it

July is a **bridge month**: *finish* the June 67B-A2B MoE (#6704, with the #6811 2T cut already closed), *clear the B200 MFU bar* (#6706) and stand up the ~120B B200 hero run (#6689) as the last dress rehearsal before the full 800+ B200 cluster, and *kick off the first MoE post-training* (#6705, #7170) — all while standing data/RL/infra/eval/inference workstreams (#6707–#6715, #6863, #6867, #6037, #7273) run underneath. Several "complete" items (the full 10T land, 10T post-training) are openly expected to spill into August, on the path to the EOY 256–500B-AYB run.

### Key references
- Milestones: #10 (July); adjacent #9 (June), #12 (August)
- Hero runs: #6689, #6704, #6705, #6716, #6811, #7170
- Hero-run planning sub-issues: #6700, #6701, #6702, #6703
- Commitment / perf: #6706, #6710
- Workstreams: #6707, #6708, #6709, #6711, #6712, #6713, #6715
- Epics / DoD: #6863, #6867, #6037, #7273
- Aug-1 sizing: #7073, #7074
