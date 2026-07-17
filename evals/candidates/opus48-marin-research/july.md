# Marin's July 2026 plan

## The one-line milestone

The July milestone is stated verbatim at the top of the weekly summaries as: **"complete 67B‑A2B MoE; start XB‑AYB MoE on B200s; start post‑training Marin MoEs."** Crucially, the team frames July as **a preparation month, not a finish line** — "The July milestone runs to mid‑August and is a preparation month." Its whole purpose is to get ready for the year's big runs, not to ship a finished artifact ([summary week of 2026‑07‑06](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).

## What July is preparing *for*

The throughline for the rest of 2026 is the **Marin 2026 contender / hero run**: when the **twelve NVL72 (B200) racks land around September 15**, the team kicks off its largest hero run — a **~512B‑A16B model on 20T tokens**. July's job is to be ready for that September run (and for a ~120B‑A8B B200 run at the end of July) by finishing/cooling down the two in‑flight pretraining runs, getting Blackwell MFU up, locking data + architecture, and pricing post‑training ([summary week of 2026‑07‑06](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)). The B200 run is explicitly "the last run before we train on the full 800+ B200 cluster," on the path to an EOY 256–500B‑AYB model ([#6689](https://github.com/marin-community/marin/issues/6689)).

## Four workstreams (the structure of the plan)

The month is organized as four workstreams ([summary week of 2026‑07‑06](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)):

1. **NVL72 (B200) pretraining MFU toward 25%** for the September run.
2. **A pretraining recipe that sets up post‑training well.**
3. **Post‑training infrastructure that is fast enough** (and a working definition of "fast enough").
4. **A post‑training recipe** for the best model the available FLOPs allow.

These map onto two concrete tracks: **hero runs** (the actual use of compute) and **commitments** (what must land to be ready), with a large body of **investment‑area** work underneath.

## Hero‑run track: two pretraining runs in flight

- **67B‑A2B Grug MoE on TPU v4‑2048, 10T tokens** — the centerpiece run, tracked in [#6704](https://github.com/marin-community/marin/issues/6704) and logged day‑to‑day on [#6044](https://github.com/marin-community/marin/issues/6044); completion is **targeted around August 30** (so "complete 67B‑A2B" spills past July into mid‑August). By the week of July 6 it had crossed ~2.1T of its 10.07T tokens at ~18.6% MFU. Its first **intermediate cooldown at 2T tokens** ([#6811](https://github.com/marin-community/marin/issues/6811)) closed at **2.277 Paloma macro‑loss** — a *strongly favorable early read* against the preregistered **2.269 stage‑1 target** (measured at the 8T / 80% mark), though the summary is careful that this is **not apples‑to‑apples** (the 2T cut fully decays LR, uses the phase‑2 mix, and evaluates at 8× context) so it does **not** retire the 8T preregistration. The intermediate checkpoint was published to CoreWeave + a v6e inference zone so mid‑training, SFT, RL, and inference can start early; a **second cooldown at 5T is planned for early August** ([#6811](https://github.com/marin-community/marin/issues/6811), loss‑registration close‑out [#6046](https://github.com/marin-community/marin/issues/6046)).
- **End‑of‑month ~120B‑A8B B200 run on 2 NVL72** — the "start XB‑AYB MoE on B200s" half of the milestone ([#6689](https://github.com/marin-community/marin/issues/6689)). Its data mix, architecture, and preregistered loss target were **still open** as of the latest week — no decision recorded yet — pending the MFU and post‑training inputs below ([#7073](https://github.com/marin-community/marin/issues/7073)).
- **11B‑A1.5B validation run on 64×H100** — a hardware‑and‑loss dry run for the Blackwell/GPU path, deliberately overtrained to 500B tokens to exercise the GPU stack (not to deliver an artifact); it cleared **~23.8% MFU** ([#6716](https://github.com/marin-community/marin/issues/6716)).

## Commitments (what must land to be ready)

- **B200 MFU above 20%** before the end‑of‑month run kicks off ([#6706](https://github.com/marin-community/marin/issues/6706)). *Achieved so far: 17.8% single‑node on 8×B200* via Tri Dao's QuACK/SonicMoE grouped kernel — **not yet at the 20% bar**, with fp8 and pipeline‑parallel work continuing ([#6706](https://github.com/marin-community/marin/issues/6706), kernel wiring [#7012](https://github.com/marin-community/marin/issues/7012)). The **25% target is for the September run** and is tracked separately in [#6710](https://github.com/marin-community/marin/issues/6710).
- **Price post‑training.** Benjamin Feuer penciled post‑training at **~0.25–0.9× the 10T pretrain in hardware FLOPs** (dominated by RL over the experts), and only ~5–20% of the 120B run; in literal tokens it's tiny (<0.5% of a 10T pretrain) ([#7074](https://github.com/marin-community/marin/issues/7074)). This is a *target/estimate*, not a measured result. The broader EOY post‑training plan question is tracked in [#6053](https://github.com/marin-community/marin/issues/6053), with the 67B‑A2B post‑training run itself scoped in [#6705](https://github.com/marin-community/marin/issues/6705) (realistically full‑10T post‑training won't begin until early/mid August).
- **Lock data + architecture + preregistered loss target** for the B200 run ([#7073](https://github.com/marin-community/marin/issues/7073)), fed by the datakit July‑hero release ([#6037](https://github.com/marin-community/marin/issues/6037)) and the architecture/data‑mix work below.

## Investment areas (where the bulk of the month's effort actually goes)

- **Cluster infra / reliability** — **Iris cross‑cluster federation went live**, so marin can hand *whole jobs* to CoreWeave and watch them from where they were launched ([#7064](https://github.com/marin-community/marin/issues/7064), identity/trust layer [#7034](https://github.com/marin-community/marin/issues/7034), epic [#6715](https://github.com/marin-community/marin/issues/6715)). The eager import‑time **Executor was retired for lazy ArtifactSteps** (~−14k lines) ([#6649](https://github.com/marin-community/marin/issues/6649)).
- **GPU upskilling for the September large run** — a **GPU Upskilling Master Plan** ([#6998](https://github.com/marin-community/marin/issues/6998)) and **GPU MFU Learning Path** ([#6979](https://github.com/marin-community/marin/issues/6979)); pipeline parallelism (JaxPP MoE, "Hopper is back on the menu") is the named real fix ([#7024](https://github.com/marin-community/marin/issues/7024)), plus end‑to‑end fp8 wiring ([#7079](https://github.com/marin-community/marin/issues/7079)). H100 MoE weak‑scales roughly flat to 64 chips (26.7%→26.5%) at 64 total experts ([#6711](https://github.com/marin-community/marin/issues/6711)).
- **Pretraining data + architecture** — data‑mix and tokenizer work under [#6713](https://github.com/marin-community/marin/issues/6713); a content‑embedding **mixture surrogate** that priced a never‑swept bucket and beat the sweep's best ([#6969](https://github.com/marin-community/marin/issues/6969)); MoE architecture/scaling recipe [#6711](https://github.com/marin-community/marin/issues/6711).
- **Post‑training recipe/infra** — SFT‑data regroup with an **Axolotl SFT backend** ([#6839](https://github.com/marin-community/marin/issues/6839)) and Levanter "SFT with confidence" ([#7045](https://github.com/marin-community/marin/issues/7045)); **RL framework** selection ([#6708](https://github.com/marin-community/marin/issues/6708)) with MarinSkyRL now running 131k‑context RL on MoE models on CoreWeave ([#7052](https://github.com/marin-community/marin/issues/7052)); a **rollout‑speed / inference epic** to keep RL from being inference‑bound ([#6709](https://github.com/marin-community/marin/issues/6709)) and the July Grug inference epic to serve the full GrugMoE on TPU + GPU ([#6867](https://github.com/marin-community/marin/issues/6867)).

## Bottom line

July 2026 is a **readiness / preparation month** whose stated milestone is to finish the 67B‑A2B MoE (landing ~Aug 30), start the ~120B‑A8B B200 MoE at end‑of‑month, and start post‑training Marin MoEs — all in service of the **September ~512B‑A16B, 20T‑token hero run** on the twelve incoming NVL72 racks. Progress is genuine but incomplete as of the latest week: the TPU hero run's 2T cooldown read favorably (2.277 vs the 2.269 target, caveated as not apples‑to‑apples), the H100 validation cleared ~23.8% MFU, but **B200 MFU is still at 17.8% (below its 20% commitment bar)** and the **B200 run's data/architecture/preregistration remain undecided**.

---
<!--provenance-->
> *Data: marinmirror — 68026 chunks, built 1h ago · summaries through 2026-07-06_2026-07-12 (frozen 2026-07-16 eval corpus; refresh disabled).*
>
> *Query: "explain our july 2026 plan"*
>
> *Sub-queries: "July 2026 milestone plan" · "September hero run 512B-A16B NVL72" · "post-training recipe/plan for hero MoE" · "B200 end-of-month 120B-A8B run architecture & data mix" · "67B-A2B TPU hero run 10T + intermediate cooldowns" · "B200 MFU commitment toward 20/25%"*
