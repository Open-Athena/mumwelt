# July 2026 plan

July is best read as a **preparation and handoff milestone**, not as a promise that every large run finishes inside the calendar month. The plan is to (1) keep the 67B-total/2B-active, 10T-token Grug MoE training and create an intermediate checkpoint usable by post-training teams, (2) ready and start the last B200-scale stepping-stone run before the full 800+ B200 campaign, and (3) make the pretraining, serving, evaluation, SFT, and RL stack ready for the much larger end-of-year model. The full 67B run was already expected to finish only in August, and its final post-training was explicitly described as an early/mid-August activity [#6705](https://github.com/marin-community/marin/issues/6705), [#6044](https://github.com/marin-community/marin/issues/6044).

## The two hero-run tracks

The first track is the ongoing **67B-A2B MuonH MoE on TPU v4-2048**, sized at about 10.07T tokens. Its stage-1 target is a preregistered Paloma macro loss of 2.269 at 8T tokens, evaluated at sequence length 8,192 on 1,024 sequences; that is a target, not a July result [#6044](https://github.com/marin-community/marin/issues/6044). July's practical deliverable is an intermediate cooldown checkpoint so midtraining, SFT, RL, inference, and evaluation can proceed before the full run lands. The original issue said “<10T, assume 1T-ish,” so the exact cut was initially provisional; the source defines success in terms of a checkpoint useful enough to exercise the downstream stack, not merely token count [#6811](https://github.com/marin-community/marin/issues/6811).

The second track is a **B200 hero run** whose purpose is to produce the next-best post-trainable model and de-risk the final full-cluster campaign. It is explicitly the last run before using the 800+ B200 fleet; architecture, scaling recipe, data, and long-context choices are inputs rather than already-set facts [#6689](https://github.com/marin-community/marin/issues/6689), [#6701](https://github.com/marin-community/marin/issues/6701), [#6700](https://github.com/marin-community/marin/issues/6700). A gating commitment was to get B200 training above roughly **20% MFU** before launch, but the tracking issue itself phrases that bar as “Need to be at 20%+?”—so 20% should be reported as a planning bar, not as an achieved number [#6706](https://github.com/marin-community/marin/issues/6706). Ongoing kernel, FP8, communication, and parallelism work continues after that gate [#6710](https://github.com/marin-community/marin/issues/6710).

## What has to become repeatable

The plan deliberately separates a hero run from the reusable capabilities that make later hero runs possible:

- **Architecture and scaling:** settle the MoE architecture, long-context approach, and isoflop/scaling recipe used to size and forecast runs [#6711](https://github.com/marin-community/marin/issues/6711).
- **Pretraining data:** curate the mixture and productionize the CoreWeave data path [#6713](https://github.com/marin-community/marin/issues/6713). The associated diagnostics effort is not “pick one magic score”; it aims to strengthen PPL/eval signals that predict downstream quality [#6712](https://github.com/marin-community/marin/issues/6712).
- **Post-training data and experiments:** build SFT datasets [#6714](https://github.com/marin-community/marin/issues/6714), and curate RL data plus controlled ablations of what drives gains [#6707](https://github.com/marin-community/marin/issues/6707).
- **Serving and evaluation:** use the intermediate 67B checkpoint to shake out inference, SFT, RL, and evaluation before the final checkpoint arrives [#6811](https://github.com/marin-community/marin/issues/6811). This is a readiness goal, not evidence that the final post-trained model already exists.

## Current-state caveat

The July plan changed underneath the original June framing: the 67B job moved from the anticipated GPU route to TPUs [#6044](https://github.com/marin-community/marin/issues/6044). The same thread records why calendar and throughput estimates were still fluid—v4-1024 was efficient but projected about 78 days, while v4-2048 needed larger batches and replica sharding; a Muon Newton–Schulz sharding bug also had to be fixed before stable scale-up. Those are execution details supporting the central conclusion: **July's success criterion is de-risking and handoff toward the August and full-B200 runs, not claiming the entire 10T/post-training program is done in July** [#6044](https://github.com/marin-community/marin/issues/6044).

### Source/gap notes

The frozen corpus supports the workstream structure and explicit gates. Some planning issues are intentionally skeletal, so exact B200 model size, launch date, and final post-training compute budget are omitted rather than inferred.

---
<!--provenance-->
> *Data: marinmirror frozen 2026-07-16 — 68,026 chunks · summaries through 2026-07-06_2026-07-12 (no refresh).*
>
> *Query: “explain our july 2026 plan”*
>
> *Sub-queries: “July milestone and hero-run structure” · “67B-A2B intermediate cooldown and final timing” · “B200 hero run architecture/data/preregistration” · “B200 MFU gate and performance work” · “SFT/RL/post-training readiness” · “data diagnostics and pretraining mix” · “temporal changes and achieved-vs-target checks”*
