# Inference at Marin: who, where, performance, and open issues

**Short version.** Inference work splits into two tracks with different owners. (1) A **correctness/enablement track** — getting the GrugMoE model to serve in vLLM on both TPUs and GPUs — is led by **Romain Yon** and is mostly *done* for the current checkpoint size (TPU path landed and closed; GPU EP8 serving validated). (2) A **speed track** — making rollout/generation inference fast enough that RL isn't inference-bound — is owned by **Isaac Hodes** as a forward-looking investment epic that is still at the "establish a baseline" stage, with **Benjamin Feuer**'s MarinSkyRL RL stack providing the only concrete throughput numbers so far. Serving perf today is "good enough for evals," not yet tuned for serious RL-scale inference.

## Who owns what

- **Romain Yon (`yonromai`)** owns the **[Epic] July Grug Inference tasks (#6867)** — "Support full size GrugMoE model on both TPUs and GPUs. Stretch: Inference is fast enough on GPUs." He also filed and drove the underlying support issues for GPU (#6042) and TPU (#6041), and ran the GPU expert-parallel validation (#6891). [#6867](https://github.com/marin-community/marin/issues/6867), [#6042](https://github.com/marin-community/marin/issues/6042), [#6041](https://github.com/marin-community/marin/issues/6041#issuecomment-4869501446)
- **Isaac Hodes (`ihodes`)** opened and owns **Inference speed (for RL rollouts) (#6709)** — "Speed up rollout / generation inference so RL is not inference[-bound]… Focus on GPUs, H100s in particular, where we intend do most of our RL this year." [#6709](https://github.com/marin-community/marin/issues/6709)
- **Benjamin Feuer** owns the **MarinSkyRL** RL-inference bring-up on GPUs (131k-context RL on MoE models), which is where the current real inference load and throughput numbers come from. [#7052 (via Discord)](https://discord.com/channels/1354881461060243556/1524748716190339122/1524749076355088485), [#6335](https://github.com/marin-community/marin/issues/6335)
- **Will Held** did the serving-reliability debugging (#6983) and added inference-side **logit-mixing** plumbing (a teacher/student mixed-logit server) in #7113. [#6709 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)
- **Rohith Kuditipudi** carries the TPU vLLM fork patches; **Russell Power** handled fork synchronization (#7097) and `marin-serve` packaging questions. [summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)

## Where it's at (status by track)

**TPU serving — landed and closed.** GrugMoE inference on TPU via the Marin-owned `vllm`/`tpu-inference` forks closed as **completed** on July 2 (#6041), landing in **#6664** with an export path, fork pins, and a real-checkpoint v6e e2e test comparing vLLM against the Levanter/JAX reference. This week the TPU stack was in **consolidation** — advancing fork pins to landed `marin-community` main SHAs and synchronizing Rohith's overlay patches (#7025, #7094, #7097). [#6041](https://github.com/marin-community/marin/issues/6041#issuecomment-4869501446), [summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)

**GPU serving — correctness validated at EP8.** The GPU support issue (#6042, DoD "loaded and used for inference in vLLM on GPUs; correctness tested, perf okay for evals") **closed** after **eight-way expert-parallel (EP8)** serving of a real GrugMoE checkpoint passed on CoreWeave/H100 (validation PR #6891, restacked into #6966). Remaining full-size and RL-grade work moved onto sub-issues #6869 (correctness on largest checkpoint) and #6870 (perf). [#6891](https://github.com/marin-community/marin/issues/6891#issuecomment-4887307764), [#6869](https://github.com/marin-community/marin/issues/6869), [#6870](https://github.com/marin-community/marin/issues/6870)

**RL rollout inference (MarinSkyRL) — plumbed, running under real load.** 131k-context RL on MoE models is now running on CoreWeave H100s (#7052). This is the live consumer of inference throughput. [#7052 (Discord)](https://discord.com/channels/1354881461060243556/1524748716190339122/1524749076355088485)

**Rollout-speed epic (#6709) — still baseline-establishing.** Its stated first task is to measure/document the current serving baseline before a target is agreed; no sub-issues, PRs, or design docs landed against it in the latest week — activity was one reliability bug (#6983) and Discord signals about where RL step-time goes. [summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)

## Current performance (with hardware/config attached — verified against primaries)

- **RL rollout throughput (H100, MoE):** Qwen3-Coder-30B-A3B and Qwen3.5-35B-A3B each fit on **64×H100 (8 nodes)** at **~2.5 hours per step**, generating **~10M tokens per step**, at **131k context**. Feuer's own words: *"~2.5hr / step at the moment which is slower than I would like but not insane."* 131k is judged "about the limit of this hardware without additional cleverness." The closest analog to the incoming Marin 67B-A2B (Qwen3-Next-80B-A3B) is expected to need **96–128×H100**. [Discord / #7052](https://discord.com/channels/1354881461060243556/1524748716190339122/1524749076355088485)
- **TPU inference (Levanter reference vs vLLM):** David Hall's Levanter-inference optimization made it **~10x faster**, landing **within ~12% of vLLM on TPU** — "good enough for evals" but vLLM is still wanted (and is effectively a hard requirement for SkyRL). [#6041](https://github.com/marin-community/marin/issues/6041#issuecomment-4869501446)
- **GPU GrugMoE serving:** correctness at EP8 **passed** on CoreWeave/H100; the DoD for #6042 was only "perf okay (good enough for evals)" — a *serious*-inference/RL-grade perf bar is explicitly out of scope there and deferred to #6870. **No agreed serving throughput baseline or target exists yet** for the rollout-speed epic. [#6042](https://github.com/marin-community/marin/issues/6042), [#6870](https://github.com/marin-community/marin/issues/6870), [#6709](https://github.com/marin-community/marin/issues/6709)

*Caveat:* #6870's "fast enough" DoD is still a **TODO** ("Within X% of Jax on GPUs vs. within Y% of reference model") — the performance target itself is undefined. [#6870](https://github.com/marin-community/marin/issues/6870)

## Open issues / risks

- **#6709 (rollout-speed epic):** still open, no baseline documented, no perf target agreed. The whole speed track hinges on this. [#6709](https://github.com/marin-community/marin/issues/6709)
- **#6870 (GPU perf "decent"):** open; DoD not yet quantified. [#6870](https://github.com/marin-community/marin/issues/6870)
- **#6869 (full-size GrugMoE GPU correctness):** open — EP8 passed on a checkpoint, but the largest-checkpoint correctness bar remains. [#6869](https://github.com/marin-community/marin/issues/6869)
- **#6983 (brokered vLLM "wedge"):** the field wedge (brokered clients stalling >30 min against an idle, healthy engine) is **genuine but unreproduced**; the investigated repro turned out to be a distinct **4096-token position-limit NaN → HTTP 400** bug (`Out of range float values are not JSON compliant: nan`, masked by `VLLM_ALLOW_LONG_MAX_MODEL_LEN`), flagged as possibly deserving its own issue. [summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)
- **#7117 (vLLM-compatible BF16 HF export regression):** **still open** — the 134 GB / 39-shard export verified by whole-tree SHA256, but reruns of the inference assertion hit an **NCCL `ncclAlltoAll` error** on the JIT path. [summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)
- **#7085 (TPU RPA v3):** ragged-paged-attention v3 raises `NotImplementedError: Unsupported tpu_version=4` on v4-8 workers, so Rohith **can't run inference there at all**; minimal fix is a `case 4` fallback in the block-size heuristic. [summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)
- **`marin-serve` GPU gaps:** #7111/#7133 (`marin-serve --gpu` can't boot vLLM cleanly — TPU/GPU extras and torch pins collide; the per-job `uvx` workaround still needs a real H100 boot test) and #7106/#7107 (`marin-serve` fails cryptically outside a workspace checkout). [summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)
- **Fork-churn operational cost:** bumping the vLLM fork pins triggered `uv` git-lock timeouts and a missing `zephyr` module when spinning up many workers in parallel. [summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)

## Gaps in this answer

Several of the newest July issues (#6983, #7052, #7085, #7113, #7117) are referenced and summarized in the frozen weekly summary but were **not individually retrievable as issue bodies** in this frozen corpus, so their finer details rest on the weekly-summary narration rather than the primary thread. The core throughput numbers (2.5h/step, 10M tokens/step, 64×H100) and the TPU/GPU close-outs are grounded in primary Discord/issue sources cited above.

---
<!--provenance-->
> *Data: marinmirror — 68026 chunks, built 1h ago · summaries through 2026-07-06_2026-07-12 (15 weeks). No refresh (frozen eval corpus).*
>
> *Query: "Who is doing inference work and where is it at? What's the current performance and are there any open issues?"*
>
> *Sub-queries: "July Grug inference epic vLLM GrugMoE TPU/GPU support" · "inference speed for RL rollouts baseline H100 epic" · "brokered vLLM serving degrades under sustained load / 4096 NaN" · "MoE sharding grid 131k H100 RL step-time feasibility" · "GrugMoE EP8 serving CoreWeave H100 passed" · "logit mixing teacher/student vLLM RunningModel"*
