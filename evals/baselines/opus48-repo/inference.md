# Marin Inference: who, where, performance, open issues

## Who is doing it & how it's organized

Inference in Marin splits into three loosely-coupled tracks, each with a clear owner:

1. **Serving / format / harness backbone** — owned by the **Inference / Serving Infrastructure Specialist** role, defined in the epic [#6500](https://github.com/marin-community/marin/issues/6500). This role owns three components: the **Marin Inference Server** (a thin patch on vLLM, "don't reinvent the wheel"), the **Marin Format Library** (authoritative render/parse of the chat/response format), and the **Marin Agent Harness** (sidecar agent for post-training, fixed tool set: Bash/Read/Write/Update/Subagent/WebFetch/WebSearch). It is the serving/format backbone for post-training, explicitly **not** low-latency production traffic. **@penfever (Ben Feuer)** is the most active person in this area (comments, RFCs, trace audits).

2. **GrugMoE (MoE) vLLM inference** — owned by **@yonromai (Romain Yon)**, tracked under the epic [#6867](https://github.com/marin-community/marin/issues/6867) "[Epic] July Grug Inference tasks." Goal: support the full-size GrugMoE model on both TPUs and GPUs, with a stretch goal of fast-enough GPU inference. Children: [#6868](https://github.com/marin-community/marin/issues/6868) (vLLM GrugMoE on TPUs), [#6869](https://github.com/marin-community/marin/issues/6869) (on GPUs), [#6870](https://github.com/marin-community/marin/issues/6870) (GPU perf "decent"). See also [#6106](https://github.com/marin-community/marin/issues/6106).

3. **Levanter / JAX native inference stack** — the long-running in-house engine, tracked by the epic [#1898](https://github.com/marin-community/marin/issues/1898) "[Epic] Inference To Dos." Designed for single-turn batch eval during/after training (lm-eval-harness), RL rollouts (single- then multi-turn), and possibly data-rewriting — **explicitly not** production traffic. Requirements: fast weight swap, logprobs, tensor parallel, good throughput, eventually data-parallel + multi-turn.

**RL-rollout inference speed** is its own workstream, [#6709](https://github.com/marin-community/marin/issues/6709) (owned by @yonromai), focused on H100 GPUs where Marin intends to do most RL this year. First task there is still to **measure and document the baseline** before setting a target — i.e. the target itself is not yet agreed.

## Where it's at / current performance

- **The 67B-A2B model is now servable.** Per [#7170](https://github.com/marin-community/marin/issues/7170) (owner @penfever), the 2T intermediate cut finished at **Paloma macro-loss 2.2772 / macro_bpb 0.8242** and is published to GCS + CoreWeave S3 + a v6e inference zone. Two inference paths work today: **Levanter/JAX** and the **Marin fork of vLLM** (github.com/marin-community/vllm). A safetensors→vLLM export exists in `main` but the serialization contract isn't standardized; HF publication is still under discussion.
- **GrugMoE GPU perf is "decent"** ([#6870](https://github.com/marin-community/marin/issues/6870)) but the acceptance bar is still a TODO — "within X% of JAX on GPUs / within Y% of a reference model (Qwen?)" is unfilled, so there's no hard number yet.
- **Throughput vs. quality tradeoff is real and quantified** on TPU multi-host serving. Per [#6136](https://github.com/marin-community/marin/issues/6136), a Qwen3.5-122B FP8 run on v5p-32 hit **1815 tok/s at max_num_seqs=64 vs 135 tok/s at seqs=2 — a ~13× throughput delta — but the high-throughput config produces garbage output** (repetition, think-overuse). Throughput tooling measures tok/s but not quality, so the corruption went unnoticed for days.
- **RL rollout throughput** ([#6709](https://github.com/marin-community/marin/issues/6709)) has **no established baseline yet** — establishing it is the stated first task.

## Open issues / risks

**Serving infra & packaging**
- [#7143](https://github.com/marin-community/marin/issues/7143) — TPU vLLM is a source build of two forks (`vllm` + `tpu-inference`), so every cold TPU job recompiles vLLM; proposal to publish prebuilt wheels. `marin-serve --tpu` can't currently run outside a marin checkout (#7106, #7107).
- [#5985](https://github.com/marin-community/marin/issues/5985) — `marin.inference.distributed` unusable on `main` (missing zephyr/fray supporting changes).
- [#6545](https://github.com/marin-community/marin/issues/6545) — quick one-liner inference server & dashboard (CLOSED / done).

**TPU / kernel correctness & perf**
- [#6136](https://github.com/marin-community/marin/issues/6136) — decode corruption on multi-host DP=2 + max_num_seqs ≥ 4 (continuous-batching / paged-KV bug).
- [#7085](https://github.com/marin-community/marin/issues/7085) — RPA v3 unsupported on TPU v4 (`NotImplementedError: Unsupported tpu_version=4`).
- [#5672](https://github.com/marin-community/marin/issues/5672) — Delphi vLLM RPA VMEM regression.
- [#6229](https://github.com/marin-community/marin/issues/6229) — optimize v6e prefill-heavy decode gap.
- [#6480](https://github.com/marin-community/marin/issues/6480) — audit Marin-owned pins constraining TPU-vLLM refreshes; [#6123](https://github.com/marin-community/marin/issues/6123)/[#6112](https://github.com/marin-community/marin/issues/6112) — selectable/generalized TPU paged-attention backend.

**Levanter engine correctness/perf**
- [#1879](https://github.com/marin-community/marin/issues/1879) — trainer vs. inference logprob mismatch.
- [#6231](https://github.com/marin-community/marin/issues/6231) — finish token-native RL rollout data plane; [#6228](https://github.com/marin-community/marin/issues/6228)/[#6230](https://github.com/marin-community/marin/issues/6230) — Qwd3-8B TPU parity matrix / benchmark stability.

**RL-serving scheduling**
- [#5702](https://github.com/marin-community/marin/issues/5702) — adaptive inference launcher's perpetual pending queue blocks higher-priority gangs.
- [#4286](https://github.com/marin-community/marin/issues/4286) / [#6509](https://github.com/marin-community/marin/issues/6509) — packed vLLM inference on Iris RL / chunked throughput sweep.

**Forward-looking design (open RFC)**
- [#7236](https://github.com/marin-community/marin/issues/7236) — RFC on **on-policy distillation (OPD) teacher serving**. Author leans toward **Option 3: permanent marin-serve teacher endpoints on CoreWeave (top-K logprobs, K≈256)**, reusing MarinSkyRL's existing remote teacher client and doubling as the LLM-judge + eval endpoint (part of the "~10 hosted endpoints" the post-training plan wants). The pivotal open question is whether top-K (vs full-vocab reverse-KL) is sufficient. Cross-linked to the [#6500](https://github.com/marin-community/marin/issues/6500) serving epic.

## Summary

Inference work is real and active on three fronts: **@penfever** on the serving/format/harness backbone ([#6500](https://github.com/marin-community/marin/issues/6500)) and OPD teacher-serving design ([#7236](https://github.com/marin-community/marin/issues/7236)); **@yonromai** on GrugMoE vLLM support ([#6867](https://github.com/marin-community/marin/issues/6867)) and RL-rollout speed ([#6709](https://github.com/marin-community/marin/issues/6709)); and the long-running Levanter/JAX engine ([#1898](https://github.com/marin-community/marin/issues/1898)). The 67B-A2B model is servable today via both Levanter and the Marin vLLM fork ([#7170](https://github.com/marin-community/marin/issues/7170)). Concrete performance targets are still mostly TBD — GrugMoE GPU acceptance thresholds ([#6870](https://github.com/marin-community/marin/issues/6870)) and the RL-rollout baseline ([#6709](https://github.com/marin-community/marin/issues/6709)) are both unfilled — and the biggest correctness risk on record is the TPU multi-host decode-corruption-at-high-throughput bug ([#6136](https://github.com/marin-community/marin/issues/6136)).
