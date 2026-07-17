# Inference: owners, locations, performance, and open work

Inference is split between a correctness/evaluation path on TPU and an RL-throughput path centered on GPUs. Romain Yon drove much of the GrugMoE vLLM/export/fork work; David Hall worked on the Levanter/JAX reference path and performance; the July tracking issues make GPU/H100 rollout speed the forward-looking priority. The frozen corpus supports those work assignments through authorship and issue activity, but it does not define a complete formal owner roster.

## Where it runs and what is working

On TPU, GrugMoE inference reached a correctness-complete state. The work implemented Levanter-to-Hugging Face export, canonical tensor mapping, sharded safetensor loading, deterministic generation parity, and a real-checkpoint end-to-end comparison between vLLM serving and the Levanter/JAX reference. The path was validated through Marin-owned vLLM and `tpu-inference` forks and closed as complete for eval-grade inference in #6041; serious RL performance and upstreaming were explicitly out of scope ([#6106](https://github.com/marin-community/marin/issues/6106), [#6041](https://github.com/marin-community/marin/issues/6041)).

The Levanter/JAX TPU path also improved materially: the issue records a roughly 10× speedup from removing avoidable overhead, leaving it within about 12% of vLLM TPU. Treat that as a reported comparison, not a general throughput number: the source does not attach a model/batch/token-rate table to it ([#6041](https://github.com/marin-community/marin/issues/6041)).

On GPU, the intended production location is CoreWeave H100s, particularly for RL rollouts. A July report says MarinSkyRL's 131k-context MoE RL plumbing was running on CoreWeave, with a very rough estimate of ~10M generated tokens per RL step; that establishes integration, not throughput or efficiency ([Discord thread](https://discord.com/channels/1354881461060243556/1374989195109466122/1524748716190339122)).

## Current performance status

TPU correctness is ahead of GPU performance. The GPU vLLM issue's definition of done is still only “fast enough for RL,” and even its comparison threshold is left as TODO; therefore the frozen corpus does **not** support a claimed achieved GPU token rate or parity percentage ([#6870](https://github.com/marin-community/marin/issues/6870)). The July epic similarly says full-size GrugMoE support on TPU and GPU is the goal, with fast GPU inference only a stretch goal ([#6867](https://github.com/marin-community/marin/issues/6867)).

The strategic direction is unambiguous: speed up generation so RL is not inference-bound, with H100s as the main target hardware for the year's RL work ([#6709](https://github.com/marin-community/marin/issues/6709)). Thus “inference works” currently means two different things: export/load/generation correctness is demonstrated on TPU, while scalable GPU rollout serving is still an active performance and full-size-support project.

## Open issues

1. **Full-size GrugMoE on both platforms.** July's primary completion criterion is support for the full-size model on TPU and GPU, not just small correctness fixtures ([#6867](https://github.com/marin-community/marin/issues/6867)).
2. **Define and hit the GPU performance bar.** #6870 has neither a filled-in parity target nor a close-out measurement. It remains unsafe to say GPU vLLM is already “decent” merely because that phrase appears in the title ([#6870](https://github.com/marin-community/marin/issues/6870)).
3. **RL rollout throughput.** The H100-focused investment must make generation fast enough that inference does not dominate RL iteration time ([#6709](https://github.com/marin-community/marin/issues/6709)).
4. **Fork maintenance and consolidation.** The working TPU path depends on Marin-owned vLLM/`tpu-inference` forks pinned by commit; the older opaque package path was replaced, but refresh/rebase work remains part of carrying support forward ([#6041](https://github.com/marin-community/marin/issues/6041), [#6288](https://github.com/marin-community/marin/pull/6288)).
5. **Artifact/export regressions at production scale.** The correctness work validated canonical exports and sharded loading, but full-size checkpoints and both serving backends need continuing regression coverage as the model and forks move ([#6106](https://github.com/marin-community/marin/issues/6106)).

## Bottom line

Romain's vLLM/export work has made TPU inference correct and usable for evaluations; David's optimized Levanter/JAX path is a competitive reference. CoreWeave H100 is where RL inference is meant to scale, and MarinSkyRL plumbing exists there, but there is no verified final GPU throughput result in the freeze. The honest current status is “TPU correctness landed; GPU full-size support and RL-speed inference remain active.”

## Research note

Verified primary sources: #6106, #6041, #6288, #6867, #6870, #6709, and the cited CoreWeave MarinSkyRL Discord thread. Gaps: no authoritative GPU vLLM benchmark table or closed performance threshold was present; no formal ownership document was found, so people attribution is limited to visible issue/implementation leadership.

---
<!--provenance-->
> *Data: frozen marinmirror — 68,026 chunks, built 2h before research · summaries through 2026-07-06_2026-07-12; no refresh.*
>
> *Query: "who's doing inference, where is it, current perf, open issues?"*
>
> *Sub-queries: GrugMoE TPU vLLM correctness · Levanter/JAX reference perf · Marin fork/export stack · GPU vLLM full-size support · CoreWeave H100 RL rollout path · MarinSkyRL 131k integration · July inference epic · open performance bars.*
