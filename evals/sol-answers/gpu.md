# How GPU training is going

GPU training is now real and multi-node, but it is not yet a uniformly production-ready replacement for the TPU path. The strongest infrastructure result is that the CoreWeave `cw-us-east-02a` cluster was brought up from a one-node smoke to full-cluster MoE training: a 9.5B-parameter MoE completed on 64 H100s, and a later 89.67B-total/5.34B-active, 128-expert top-4 MoE trained on all 256 H100s. Those were short bring-up/stress runs, not long hero runs or evidence that efficiency was already good ([#6292](https://github.com/marin-community/marin/issues/6292)).

## Performance: meaningful progress, with model-shape-dependent ceilings

For the 90B-total/5.3B-active MoE on 256 H100s, the initial measured result was 218K tokens/s and 2.8% active-FLOPs MFU. Increasing global batch from 256 to 512 raised MFU to 5.37%; batch 1024 reached 9.32%; `save_moe` rematerialization yielded 712K tokens/s and 9.37%, effectively neutral versus 9.32%. Larger batches, PGLE, longer sequences, and several reshard/collective variants either OOMed, hung, or regressed. The campaign explicitly closed at **9.37% MFU**, calling that the practical pure-FSDP ceiling for this shape; the issue title's ~25% was a target, never an achieved result ([#6304](https://github.com/marin-community/marin/issues/6304)).

The d2560 / roughly 67B-total Grug shape exposed a different set of problems. The H100 campaign tried FA4 attention, fp32-master/bf16-live parameters, rematerialization modes, reduced batches and sequences, different model/expert axis layouts, expert AdamH, and topology-preserving minimizers. Early 256-GPU attempts died before metrics through 60.91 GiB allocation failures, XLA collective-clique startup stalls, or full-depth activation liveness. Reducing sequence length demonstrated that the full sequence was a major memory driver, and the team eventually obtained profile-bearing reduced-batch paths, but the issue still framed ≥20% at 128 GPUs as an outstanding goal rather than a closed achievement ([#6367](https://github.com/marin-community/marin/issues/6367), [#6693](https://github.com/marin-community/marin/issues/6693)).

Muon is a material GPU-specific cost. On a single 8×H100 node, an otherwise matched SGD run measured 16.19% MFU / 74.8K tokens/s / 0.438 s per step versus MuonH5 at 4.75% / 22.0K / 1.492 s. Extensive grouping, padded-bank, 4D Newton–Schulz, packed-master, and FSDP-materialization experiments found useful local layout wins, but the close-out diagnosis was that optimizer communication remains large and does not scale away with node count in FSDP-like layouts ([#6493](https://github.com/marin-community/marin/issues/6493)).

## Reliability and the current validation run

The platform plumbing is substantially better than it was: multi-host CoreWeave support, Kueue gang admission, RDMA/InfiniBand setup, CUDA 13 staging, and Iris job federation all exist. But failures remain split across model code, memory, compiler/collective startup, NCCL dependency conflicts, and cluster networking. A previous NCCL all-to-all failure was traced to conflicting NCCL versions and closed after a dependency pin ([#5377](https://github.com/marin-community/marin/issues/5377)); the new cluster nevertheless needed live bring-up work before full-scale jobs were dependable ([#6292](https://github.com/marin-community/marin/issues/6292)).

The July 11B-A1.5B, 500B-token H100 validation should be described cautiously. The weekly narrative reported roughly 23.8% MFU and projected completion, but primary W&B state in the freeze shows multiple July 13 multi-node attempts crashed, including the 4-node and 2-node `nosim` runs ([4-node run](https://wandb.ai/marin-community/marin_moe/runs/grug-d2048-L24-mla-500B-r4-nosim-4node-20260713-233635), [2-node run](https://wandb.ai/marin-community/marin_moe/runs/grug-d2048-L24-mla-500B-r2-nosim-2node-20260713-234514)). Replacement 500B runs were still marked running on July 14 rather than finished ([base run](https://wandb.ai/marin-community/marin_moe/runs/grug-d2048-L24-mla-500B-r4-b512-datakit-20260714-001739)). So ~23.8% is a measured throughput snapshot for this smaller validation shape, not evidence of a completed durable run.

## Bottom line

The GPU stack has crossed the “can it train distributed MoEs?” threshold. It can run full-cluster H100 jobs and has credible smaller-shape MFU. The remaining gap is sustained, efficient training of the production-scale sparse model: pure FSDP topped out around 9.4% on the 90B/5.3B-active shape, Muon adds severe optimizer-side overhead, and the more favorable ~23.8% validation number came from a much smaller model whose attempted long run was still recovering from crashes at the freeze. Blackwell kernel, FP8, pipeline-parallel, and grouped-expert work are therefore investments toward future bars, not results that should be back-projected onto current H100 production training.

## Research note

Verified primary sources: #6292, #6304, #6367, #6493, #5377, #6693, and the cited W&B run records. Gap: the frozen GitHub corpus contains only skeletal bodies for #6706/#6710 and no primary run record that independently substantiates the weekly summary's 17.8% B200 claim, so that number is omitted from the achieved-results account.

---
<!--provenance-->
> *Data: frozen marinmirror — 68,026 chunks, built 2h before research · summaries through 2026-07-06_2026-07-12; no refresh.*
>
> *Query: "explain how training on GPUs is going"*
>
> *Sub-queries: H100 11B validation/run state · H100 weak scaling and NCCL · 90B pure-FSDP ceiling · d2560/67B memory and sharding · Muon optimizer cost · B200 kernel/FP8 work · Iris/CoreWeave reliability.*
