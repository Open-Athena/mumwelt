# How training on GPUs is going in Marin

**Short version:** GPU training is an active, fast-maturing frontier for Marin — historically a TPU-native project, so GPU is the newer, more actively-churning path. Over roughly May–July 2026 the team stood up dedicated CoreWeave H100 clusters under the Iris scheduler, proved out real large-scale training (a ~90B-parameter sparse MoE across 256 H100s), and is now closing the remaining gaps — chiefly a fused-attention kernel path, MoE/MFU performance, scheduler robustness (Kueue/gang scheduling), profiling, and fault tolerance. It works end-to-end and scales, but throughput/MFU tuning and some kernel/plumbing polish are still in progress.

## The umbrella: GPU Upskilling Master Plan

The effort is organized under a **GPU Upskilling Master Plan** epic ([#6998](https://github.com/marin-community/marin/issues/6998), opened by David Hall). Its stated goals: (1) learn GPU quirks (NCCL etc.) at nontrivial scale, (2) get **Blackwell MoE training really fast at ~600–800 chips**, (3) use all available GPUs effectively, and (4) deliver useful artifacts to post-training. It fans out into workstreams: a large **A2B MoE on Hopper** (target: A2B MFU ≥ 20–25% on ~128–256 Hoppers), a Blackwell smoketest, FP8/MXFP8 sanity checks, and a fused expert-parallel (EP) MoE kernel. The plan is candid about current Hopper pain points: high sparsity + Adam taxing HBM, a slow MoE block, and Muon overhead — with pipeline parallelism named as the real long-term fix (tracked across [#4302](https://github.com/marin-community/marin/issues/4302), [#6367](https://github.com/marin-community/marin/issues/6367), [#6979](https://github.com/marin-community/marin/issues/6979), [#6841](https://github.com/marin-community/marin/pull/6841)).

## The GPU platform: CoreWeave + Iris

Marin's GPU capacity lives on CoreWeave, managed by the in-house **Iris** scheduler on a Kubernetes backend. The `US-EAST-02A` cluster (`cw-us-east-02a`) was brought up and validated in [#6292](https://github.com/marin-community/marin/issues/6292), using a pinned-warm, no-autoscale model (`buffer_slices == max_slices`) that scaled from 8 H100 nodes to the full region quota of **32× H100 (256 GPUs)** plus Genoa CPU nodes, backed by Cloudflare R2 object storage. An earlier `cw-rno2a` H100 cluster is also in use (referenced in the attention work below).

## Headline result: a real ~90B MoE trained on 256 H100s

The cluster bringup culminated in a genuine at-scale run, not just a canary ([#6292](https://github.com/marin-community/marin/issues/6292)):

- A **Grug MoE with 89.67B total / 5.34B active parameters** (128 experts, top-4, ~16.8× sparsity) trained across all **32 nodes / 256 H100s** — 50/50 steps, loss decreasing, 218K tokens/sec, zero OOM and zero sharding errors.
- This required real engineering beyond a bigger canary: explicit `expert_axis_size` / `replica_axis_size` knobs building a mesh `(replica_dcn=1, data=32, expert=8, model=1)` — FSDP over the cross-node `data` axis via InfiniBand, expert parallelism over the intra-node NVLink `expert` axis. A multi-host shared-expert sharding fix ([#6296](https://github.com/marin-community/marin/pull/6296)) was a prerequisite.
- Honest caveats from the maintainers: **MFU was low (~3%)** — this was a correctness/scale run with reduced batch/seq and a wasteful FSDP reshard, not throughput-tuned (which is exactly what the Master Plan's ≥20–25% Hopper MFU goal targets). Two real bugs surfaced and were fixed (an XLA SPMD fully-replicated-reshard OOM, and a stale Kueue workload that held Topology-Aware-Scheduling slots for ~100 min).

Real-checkpoint GPU serving/validation for GrugMoE also landed as a CoreWeave H100 smoke/e2e gate ([#6966](https://github.com/marin-community/marin/issues/6966)).

## The biggest active kernel gap: fused attention on GPU

A significant, recently-diagnosed issue is that **Levanter dense-LM attention was silently running non-fused on GPU** ([#7013](https://github.com/marin-community/marin/issues/7013)). The default GPU backend was NVTE (Transformer Engine), but no Marin environment ships `transformer_engine` — the `gpu` extra ships `flash-attn-4[cu13]` + `nvidia-cutlass-dsl` instead. So every dense-LM GPU run (Qwen/Llama pipelines included) fell back to an unfused O(seq²) reference kernel that also OOMs at seq 8192. On a seq-8192 / GQA 32-8 / head-dim-128 shape, the fused FA4 path is **~3.4× faster** than the fallback (11.3 ms vs 38.9 ms fwd+bwd/iter on an H100).

The fix is in flight:
- [#7140](https://github.com/marin-community/marin/pull/7140) (**closed**) wired FA4 CuTe in as `AttentionBackend.GPU_FLASH` and made it the GPU default via the general attention path.
- [#7015](https://github.com/marin-community/marin/pull/7015) (**open**) adds `AttentionBackend.FA4` backed by the FA4/CuTe segmented kernels already living in `levanter.grug.attention`, making it the GPU default (falling back to NVTE / blocked JAX-flash on unsupported configs), with fwd+grad parity validated on a cw-rno2a H100 pod. PR feedback has been integrated.
- Getting FA4 CuTe + THD working inside the CoreWeave task image itself (missing `cutlass` module) was tracked and closed in [#6226](https://github.com/marin-community/marin/issues/6226).

## Scheduler / infra hardening for GPU

The Iris GPU control plane has seen a steady stream of merged fixes, indicating active production use:

- **Kueue made mandatory** on the k8s backend with priority preemption, so higher-priority gangs can preempt lower-priority `batch` jobs instead of starving ([#7207](https://github.com/marin-community/marin/pull/7207)).
- **GB200 NVL72 gang coscheduling** on the NVLink domain (18-node rack) plus admitting single-GPU pods under Topology-Aware Scheduling ([#7217](https://github.com/marin-community/marin/pull/7217)) — i.e. Blackwell/GB200 support is arriving alongside H100.
- CPU pods made able to reuse idle GPU nodes via a selector-less `cw-cpu` flavor ([#7226](https://github.com/marin-community/marin/pull/7226)).
- CUDA/cuDNN toolchain staging and precedence fixes for GPU tasks ([#6601](https://github.com/marin-community/marin/pull/6601), [#6947](https://github.com/marin-community/marin/pull/6947), [#7031](https://github.com/marin-community/marin/pull/7031)); isolated uv-tool env for CUDA vLLM serving ([#7133](https://github.com/marin-community/marin/pull/7133)).
- Earlier CI-canary race conditions during controller rollout were found and fixed ([#5469](https://github.com/marin-community/marin/issues/5469)).

## Observability and reliability, in progress

- **Hardware topology logging** on trainer startup — GPU runs record the raw `nvidia-smi` topology matrix plus GPU-GPU/GPU-NIC link counts ([#7009](https://github.com/marin-community/marin/pull/7009), merged).
- **Nsight Systems profiling** for GPU jobs via `iris job run --gpu ... --nsys`, verified on 4× GB200 ([#7222](https://github.com/marin-community/marin/pull/7222), open); plus GPU profiling-flag docs ([#6577](https://github.com/marin-community/marin/pull/6577)).
- A **Grug JAX fault-tolerance design** is under discussion ([#7014](https://github.com/marin-community/marin/issues/7014)) — a sign the team is moving from "it runs" toward "it survives node failures at scale."

## What's queued next

- A **Grug 67B-A2B two-stage chat SFT** launcher targeting the CoreWeave `cw-us-east-02a` H100 cluster (FSDP + ring-EP path), fine-tuning the June TPU 67B MoE cooldown checkpoint ([#7250](https://github.com/marin-community/marin/issues/7250), open).
- **RL on GPU**: making the Iris RL config resource-aware with a GPU smoke probe ([#4524](https://github.com/marin-community/marin/issues/4524), open).
- Ongoing MoE kernel performance work (e.g. SonicMoE-style GMM/MLP kernels, [#5328](https://github.com/marin-community/marin/issues/5328); JAX semantic source-push MLP, [#6841](https://github.com/marin-community/marin/pull/6841)).

## Bottom line

GPU training in Marin has crossed the "does it actually work at scale" bar — a real ~100B-class sparse MoE trains end-to-end on 256 H100s, with the scheduler, storage, and multi-host sharding all validated. The remaining work, framed by the GPU Upskilling Master Plan ([#6998](https://github.com/marin-community/marin/issues/6998)), is about efficiency and polish rather than viability: landing the fused FA4 attention default so dense-LM runs stop leaving throughput on the table, driving Hopper MoE MFU up to the 20–25% target (and getting Blackwell fast at ~600–800 chips), and building out fault tolerance, profiling, and RL/SFT GPU launchers.
