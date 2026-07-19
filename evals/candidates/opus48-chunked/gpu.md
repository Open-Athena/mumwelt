# How GPU training is going

**Short version:** GPU training went from "not ready, hand the hero run to TPU" in late June to a credible second platform in July. Hopper (H100) is the success story — a 500B-token MoE validation run completed end-to-end, and the stack now weak-scales nearly flat to 64 GPUs. Blackwell (B200) is close but has **not** cleared the 20% MFU bar that gates the end-of-month run. GB200/NVL72 rack bringup is the rocky frontier: it trains cleanly to 128 GPUs and dies above 256. Meanwhile the biggest single win of the month was a *bug fix* — every dense-LM GPU run had been silently falling back to an unfused attention kernel.

---

## 1. The arc: June wrote GPUs off, July brought them back

The load-bearing context is that **the June 67B-A2B production run was taken away from the H100 cluster and staged onto TPU v4** because the GPU stack couldn't carry it. From the week of Jun 22–28 ([summary](https://mws.oa.dev/summaries/summary-2026-06-22_2026-06-28.html)):

> "On the headline question of whether the June 67B-A2B run started: it did, but on TPU rather than on the H100 cluster. The d2560 production shape that hit the first-step memory wall on GPU last week was staged onto TPU v4 instead."

Two things were set aside to get there. **Muon was shelved on GPU** — David Hall measured Newton-Schulz at "roughly 500 ms of a 1,500 ms step" on two nodes, and in [#gpu on 2026-06-26](https://discord.com/channels/1354881461060243556/1399998407657001062/1520200189690773564) wrote: *"We're currently at ~19.9 MFU on 4 nodes w/ SGD/Adam… **Muon needs to go for now.** 10% step-count improvement isn't worth atm. **We'll need much faster interconnect or PP.**"* And **pipeline parallelism was parked** — Russell Power's zero-bubble investigation ([#6532](https://github.com/marin-community/marin/issues/6532), [PR #6534](https://github.com/marin-community/marin/pull/6534)) was gradient-exact and unlocked memory scaling but lost on throughput; #6532's close-out on 2026-06-25 reads *"**Parked.** The PP / zero-bubble investigation for grug-MoE has been moved out of marin"*, with PR #6534 closed unmerged.

By 2026-07-10 the verdict had flipped. David Hall, [#gpu](https://discord.com/channels/1354881461060243556/1399998407657001062/1525217063830949888): *"codex is seemingly making good progress on jaxpp [#7024](https://github.com/marin-community/marin/issues/7024) . **Hopper is back on the menu I think**"*.

## 2. Where the numbers actually stand

Every figure below is tagged with hardware, because carrying one across platforms is the easiest way to be wrong here.

| Platform | Best measured MFU | Config | Status |
|---|---|---|---|
| **H100, single node** | **26.7%** | `model.py` full MoE, 8×H100 | achieved ([#6979](https://github.com/marin-community/marin/issues/6979)) |
| **H100, 8 nodes** | **26.5%** | 64×H100, global batch 1024 | achieved — weak-scales flat |
| **H100, real 500B run** | **~23.8%** steady-state (~21% wall-clock) | 64×H100, d2048/L24, GQA 4:1 | run **completed** |
| **H100, JaxPP pipeline** | **18.26%** mean | 32×H100 cw-rno2a, `std_1f1b` | below the 20 bar |
| **B200, single node** | **17.8%** whole-model | 8×B200 at d5120 (14.9% at d2560) | **below the 20% gate** |
| **GB200/NVL72** | — | d5120/L48, ~129B params | bringup; fails above 256 GPUs |

### Hopper: the good news

Larry Dial's GPU MFU Learning Path ([#6979](https://github.com/marin-community/marin/issues/6979)) built the Hopper story from scratch on one 8×H100 node and produced the most useful artifact of the month — an *itemized* accounting of where MFU goes. Per the [Jul 6–12 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html): an lm_head-only cross-entropy benchmark at **50.3%**, a full dense d2560/26-layer model at **56.2%** (FlashAttention-4 CUTLASS, ZeRO-1, scan-over-layers with full rematerialization), then a decline as each MoE correctness feature is switched on — "from a 34.4% throughput probe down to **26.7%** for the full `model.py`, with real Muon's Newton-Schulz orthogonalization the single biggest give-back at **~3.7 points** and each norm only ~0.4–1 point."

The multinode result is the headline:

> "`model.py` weak-scales essentially flat from 1 to 8 nodes — **26.7% at 8 GPUs to 26.5% at 64 H100s** and global batch 1024 — because replica/DDP hides the once-per-step gradient all-reduce over the data-center network while keeping the frequent FSDP all-gathers on intra-node NVLink, whereas **sharding the model across nodes costs about 4 points**."

That insight drove a concrete recipe change: dropping total experts from 256 to 64 (still top-4) lets the full model fit on one node, which Larry Dial proposed as the H100 working copy.

**The real 500B-token validation run completed.** [`grug-d2048-L24-gqa4-500B-r8-nosim-v2`](https://wandb.ai/marin-community/marin_moe/runs/grug-d2048-L24-gqa4-500B-r8-nosim-v2-20260709-042053) finished on CoreWeave at `throughput/total_tokens = 500,000,882,688` (the exact 500B budget), `eval/bpb = 0.7273`, `parameter_count = 1.077e10`. The point was never the artifact — as Isaac Hodes put it, *"this is more about exercising the hardware instead of delivering an artifact."*

**Two honest deflators on the 23.8%.** First, it is a steady-state reading, not a run average: 500B tokens ÷ 412,142.7 s of wall clock = **1.213M tok/s**, against the quoted 1.37M — about 11% lower, i.e. nearer ~21% realized. Second, the summary flags its own caveat: *"the FLOP counter treats attention as fully quadratic and ignores the 2048-token sliding window, so the reported MFU is slightly optimistic on the attention term."* Also worth noting the first launch of that run [crashed](https://wandb.ai/marin-community/marin_moe/runs/grug-d2048-L24-gqa4-500B-r8-20260709-031827) and the finished run is a relaunch.

### Blackwell B200: closing, not closed

Commitment [#6706](https://github.com/marin-community/marin/issues/6706) gates the end-of-month ~120B-A8B run on 2 NVL72: B200 MFU must clear 20%. It has not. The progression, all on a single 8×B200 node ([#7012](https://github.com/marin-community/marin/issues/7012), via the Jul 6–12 summary):

> "Will Held reproduced the H100 config at **12.5% MFU**, established that batch is a dead lever (the workload is arithmetic-intensity-starved, not batch-starved) while width is live (**16.2% at d5120**), then wired Tri Dao's QuACK SM100 grouped GEMM — the SonicMoE kernel — into the expert MLPs via a torch-free CUTLASS shim, lifting **whole-model 8×B200 MFU to 17.8% at d5120** (14.9% at d2560), with the QuACK gated GEMM alone at ~1,175 TFLOP/s, ~52% of peak"

d5120 is the right shape to quote — every GB200 scale-out run is `gb200-d5120-l48-*` at ~129B params, i.e. the ~120B-A8B target. The summary is blunt about the gap: *"It is not there yet, but the gap is closing."* A separate 25% target is set for the September ~512B-A16B run on twelve NVL72 racks, tracked in [#6710](https://github.com/marin-community/marin/issues/6710).

### GB200/NVL72: a scaling wall, not a smooth bringup

This is the newest and least documented area — and the only evidence in the corpus is W&B run records dated 2026-07-16/17, with **no GitHub or Discord commentary at all**. Of 48 `gb200-*` runs: 14 finished, 20 crashed, 13 failed, 1 ambiguous. Raw churn is expected in a one-engineer config sweep, but the pattern by replica count is not noise:

| GPUs | State | Steps reached |
|---|---|---|
| 64 (r1/r2) | **finished** | 29 |
| 128 (r2) | **finished** | 29 |
| 256 (r4) | [crashed](https://wandb.ai/marin-community/marin_moe/runs/gb200-d5120-l48-256gpu-b4096-r4-intrarack-pad) | 23 |
| 512 (r8) | crashed | 12 |
| 640 (r10) | [crashed](https://wandb.ai/marin-community/marin_moe/runs/gb200-d5120-l48-640gpu-b10240-r10-intrarack-pad) | 0 |

That is monotone degradation with scale: clean to 128 GPUs, degrading at 256, dead at 512–640. The newest doc in the corpus, [`gb200-d5120-l48-512gpu-b8192-r8-segfix`](https://wandb.ai/marin-community/marin_moe/runs/gb200-d5120-l48-512gpu-b8192-r8-segfix) (2026-07-17), is a fix attempt whose name suggests a segment bug — but the corpus reports its state inconsistently, so I won't assert an outcome. One genuine full training run did land: [`gb200-d2048-l24-32gpu`](https://wandb.ai/marin-community/marin_moe/runs/gb200-d2048-l24-32gpu), 499 steps / 1.05B tokens, finished.

## 3. Pipeline parallelism: revived, and still 1.74 points short

David Hall's [#7024](https://github.com/marin-community/marin/issues/7024) stood JaxPP pipeline-parallel MoE training up across four 8×H100 nodes on cw-rno2a, pivoting to an explicit `jaxpp.experimental.mpmd` path with stage-local weights after the automatic schedule stayed blocked on JaxPP sharding inference. Best measured point — `std_1f1b`, batch 8192 / 256 microbatches, four six-layer stages, ring EP, CuTe FA4 — reached **mean MFU 18.26 at ~414k tokens/s**, "saturating pipeline occupancy about 1.74 points below the 20 bar."

What followed deserves credit as method even though it failed as engineering: **a disciplined week of negative results**, each A/B-gated on H100 before scaling — latency-hiding scheduler flag (+0.4%, within noise), pure-XLA output-oriented ring combine (−52%), explicit Triton routing kernels (neutral), transfer-priority task ordering (−2.9%), input-gradient-first backward split (−29%), exact two-chunk bulk ring (−14% fwd+bwd). None closed the gap.

The one lever that *did* clear the performance gate hit a correctness wall instead: an EP-local QuACK/Sonic grouped-MLP adapter measured 1.16× forward and 1.108× fwd+bwd **but failed output parity** — a standalone repro pinned the ~2.6% mismatch to QuACK's fused approximate SwiGLU (fast `exp2` and reciprocal), "turning it into a semantic-policy question rather than a bug." That path's own blocker, a shared TVM-FFI handler hanging under concurrent multi-device calls, was root-caused in [#7110](https://github.com/marin-community/marin/issues/7110) and fixed with a per-handler mutex validated at 32-H100 scale — but at 13.96 MFU it is not competitive with ring EP.

For calibration on how far this has come: the older single-shape d2560 push [#6367](https://github.com/marin-community/marin/issues/6367) still sits at *"best clean 8-node no-tensor-parallel smoke still at ~1.8 MFU"* and was effectively handed off to #7024.

## 4. fp8 crossed from kernels into a real training path

The fp8 lane is the cleanest engineering arc of the period, and it is **H100 work**, not Blackwell.

- [PR #6880](https://github.com/marin-community/marin/pull/6880) delivered `Fp8RaggedDotOp`, "a **~1.41× fwd+bwd speedup** relative to the existing bf16 implementation for the MoE expert MLP (measured at D=2560, F=1280, T=65536)" — well past the ≥20% bar preregistered in [#6824](https://github.com/marin-community/marin/issues/6824).
- [#6930](https://github.com/marin-community/marin/issues/6930) tuned the backward, moving acceptance from "w13 1.32x to 1.41x and w2 1.19x to 1.27x."
- [#6911](https://github.com/marin-community/marin/issues/6911) took it over the wire and met its preregistered ≥1.4× goal: *"ring EP16 bf16 33.47 → fp8 23.28 ms/step = **1.438×** on 2×8×H100 over IB."* The design lesson: **fp8 belongs on permutation legs only** — E4M3 forward, E5M2 backward, reductions staying bf16, because decomposing the ring reduce-scatter into an fp8 all-to-all measured **0.885×** against NCCL's hierarchical path.
- The Jul 6–12 summary reports the retuned production figure as **1.53× vs bf16** at d2560/E256/K4, 2-node EP16, wired end-to-end through grug in draft [PR #7079](https://github.com/marin-community/marin/pull/7079) behind a single `GrugModelConfig.fp8` switch. Isaac Hodes closed the original [#6048](https://github.com/marin-community/marin/issues/6048) "try out fp8 training" as done.

**Crucially, fp8 is opt-in and defaults stay bf16.** The gate is explicit: *"Mechanism is derisked; scientific validation is still required before any training run — default stays bf16"* ([#6911](https://github.com/marin-community/marin/issues/6911#issuecomment-4878544634)), with loss-curve validation on a real trajectory tracked in [#6699](https://github.com/marin-community/marin/issues/6699). No corpus document specifies the loss-delta threshold that would flip it.

## 5. Kernels and attention

**The month's highest-leverage find was a bug.** In [#7013](https://github.com/marin-community/marin/issues/7013), @timodonnell found that levanter's GPU default attention backend was NVTE (Transformer Engine), *which no marin environment ships* — the `gpu` extra installs flash-attn-4 instead — "so **every dense-LM GPU run silently fell back to an unfused reference path** that also OOMs at sequence 8192." The fix, [PR #7015](https://github.com/marin-community/marin/pull/7015), wires an FA4 backend and makes it the GPU default: **3.4× faster (11.3 vs 38.9 ms/iter on an H100)**. Blast radius is dense-LM GPU runs specifically — Grug/MoE runs select their implementation explicitly and were unaffected — but it means dense-LM GPU measurements before this fix were taken on a broken default. It is also not fully finished: on 2026-07-13 Will Held noted in [#gpu](https://discord.com/channels/1354881461060243556/1526011060451021034/1526239558948356187) *"There's free MFU on the table if you also switch Levanter to use the Flash attention kernel, I'll make a PR to make that default for Qwen and Llama models as well."*

**Fused MoE kernels** are the strategic bet. David Hall's Pallas Mosaic GPU source-push MoE ([#6597](https://github.com/marin-community/marin/issues/6597)) hit a stable Hopper target-shape median of **218 TFLOP/s per rank** (~13.8% faster than the serial ring-prologue baseline), with a one-node 20-step trainer smoke running end-to-end at **~20% MFU and 554k tokens/s**. It now completes the exact EP8 target forward *and backward* through one JAX-native custom VJP — though the integrated fwd+bwd figure, **242.8 ms / ~31.9 TFLOP/s per rank**, is far below the forward-only number, with "serialized operand movement around WGMMA named as the remaining bottleneck." The Blackwell sibling [#6933](https://github.com/marin-community/marin/issues/6933) had its intended fully-fused design *ruled out* (the JAX/Mosaic stack cannot lower peer-id memory references in Warpgroup mode), so the shipped path is staged transport plus local compute; the summary reports it cleared its 300 TFLOP/s-per-rank forward target at **666 useful TFLOP/s per rank** steady-state once outer-JIT compiled, integration in draft [PR #6970](https://github.com/marin-community/marin/pull/6970).

**One alternative was evaluated and rejected.** [#7114](https://github.com/marin-community/marin/pull/7114) tested a CuTe/NVSHMEM push/pull transport against Mosaic source-push. The primitives all worked (warp `put_signal` at 5.30 µs / 1.16 GB/s per PE) — "but the decision was to keep Mosaic — transport overlap more than halves concurrent GEMM throughput and the JAX/XLA custom-call path will not compile against the tested CUTLASS/NVSHMEM stack."

## 6. Cluster and infrastructure

The GPU fleet grew and got more reliable, but not quietly.

- **A multi-node NCCL bootstrap bug was silently breaking every CoreWeave cluster.** [PR #6941](https://github.com/marin-community/marin/pull/6941) (2026-07-04): an exact-match `NCCL_SOCKET_IFNAME` "is not honored by the current task image's NCCL/XLA build: the filter matches no interface, so NCCL's bootstrap finds no socket and multi-node clique init fails." Switching to exclude-list form across `cw-rno2a`, `cw-us-east-02a` and `ci-coreweave` (*"all latently broken by the same pattern"*) "got an 8-node / 64×H100 grug-moe run past clique init into training."
- **cw-rno2a came online** as a 512-H100 cluster ([PR #6909](https://github.com/marin-community/marin/pull/6909)), alongside cw-us-east-02a's 256.
- **The biggest reliability event was a controller cascade**, not a GPU fault: [#6944](https://github.com/marin-community/marin/issues/6944) — a large fan-out on cw-rno2a "drove the k8s controller into a repeated crash-restart loop… **Two soak arms died this way**," via OOMKill at 16Gi then a liveness-probe kill at 128Gi. Fixed in [#6945](https://github.com/marin-community/marin/pull/6945) and [#6943](https://github.com/marin-community/marin/pull/6943).
- **The daily GPU canary ferry is a steady stream of small breakages** — SIGSEGV at profiler upload ([#6108](https://github.com/marin-community/marin/issues/6108), reopened), a missing `expert` axis in the batch pspec ([#6252](https://github.com/marin-community/marin/issues/6252)), a controller redeployed mid-run wiping the canary job ([#6808](https://github.com/marin-community/marin/issues/6808)), and a trailing-slash `MARIN_PREFIX` corrupting tokenize cache paths ([#6904](https://github.com/marin-community/marin/issues/6904)). For the week of Jul 6–12 the CoreWeave GPU canary ferry ran **4 passes / 3 failures**, against 7/7 for the TPU ferry.
- **Iris cross-cluster federation went live**, so marin can hand whole jobs to CoreWeave — [#6826](https://github.com/marin-community/marin/pull/6826) (observable peers) → [#6835](https://github.com/marin-community/marin/pull/6835) (federated handoff) → [#6884](https://github.com/marin-community/marin/pull/6884) (cluster-native), tracked as [#7064](https://github.com/marin-community/marin/issues/7064). The observable payoff: the d512 compute-optimal ablation ladder ran the same day on CoreWeave H100 *and* TPU v5p-8.

## 7. Two things worth flagging that the narrative sources don't cover

**The architecture question may not be settled.** The Jul 6–12 summary says the GQA-vs-MLA call was "largely settled last week" in GQA's favor. But W&B shows that immediately after the GQA 500B run finished, the team **relaunched the whole 500B validation at MLA** on 2026-07-13/14: [`grug-d2048-L24-mla-500B-r4-b512-datakit`](https://wandb.ai/marin-community/marin_moe/runs/grug-d2048-L24-mla-500B-r4-b512-datakit-20260714-001739) (running, ~249B tokens, `eval/bpb 0.7805`) and a `qkvscale-noswa` variant (running, ~212B, `0.7872`), with five sibling MLA launches on 07-13 all crashed. On the evidence so far MLA is *behind* GQA's finished 0.7273 — but at roughly half the token budget, so this is not a fair comparison, and **no source in the corpus comments on it**. Treat as an open question, not a result.

**Muon is not shelved any more.** The June "Muon needs to go for now" call was GPU-specific and has since eroded: it is in production on the live TPU hero run, [PR #7118](https://github.com/marin-community/marin/pull/7118) adds error-aware Muon feedback policies, and MuonH runs are *finishing* on GB200 as of 2026-07-16 ([`gb200-d5120-l48-64gpu-b1024-r1-muonh-cudaasync`](https://wandb.ai/marin-community/marin_moe/runs/gb200-d5120-l48-64gpu-b1024-r1-muonh-cudaasync)). The accurate statement is that Muon remains *expensive* on GPU — the single biggest MFU give-back at ~3.7 points, and named as an obstacle in the GPU Upskilling Master Plan ([#6998](https://github.com/marin-community/marin/issues/6998)) — not that it is set aside.

## 8. The main risk

The ~120B-A8B B200 run is supposed to kick off on 2 NVL72 at the end of the month, but as of the Jul 6–12 summary its **data mix, architecture, and preregistered loss target were all still undecided** — [#7073](https://github.com/marin-community/marin/issues/7073) is at 4/14 sub-issues closed, and "no decision was recorded this week." Its gating bar (#6706, 20% B200 MFU) is unmet at 17.8%. And the rack-scale evidence above suggests multi-rack GB200 training is not yet stable past 128 GPUs. Those three facts point the same direction.

## Caveats on this answer

- **Corpus edge.** The frozen corpus's GitHub mirror ends around **2026-07-05**. Issues #7012, #7013, #7015, #7024, #7073, #7079, #7110, #7114, #6970, #6979, #6998 are **not indexed as primary threads** — every number attributed to them rests on the [Jul 6–12 weekly summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html) and could not be cross-checked against the issue text. Discord and W&B records do extend to ~2026-07-17, which is how the GB200 and MLA findings were reachable at all.
- **Issue #6716 is a stub.** The H100 hero-run issue in the mirror is titled "[Hero run] Land 8B-A1B XT on H100s" with zero comments; the 11B-A1.5B framing and the 23.8% figure come from the summary, corroborated on completion and model size (but not on MFU) by W&B.
- **One stale number deliberately omitted from the headline:** the often-quoted *"90B-total/5.3B-active MoE plateaued at 9.37% MFU on 256 H100"* ([#6321](https://github.com/marin-community/marin/issues/6321)) is from **2026-06-10**, was explicitly scoped by its author as *"the practical pure-FSDP ceiling"* ([#6304](https://github.com/marin-community/marin/issues/6304#issuecomment-4674696103)), and describes a shape that is no longer a production target. Quoting it as the current state of H100 training would understate reality by roughly 2.5×.
- **The 20% bar is a B200 commitment, not a universal one.** Hopper already clears it in the `model.py` path; applying it to H100 understates the Hopper position.

---
<!--provenance-->
> *Data: marinmirror — 86367 chunks, built 0h ago · summaries through 2026-07-06_2026-07-12 (GitHub mirror edge ~2026-07-05; Discord/W&B to ~2026-07-17). No refresh triggered this run.*
>
> *Query: "explain how training on GPUs is going"*
>
> *Sub-queries: "B200 / NVL72 Blackwell MFU push: QuACK SonicMoE grouped GEMM, the 20% bar, the 25% September target" · "H100 Hopper MoE MFU learning path and GPU upskilling master plan: itemized give-backs, Muon Newton-Schulz overhead, weak scaling to 64 H100s" · "JaxPP pipeline-parallel MoE training on 8xH100 nodes: std_1f1b, mean MFU 18.26, the negative-result A/Bs, QuACK handler hang" · "fp8 / MXFP8 training in grug MoE: Fp8RaggedDotOp, over-the-wire collectives, and the gate before flipping defaults" · "Marin 11B-A1.5B validation run on 64xH100 CoreWeave, plus GPU cluster reliability, NCCL bootstrap failures, and Iris federation" · "GPU attention backend and fused MoE kernels: FlashAttention-4 as GPU default, Pallas Mosaic source-push, CuTe NVSHMEM transport" · adversarial verification pass on all load-bearing numbers*
