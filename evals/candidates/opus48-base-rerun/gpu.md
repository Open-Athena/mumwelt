# How GPU training is going

**Short version:** GPU training has crossed from "can it run at all" to "can it run *efficiently enough* to bet a hero run on." The reliability story is largely won — a 64×H100 run just finished 500B tokens clean. The efficiency story is not: Blackwell sits at **17.8% MFU single-node against a 20% bar**, and every path to closing that gap (fp8, pipeline parallelism, fused EP kernels) is measured, promising, and *not yet landed as a default*.

---

## 1. The frame: July is a preparation month for two GPU hero runs

The year's throughline is the Marin 2026 contender. Per the week-of-July-6 summary: *"when the twelve NVL72 racks land around September 15, the team kicks off its largest hero run — a ~512B-A16B model on 20T tokens. July's job is to be ready for that, and for the ~120B-A8B B200 run that kicks off on 2 NVL72 at the end of the month"* ([summary 2026-07-06_2026-07-12](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).

Note the current production hero run is **not** on GPU: the 67B-A2B Grug MoE runs on **TPU v4-2048** ([#6704](https://github.com/marin-community/marin/issues/6704), logged on [#6044](https://github.com/marin-community/marin/issues/6044)). GPUs today carry validation runs, MFU engineering, and the *future* hero runs. That split is deliberate and dates to June, when the 67B-A2B production shape was moved off CoreWeave H100s onto TPU after three weeks stalled on a first-step memory wall and Muon's poor fit under FSDP ([summary 2026-06-22_2026-06-28](https://mws.oa.dev/summaries/summary-2026-06-22_2026-06-28.html)).

**Caveat on shapes and dates:** the "~120B-A8B", "~512B-A16B", "twelve NVL72" and "September 15" specifics trace to the weekly-summary narrative only. The hero-run tracking issue [#6689](https://github.com/marin-community/marin/issues/6689) still reads *"288 B200 (4xNVL72)"* with FLOP budget, tokens, data mix and pre-registered loss all blank — a June-vintage conflict with the summary's "2 NVL72" that nothing in the corpus reconciles.

---

## 2. Blackwell (B200): 17.8% MFU, bar is 20% — ACHIEVED vs TARGET

This is the headline number and the most misread one.

**Achieved:** Will Held *"reproduced the H100 config at 12.5% MFU, established that batch is a dead lever (the workload is arithmetic-intensity-starved, not batch-starved) while width is live (16.2% at d5120), then wired Tri Dao's QuACK SM100 grouped general matrix multiply (GEMM) — the SonicMoE kernel — into the expert MLPs via a torch-free CUTLASS shim, lifting whole-model 8×B200 MFU to 17.8% at d5120 (14.9% at d2560), with the QuACK gated GEMM alone at ~1,175 TFLOP/s, ~52% of peak"* ([#7012](https://github.com/marin-community/marin/issues/7012), via [summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).

Three qualifications that matter:

1. **17.8% is a single 8×B200 node.** Both bars — 20% for the end-of-month run, 25% for September — are stated for **NVL72 rack scale**. The achieved number is not measured on the hardware the bar is about.
2. **1,175 TFLOP/s / ~52% of peak is the kernel in isolation**, not whole-model. Don't conflate the two.
3. **The 20% bar is not met.** [#6706](https://github.com/marin-community/marin/issues/6706) (whose original body is one line — *"Need to be at 20%+?"*) was retitled to *"Get B200 MFUs above 20% in advance of Aug 1 run"*, and the summary states plainly: *"It is not there yet, but the gap is closing."* The 25% figure is a separate September target tracked in [#6710](https://github.com/marin-community/marin/issues/6710) — **never report 25% as a Blackwell achievement.**

Older, independent B200 evidence corroborates that the fundamentals are decent but the Grug path was behind: at a pinned d5120/L8, EP=8, 8×B200 fair-fit comparison, Megatron TE+DeepEP hit 91,575 tok/s vs Grug's 24,351 (ring) ([#5815](https://github.com/marin-community/marin/issues/5815#issuecomment-4615932878)) — a gap Marin then closed to **82,622 tok/s** with ring + `gpu_fa4_thd` + outer optimizer clip ([#6139](https://github.com/marin-community/marin/issues/6139#issuecomment-4619246391)).

---

## 3. Hopper (H100): the best GPU numbers Marin has, and they're respectable

### The validation hero run completed

The clearest good news. [#6716](https://github.com/marin-community/marin/issues/6716) brought the **Marin 11B-A1.5B** MoE (d2048, 24 layers, 64 experts top-4, GQA 4:1, ~1.53B active / 10.6B total) up on **64×H100 on CoreWeave `cw-us-east-02a`**. It *"held ~23.8% MFU (~1.37M tokens/s, ~1.53 s/step at batch 512 × seq 4096)"* ([summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).

The point was explicitly not the artifact — the run was pushed to 500B tokens, well past compute-optimal, *"to shake out the GPU training stack and exercise the hardware ahead of the larger Blackwell runs — as Isaac Hodes put it, 'this is more about exercising the hardware instead of delivering an artifact.'"*

- **It finished.** W&B run [`grug-d2048-L24-gqa4-500B-r8-nosim-v2-20260709-042053`](https://wandb.ai/marin-community/marin_moe/runs/grug-d2048-L24-gqa4-500B-r8-nosim-v2-20260709-042053) shows `state=finished`, 500,000,882,688 tokens, `train/loss 1.6289`, `eval/bpb 0.7273`, ~4.77 days wall clock. Note this is W&B evidence: the summary was written while the run was still in flight, and **no GitHub or Discord close-out reporting final numbers exists in the corpus.**
- **The 23.8% is slightly optimistic**, by the summary's own admission: *"the FLOP counter treats attention as fully quadratic and ignores the 2048-token sliding window."*
- Minor naming drift: #6716's title is *"[Hero run] Land 8B-A1B XT on H100s"* — the shipped model was re-scoped to 11B-A1.5B.

The GPU stack also **tracks the TPU reference on quality**: two d512 compute-optimal chains on 8×H100 walked Paloma macro loss down cleanly with each stack change — *"pure main 3.7103, +vectorized-map (VMAP) w_gate fix 3.6954, +256 experts 3.6359"* — with the residual gap to TPU traced to **data, not hardware** (the TPU ablations ran the nemotron mixture). Re-scoring on the shared Uncheatable cache *"flipped the ranking, with datakit ahead"* ([summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).

### The MFU teardown: where the FLOPs actually go

Larry Dial's H100 learning path ([#6979](https://github.com/marin-community/marin/issues/6979)) built the story up from scratch on a single 8×H100 node: *"an lm_head-only cross-entropy benchmark at 50.3% MFU (Liger chunked CE, one chunk per device), a full dense d2560/26-layer model at 56.2% (FlashAttention-4 CUTLASS, ZeRO-1, scan-over-layers with full rematerialization), and then an itemized progression as each correctness feature that makes it a real trainable MoE is switched on — from a 34.4% throughput probe down to 26.7% for the full `model.py`, with real Muon's Newton-Schulz orthogonalization the single biggest give-back at ~3.7 points and each norm only ~0.4–1 point"* ([summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).

Read that ladder correctly: **the MoE-correctness give-backs run 34.4% → 26.7%**, not 56.2% → 26.7%. The dense→MoE step is a separate, larger drop.

### Weak scaling is flat — the week's best structural result

*"model.py weak-scales essentially flat from 1 to 8 nodes — 26.7% at 8 GPUs to 26.5% at 64 H100s and global batch 1024 — because replica/DDP hides the once-per-step gradient all-reduce over the data-center network while keeping the frequent Fully Sharded Data Parallel (FSDP) all-gathers on intra-node NVLink, whereas sharding the model across nodes costs about 4 points"* ([summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).

**This flatness is conditional**, and the condition is the whole trick: it requires the model to fit one node, which needed dropping total experts from 256 to 64 (still top-4). Larry Dial proposed that as the H100 working copy.

### The older, harder-won Hopper baseline

For context on how far this came: the 90B-A5.3B campaign on 256 H100 went *"2.8% → 9.37% MFU (3.3×)"* and called ~9.4% *"the practical pure-FSDP ceiling for this model on 256 H100"*, with a profile showing *"communication 82.3% / compute 13.9% / stall 3.9%"* and 47,550 latency-bound all-gathers per step ([#6304](https://github.com/marin-community/marin/issues/6304#issuecomment-4674696103)). The 25% DoD in that issue title was **not met**. The June headline was *"~19.9 MFU on 4 nodes w/ SGD/Adam"* ([Discord, 2026-06-26](https://discord.com/channels/1354881461060243556/1399998407657001062/1520200189690773564)); the standing workstream target is *"> 20 MFU at 128 GPU scale"* ([#6693](https://github.com/marin-community/marin/issues/6693)), still open.

---

## 4. The three levers being pulled to close the gap

David Hall's **GPU Upskilling Master Plan** ([#6998](https://github.com/marin-community/marin/issues/6998)) lays out four workstreams — *"an XXB-A2B target on Hopper (MFU 20–25% on 128–256 H100s), a Blackwell A8B smoketest, MXFP8 training on Blackwell, and a fused expert-parallel (EP) MoE kernel"* — and names the Hopper obstacles plainly: *"high sparsity plus Adam taxes high-bandwidth memory, the MoE block is slow, and Muon carries high overhead at high sparsity, with pipeline parallelism (PP) the real fix"* ([summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).

### (a) Pipeline parallelism — revived, and the most interesting live thread

PP has a two-act history. **Act one was a negative result:** Russell Power's zero-bubble prototype was gradient-exact and unlocked memory scaling (an 8M-token batch fits under 1F1B where FSDP OOMs), but *"it doesn't beat FSDP on throughput at the 100M–1B / v6e-8 scale (best PP ≈ 22.9k tok/s vs FSDP ≈ 145k tok/s; ~0.78× single-host). PP only wins once the FSDP all-gather crosses DCN (~1.22× FSDP on v6e-32)"* — parked to an external repo, PR #6534 closed unmerged ([#6532](https://github.com/marin-community/marin/issues/6532#issuecomment-4802268924)). On a single 8×H100 node the diagnosis was sharper still: throughput stayed flat at ~8,000 tok/s while the bubble fell 47%→30%, so *"the bottleneck is not the bubble; it is the per-stage-call overhead… ZB would yield ~no speedup on a single GPU node"* ([#6532](https://github.com/marin-community/marin/issues/6532#issuecomment-4790773618)).

**Act two is JaxPP, in exactly the multi-node regime act one identified as PP's only GPU home.** [#7024](https://github.com/marin-community/marin/issues/7024) stood up JaxPP pipeline-parallel MoE training across four 8×H100 nodes on RNO2A. Best measured point — *"explicit `std_1f1b`, batch 8192 / 256 microbatches, four six-layer stages, ring EP, CuTe FA4, eight-warp Pallas-Triton grouped GEMM — reached mean MFU 18.26 at ~414k tokens/s, saturating pipeline occupancy about 1.74 points below the 20 bar"* ([summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)); W&B corroborates a clean monotonic microbatch ladder from 16.10 MFU at 16 microbatches to 18.28 at 256 ([run](https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-l24-e64k4-b8192-s4096-p4m256-20260711-0107)).

Two honest caveats: that *"20 bar"* is the **Hopper** target, not the B200 commitment #6706 — don't cross-compare. And 18.26 has **not** been shown to beat the ~19.9 MFU non-PP 4-node baseline.

What followed was *"a disciplined week of negative results, each gated on H100 A/B evidence before scaling: a latency-hiding scheduler flag (+0.4%, within noise), a pure-XLA output-oriented ring combine (−52%), explicit Triton routing kernels (performance-neutral), transfer-priority task ordering (−2.9%), an input-gradient-first backward split (−29%), and an exact two-chunk bulk ring (−14% fwd+bwd)"*. The most promising lever, an EP-local QuACK/Sonic grouped-MLP adapter, *"cleared the performance gate (1.16× forward, 1.108× fwd+bwd) but failed output parity"* — a ~2.6% mismatch pinned to QuACK's fused approximate SwiGLU, *"turning it into a semantic-policy question rather than a bug."* Still, David Hall's read on July 10: **"Hopper is back on the menu"** ([Discord](https://discord.com/channels/1354881461060243556/1399998407657001062/1525217063830949888)).

### (b) fp8 — real, measured, and still off by default

The fp8 lane crossed from kernels into a training path. Matt Wittmann's **draft** PR [#7079](https://github.com/marin-community/marin/pull/7079) threads stateful fp8 ragged-dot ops through the Grug MoE expert MLP and adds opt-in fp8 over-the-wire collectives — *"E4M3 forward and E5M2 backward on the permutation legs only, reductions staying bf16 since decomposing the reduce-scatter into an fp8 all-to-all measured 0.885× against NCCL's hierarchical path."* At the production d2560/E256/K4 shape *"the full MoE layer measures 1.53× vs bf16 at 2-node expert-parallelism (EP) 16 over InfiniBand on the ring backend (1.65× on all-to-all)"* ([summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).

Earlier measured legs: **1.41× fwd+bwd** for the H100 fp8 ragged-dot expert MLP ([#6880](https://github.com/marin-community/marin/pull/6880)) and **1.438× end-to-end** on 2×8×H100 over IB ([#6911](https://github.com/marin-community/marin/issues/6911#issuecomment-4878544634)).

**The gate is unmet:** *"Defaults stay bf16; the gate before flipping any default is loss-curve validation on a real trajectory, tracked in [#6699](https://github.com/marin-community/marin/issues/6699)."* And the B200 fp8 recipe issue [#5816](https://github.com/marin-community/marin/issues/5816) still reads *"Results: Pending."* Isaac Hodes did close the original "try out fp8 training" issue [#6048](https://github.com/marin-community/marin/issues/6048) as done — kernels proven, defaults unchanged.

### (c) Fused expert-parallel MoE kernels — genuine forward wins, unfinished backward

David Hall's Pallas/Mosaic MGPU source-push MoE ([#6597](https://github.com/marin-community/marin/issues/6597)) is *"in the spirit of SonicMoE (which doesn't handle EP)"* and NVLink-only. At the target shape it hit **2.09× over the ragged all-to-all baseline** on forward (0.03923s vs 0.08204s, matching baseline, zero dropped routes) ([comment](https://github.com/marin-community/marin/issues/6597#issuecomment-4826981254)), and a single-node 8×H100 trainer smoke at **~20% MFU** ([comment](https://github.com/marin-community/marin/issues/6597#issuecomment-4837912107)). He told the GPU channel it is *"showing some promise in terms of actually getting overlap with the all-gathers which nothing else I've tried seems to give me"* ([Discord](https://discord.com/channels/1354881461060243556/1399998407657001062/1523768306526322789)).

But fwd+bwd sits at ~14% roofline (69.05 ms, ~139.95 TFLOP/s/rank) against a *"~250 TFLOP/s"* target ([comment](https://github.com/marin-community/marin/issues/6597#issuecomment-4829033738)), *"backward is still the integration gap"*, and multi-node was blocked on NVSHMEM bootstrap under Iris's one-process-per-GPU topology — then routed around rather than solved. The Blackwell sibling [#6933](https://github.com/marin-community/marin/issues/6933) cleared its 300 TFLOP/s-per-rank forward target (666 useful TFLOP/s/rank steady-state once the staged MLP call is outer-JIT compiled), with backend integration in draft [#6970](https://github.com/marin-community/marin/pull/6970). A parallel CuTe/NVSHMEM transport study [#7114](https://github.com/marin-community/marin/issues/7114) proved out but **the decision was to keep Mosaic** — transport overlap more than halved concurrent GEMM throughput.

---

## 5. The optimizer split: GPU dropped Muon, TPU kept it

This is the cleanest hardware-divergence story in the corpus, and it is **still standing**.

The measurement: David Hall's roofline dashboard found Muon eating *"~500ms of a 1500ms step time in this particular 2 node run"* on H100 ([#6573](https://github.com/marin-community/marin/issues/6573#issuecomment-4775876844)). A single-node A/B was starker — MuonH5 at 4.75 MFU / 1.492 s/step vs SGD at 16.19 MFU / 0.4378 s/step ([#6493](https://github.com/marin-community/marin/issues/6493#issuecomment-4738899640)). The mechanism is comms, not compute: Newton-Schulz wants whole 2D matrices local, conflicting with FSDP's layout, so moving updates back costs an expert-weight-sized payload *"about 121.875 GiB bf16 globally… more or less exactly equal to the FSDP param all-gather… So we are basically doubling the all-gather cost per step"* ([Discord](https://discord.com/channels/1354881461060243556/1399998407657001062/1518073964981293106)).

The decision, from the June 26 OA meeting: *"Muon needs to go for now. 10% step-count improvement isn't worth atm. We'll need much faster interconnect or PP"* ([Discord](https://discord.com/channels/1354881461060243556/1399998407657001062/1520200189690773564)). Note that ~10% is a **step-count/sample-efficiency** gain, not throughput.

**On TPU the same optimizer is nearly free:** on v4-1024 at 21.4% MFU, *"Optimizer NS + bookkeeping: ~206 ms, ~0.9%"* of a ~22.4 s step ([#6493](https://github.com/marin-community/marin/issues/6493#issuecomment-4787096060)). **~0.9% on TPU vs ~33% on 2-node H100** — that gap is the entire argument, and any "Newton-Schulz is ~1% overhead" claim is TPU-only and must not be carried to GPU.

The 67B-A2B TPU hero run and its cooldown remain MuonH. No Muon-on-GPU re-enablement has landed — though PP is the named condition under which it would return, and Russell Power noted *"muon has nice properties around PP wrt stale gradient updates."*

---

## 6. Reliability: mostly won, with one instructive false alarm

The GPU fleet tripled in early July. Russell Power stood up **`marin-rn02a` — 64 nodes × 8 H100 = 512 GPUs**, pinned warm rather than autoscaled since the capacity is prepaid ([#6909](https://github.com/marin-community/marin/pull/6909)). Its first ~1,500-pod soak *"shook a week of hardening out of the Iris control plane"* ([summary 2026-06-29_2026-07-05](https://mws.oa.dev/summaries/summary-2026-06-29_2026-07-05.html)).

Fixed, mostly within days:

- **Multi-node NCCL bootstrap.** `NCCL_SOCKET_IFNAME` used NCCL's `=` exact-match prefix, which the task image's NCCL build doesn't honor — *"Bootstrap : no socket interface found"*. Switched to the exclude-list form across all CoreWeave clusters ([#6940](https://github.com/marin-community/marin/issues/6940) → [#6941](https://github.com/marin-community/marin/pull/6941)), verified on 8 nodes / 64×H100. A recurrence on cw-rno2a ([#7018](https://github.com/marin-community/marin/issues/7018)) was closed as already-fixed by that change.
- **cuDNN precedence.** jaxlib was loading cuDNN 9.10.2 from CUDA 12 torch deps despite the CUDA 13 stack ([#6947](https://github.com/marin-community/marin/pull/6947)); a follow-on where an `--offline` reinstall missed the uv cache and aborted the GPU canary was dropped in [#7031](https://github.com/marin-community/marin/pull/7031).
- **Control plane.** Kueue's webhook scoped to Iris's namespace ([#6894](https://github.com/marin-community/marin/pull/6894)); controller SQLite moved off NFS to node-local NVMe after OOM and liveness-probe kills ([#6943](https://github.com/marin-community/marin/pull/6943), [#6945](https://github.com/marin-community/marin/pull/6945)); an `iris-system` PriorityClass above every user band ([#6929](https://github.com/marin-community/marin/pull/6929)).
- **Federation went live.** Marin can now hand whole jobs to CoreWeave ([#7064](https://github.com/marin-community/marin/pull/7064)), with federated jobs parking as `QUEUED_HANDOFF` until a peer reports free capacity ([#7108](https://github.com/marin-community/marin/pull/7108)) and child-cluster endpoints reachable through the parent `/proxy` ([#7109](https://github.com/marin-community/marin/pull/7109)).

**The instructive one — do not repeat the wrong version.** [#6950](https://github.com/marin-community/marin/issues/6950) opened as an InfiniBand fabric fault: two 128k-vocab arms hung ~7 minutes in, every attempt, sharing five leaf-group-71 nodes. Cordoning that leaf group **did not help** (*"This refutes 'leaf-71 is the sole culprit'"*), and purpose-built reproducers cleared both the cross-leaf fabric and S3 starvation. Root cause, same day: *"it is not the fabric, not S3 starvation, and not the autotune cache. It is a tokenizer staging thread-race"* ([comment](https://github.com/marin-community/marin/issues/6950#issuecomment-4883906498)) — CPython's `lru_cache` doesn't serialize concurrent first-calls, two of eight tasks died on an uncaught HF-Hub 404, the survivors parked forever in the first collective. Fixed in [#6955](https://github.com/marin-community/marin/pull/6955); *"the first 128k-plain arm to ever survive."*

One notable **still-open** item: a monitor to detect stalled GPU processes from their NCCL log signatures ([#6938](https://github.com/marin-community/marin/issues/6938)).

There was also a silent, expensive default: *"levanter's GPU default attention backend was NVTE (Transformer Engine), which no marin environment ships — the gpu extra installs flash-attn-4 instead — so every dense-LM GPU run silently fell back to an unfused reference path that also OOMs at sequence 8192"* ([#7013](https://github.com/marin-community/marin/issues/7013)); the FA4 fix ([#7015](https://github.com/marin-community/marin/pull/7015)) is *"3.4× faster than the fallback (11.3 vs 38.9 ms/iter on an H100)."* Scope matters: this hit the **dense-LM** path — Grug MoE runs were already on `gpu_fa4_cute`, so the hero-run path was not silently degraded.

---

## 7. What's decided, what's open

| | Status |
|---|---|
| H100 stack correctness & quality parity with TPU | **Done** — d512 ablation chain tracks; residual gap is data, not hardware ([summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)) |
| 64×H100 500B-token validation run | **Completed** ([W&B](https://wandb.ai/marin-community/marin_moe/runs/grug-d2048-L24-gqa4-500B-r8-nosim-v2-20260709-042053)) |
| Muon on GPU | **Shelved**, conditionally on PP ([#6693](https://github.com/marin-community/marin/issues/6693)) |
| Attention for the September run | **GQA** (working call), MLA rejected on FLOP accounting ([#6522](https://github.com/marin-community/marin/issues/6522)); MFU check at d5260 pending in [#6889](https://github.com/marin-community/marin/issues/6889) |
| B200 ≥20% MFU | **Open** — 17.8% single-node ([#6706](https://github.com/marin-community/marin/issues/6706)) |
| B200 25% for September | **Open target** ([#6710](https://github.com/marin-community/marin/issues/6710)) |
| >20 MFU at 128 H100 | **Open** ([#6693](https://github.com/marin-community/marin/issues/6693)) |
| fp8 as default | **Open** — gated on loss-curve validation ([#6699](https://github.com/marin-community/marin/issues/6699)) |
| Data mix, architecture, preregistered loss for the B200 run | **Open**, 4/14 sub-issues closed, *"no decision was recorded this week"* ([#7073](https://github.com/marin-community/marin/issues/7073)) |
| Multi-node fused Pallas/Mosaic EP kernel | **Blocked/routed around** ([#6597](https://github.com/marin-community/marin/issues/6597)) |

---

## 8. Gaps and cautions in this answer

- **Corpus boundary.** The GitHub-issue corpus ends around **2026-07-05**; weekly summaries run through **2026-07-12**; W&B records extend to ~07-16. Everything numbered ≳#6960 (#7012, #6979, #6998, #7024, #7079, #7013, #7015, #7073) is **summary-sourced only** — I could not cross-check those figures against primary issue text. Treat them as single-sourced.
- **No B200 rack-scale MFU exists anywhere in the corpus.** All Blackwell MFU figures are single-node 8×B200.
- **No preregistered loss value** exists for either upcoming GPU run; [#6702](https://github.com/marin-community/marin/issues/6702)'s body is empty and #6689's pre-registered-loss cell is blank.
- **No post-completion writeup** for the 500B H100 validation run — no final Paloma number was reported in the corpus.
- **Rack-count conflict** for the end-of-month run (2 vs 4 NVL72) is unresolved in-corpus.
- A separate **MLA variant** of the d2048 shape was launched July 13–14 and is still running at 32 H100s ([W&B](https://wandb.ai/marin-community/marin_moe/runs/grug-d2048-L24-mla-500B-r4-b512-datakit-20260714-001739)) — do not conflate it with the finished GQA run.

---
<!--provenance-->
> *Data: marinmirror — 68026 chunks, built 45h ago · summaries through 2026-07-06_2026-07-12. Refresh disabled for this run (frozen eval corpus); GitHub-issue coverage ends ~2026-07-05.*
>
> *Query: "explain how training on GPUs is going"*
>
> *Sub-queries: "B200 / NVL72 MFU push toward 20% and 25% — QuACK SonicMoE grouped GEMM, width vs batch lever, fp8 ragged dot" · "GPU Upskilling Master Plan and the H100/Hopper MoE MFU learning path — per-feature itemization, Muon Newton-Schulz overhead, weak scaling, expert count" · "Pipeline parallelism for MoE training on GPUs — JaxPP mpmd 1F1B, zero-bubble prototype parked, fused EP Pallas/Mosaic kernel" · "Marin 11B-A1.5B hardware and loss validation run on 64 H100s at CoreWeave — MFU, 500B token horizon, d512 ablation chain" · "CoreWeave GPU cluster infrastructure and reliability — RNO2A 512 H100s, NCCL bootstrap, cuDNN staging, GPU canaries, Iris federation" · "Muon shelved on GPU vs kept on TPU; GPU attention backend NVTE vs FlashAttention-4 default bug" · "Upcoming GPU hero runs — end-of-month ~120B-A8B on 2 NVL72, September ~512B-A16B, and what gates them" · plus an adversarial verification pass over 10 load-bearing numeric claims*
