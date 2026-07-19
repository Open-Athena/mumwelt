# Inference at Marin: who, where it stands, performance, and open issues

## TL;DR

Inference is currently **five parallel lanes with one clear owner**: **Romain Yon (`yonromai`)** owns the GrugMoE-on-vLLM serving stack (TPU *and* GPU) under the July epic [#6867](https://github.com/marin-community/marin/issues/6867); **David Hall (`dlwh`)** owns Levanter's own inference/rollout backend ([#6227](https://github.com/marin-community/marin/issues/6227)); **Russell Power (`rjpower`)** owns serving infra (`marin-serve`, [#6556](https://github.com/marin-community/marin/pull/6556)); **Isaac Hodes (`ihodes`)** owns the rollout-speed investment area ([#6709](https://github.com/marin-community/marin/issues/6709)); and **Benjamin Feuer (`penfever`)** is the downstream RL consumer.

**Status: correctness has largely landed; performance has not been measured.** The TPU path closed as completed on 2026-07-02, and a "v0" of GPU inference landed in `main` on 2026-07-14. But the headline answer to "what's the current performance" is that **no GrugMoE inference throughput number exists anywhere in the corpus** — the acceptance bar in [#6870](https://github.com/marin-community/marin/issues/6870) is a literal unfilled `TODO`, and [#6709](https://github.com/marin-community/marin/issues/6709)'s stated first task — measure a baseline — had not yet been done as of the corpus edge.

---

## 1. Who is doing the work

| Lane | Owner | Tracking |
|---|---|---|
| GrugMoE serving on vLLM (TPU + GPU) | **Romain Yon** (`yonromai` / "romain") | [#6867](https://github.com/marin-community/marin/issues/6867), [#6868](https://github.com/marin-community/marin/issues/6868), [#6869](https://github.com/marin-community/marin/issues/6869), [#6870](https://github.com/marin-community/marin/issues/6870) |
| TPU fork overlays, worker bring-up | **Rohith Kuditipudi** (`rohithck`) | [#7085](https://github.com/marin-community/marin/issues/7085), #7025 |
| `marin-serve`, serving infra, fork consolidation | **Russell Power** (`rjpower`) | [#6556](https://github.com/marin-community/marin/pull/6556), #7097 |
| Levanter inference + RL rollout backend parity | **David Hall** (`dlwh`) | [#6227](https://github.com/marin-community/marin/issues/6227), [#1898](https://github.com/marin-community/marin/issues/1898) |
| Rollout-speed strategy / target-setting | **Isaac Hodes** (`ihodes`) | [#6709](https://github.com/marin-community/marin/issues/6709) |
| Brokered logit mixing; serving-reliability bugs | **Will Held** (`willheld`) | #7113, #6983 |
| RL consumer (MarinSkyRL) | **Benjamin Feuer** (`penfever`) | #7052, #7164 |

Romain Yon is the centre of gravity: he filed the entire July epic tree and authored [#6664](https://github.com/marin-community/marin/pull/6664), [#6733](https://github.com/marin-community/marin/pull/6733), [#6891](https://github.com/marin-community/marin/pull/6891), [#6965](https://github.com/marin-community/marin/pull/6965) and [#6966](https://github.com/marin-community/marin/pull/6966).

---

## 2. Where it's at

### The epic structure

The June epic [#6041 "Inference: GrugMoE support in vLLM TPU"](https://github.com/marin-community/marin/issues/6041) **closed as completed on 2026-07-02**. Its close-out (yonromai):

> "Closing as completed. The GrugMoE TPU vLLM path landed in #6664, including the export path, Marin-owned vLLM/tpu-inference fork pins, and a real-checkpoint TPU e2e comparing vLLM serving against the Levanter/JAX reference path."

Both June issues were **deliberately correctness-only**. [#6042](https://github.com/marin-community/marin/issues/6042) and #6041 carry the identical scope note:

> "Correctness is tested, perf is okay (good enough for evals). Out of scope: - Perf good enough for serious inference (e.g. RL). - Upstreaming"

Romain Yon then opened [#6867 "[Epic] July Grug Inference tasks"](https://github.com/marin-community/marin/issues/6867) the same day. Its entire body:

> "DoD: Support full size GrugMoE model on both TPUs and GPUs. Stretch: Inference is fast enough on GPUs."

Note **performance is explicitly a stretch goal, ranked below correctness**. The three sub-issues, all opened 2026-07-02:

- [#6868](https://github.com/marin-community/marin/issues/6868) — "vLLM is able to *correctly* do inference on largest GrugMoE checkpoint to date, on TPUs."
- [#6869](https://github.com/marin-community/marin/issues/6869) — the same, "on GPUs."
- [#6870](https://github.com/marin-community/marin/issues/6870) — "DoD: vLLM GrugMoE Inference is fast enough* for our RL needs on GPUs. **Fast enough: TODO: Within X% of Jax on GPUs vs. within Y% of reference model (e.g. Qwen?)**"

### TPU: landed, but with a live hardware gap

The TPU path landed via [#6664](https://github.com/marin-community/marin/pull/6664) ("correctness-first GrugMoE support for Marin's TPU vLLM path") and was carried forward by [#6733](https://github.com/marin-community/marin/pull/6733), which pinned the Marin-owned `vllm` and `tpu-inference` forks to `tpu-inference v0.23.0`, converging on `jax==0.10.1` / `libtpu==0.0.41`.

**Validation was on v6e-4 only.** The #6041 close-out lists as out of scope: "broad throughput/latency coverage, tensor/pipeline parallel serving, upstreaming, and wider router/logprob coverage."

That narrow validation surface is likely why **v4 TPUs broke undetected**. Rohith Kuditipudi, 2026-07-10 in `#code-talk`: ["seems I'm unable to run inference on v4-8s"](https://discord.com/channels/1354881461060243556/1366632114316906506/1525023853258997821), filing [#7085](https://github.com/marin-community/marin/issues/7085). Romain replied "Ill take a look, thanks for filing" and hours later posted a fix titled **"[tpu-inference] Support RPA v3 on TPU v4..."** — i.e. ragged paged attention v3 lacks a v4 case. *Caveat: that fix lives in the `marin-community/tpu-inference` fork repo, which is outside this corpus, so I cannot confirm it landed.*

During July the fork pins consolidated further (#7025, #7094), and Russell Power opened #7097 to unify them so Grug needs only a single patch set ([week of Jul 6 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).

### GPU: expert-parallel correctness proven, then a v0 landed

The GPU sibling [#6042](https://github.com/marin-community/marin/issues/6042) closed when eight-way expert-parallel (EP8) serving of a real GrugMoE checkpoint passed on CoreWeave/H100 ([week of Jul 6 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).

The evidence is [#6891](https://github.com/marin-community/marin/pull/6891) (2026-07-03): a harness serving checkpoint `moe_may_compute_opt_d512_ep1-05c39b/checkpoints/step-10980` with **TP=1, DP=8, EP=8, `max_num_seqs=16`** on an **8×H100 CoreWeave** node, compared against a JAX/Levanter reference. It [passed](https://github.com/marin-community/marin/issues/6891#issuecomment-4872341972):

> "Remote pytest: 3 passed in 274.29s… vLLM observed TP=1, DP=8, EP=8… DP ranks covered by X-data-parallel-rank: 0..7. Routed expert owner ranks covered: 0..7. vLLM/Levanter batch match: true."

**Its scope was tightly bounded by the author**: "This validates short greedy parity and routed expert coverage for one real checkpoint and one prompt batch, **not performance, sampling parity, logprob parity, broad context windows, or long-running stability**."

Probing GPU attention backends surfaced a genuine model-side bug, [#6964](https://github.com/marin-community/marin/issues/6964): Grug MoE's Exclusive Self-Attention value correction assumed one hard-coded query-head sharding, while "GPU attention backends can choose a backend-specific layout for the attention output." Fixed in [#6965](https://github.com/marin-community/marin/pull/6965). #6891 was then [closed as superseded](https://github.com/marin-community/marin/issues/6891#issuecomment-4887307764) by [#6966](https://github.com/marin-community/marin/pull/6966), which restacks the harness on the fix with "dual TRITON_ATTN/FLASH_ATTN summary gating."

**Latest state (2026-07-14)** — the freshest primary signal in the corpus, romain in `#inference`:

> ["FYI a v0 of Grug 67b a2b inference on GPUs has landed in main."](https://discord.com/channels/1354881461060243556/1385733711013871729/1526388201915289642) … "I'm sure there'll be many rough edges, and I'm still actively working on it."

Per the weekly summary, full-size 67B coverage arrived via #7116 (an 8×H100 BF16 test asserting the token after "The United States Of") and #7117 (a 134 GB HF export over 39 shards).

### Serving surface: `marin-serve` is TPU-only at the freeze

[#6556](https://github.com/marin-community/marin/pull/6556) (rjpower, 2026-06-22) describes `marin-serve` as booting "vLLM on a single-host slice" of "an Iris TPU slice," verified only on v5litepod-4 and v6e-4. There is no `--gpu` path in-corpus, and it is structurally blocked: `marin-core[vllm]` hard-wires `VLLM_TARGET_DEVICE = "tpu"` and remains mutually exclusive with `marin-core[gpu]` ([#6485](https://github.com/marin-community/marin/pull/6485), [#6646](https://github.com/marin-community/marin/pull/6646)). Per the weekly summary, #7111/#7133 opened a CUDA path that provisions vLLM per-job via `uvx`, but it "still needs a real H100 boot test."

---

## 3. Current performance

**This is the weakest-evidenced part of the picture, and the honest answer is that the number does not exist yet.**

- **No GrugMoE inference throughput, latency, or decode-speed figure appears anywhere in the corpus**, on GPU or TPU. The only number attached to the GPU validation is test wall-clock ("3 passed in 274.29s"), which is a correctness harness, not a benchmark.
- **The bar itself is undefined.** [#6870](https://github.com/marin-community/marin/issues/6870)'s DoD still reads "Within X% of Jax on GPUs vs. within Y% of reference model" — X and Y are unfilled placeholders, with zero comments on the issue.
- **The baseline was never taken.** [#6709 "Inference speed (for RL rollouts)"](https://github.com/marin-community/marin/issues/6709) (ihodes, 2026-06-26) targets "Focus on GPUs, H100s in particular, where we intend do most of our RL this year," and its stated first task is to measure and document the current baseline before setting a target. The [week of Jul 6 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html) confirms it is *"still at the baseline-establishing stage"*, with "No sub-issues, PRs, or design docs landed against it this week."

### The measured numbers that *do* exist — different lanes, do not transfer to GrugMoE

**(a) Levanter vs vLLM rollout parity — Qwen3-8B dense, TPU** ([#6227](https://github.com/marin-community/marin/issues/6227), David Hall). The epic tracks "the work to make Levanter TPU inference a reliable RL rollout backend for dense 8B-class models," and both terminal benchmark rows were explicitly marked **`target fail`** against its parity DoD. (The exact numeric threshold sits in the epic body, which this corpus truncates, so I am not quoting a figure for it.)

| Config | vLLM-TPU | Levanter | Ratio | Verdict |
|---|---|---|---|---|
| v6e-8, TP=8, `prefill_b8_i2048_o128_n1` | 1264.03 decode tok/s / 21488.50 total | 957.72 decode / 16281.24 total | **0.758** | [target fail](https://github.com/marin-community/marin/issues/6227#issuecomment-4641430081) |
| v5p-8, TP=4, `mixed_b32_i512_o512_n1` | 5215.67 decode / 10431.34 total | 3131.59 decode / 6263.19 total | **0.600** | [target fail](https://github.com/marin-community/marin/issues/6227#issuecomment-4641836979) |

⚠️ **Staleness:** all #6227 activity stops on 2026-06-07 — roughly six weeks before the corpus edge. These are the latest in-corpus numbers for that lane, but the epic went quiet afterwards; treat them as a June snapshot, not necessarily today's state.

**(b) End-to-end RL step time — Qwen MoEs on H100** (Benjamin Feuer, 2026-07-09, primary source): ["Qwen3 30B A3B and Qwen3.5 35B A3B both fit on 64xH100 (8 nodes). ~2.5hr / step at the moment which is slower than I would like but not insane"](https://discord.com/channels/1354881461060243556/1524748716190339122/1524749076355088485), with ["~10M tokens generated per step"](https://discord.com/channels/1354881461060243556/1524748716190339122/1524801933553041682) (tracked in #7052). This is the demand signal #6709 exists to address — but it is **Qwen, not GrugMoE**.

**(c) A relative Levanter speedup** (David Hall, 2026-06-01): ["i threw goal mode at making levanter inference faster and it made it ~10x faster (mostly some dumb things we were doing) which is within ~12% of vllm tpu"](https://discord.com/channels/1354881461060243556/1385733711013871729/1511034914059976784). Relative only, no absolute tok/s, TPU.

---

## 4. Open issues

**Blocking / unresolved at the corpus edge:**

1. **v4 TPU inference broken** — [#7085](https://github.com/marin-community/marin/issues/7085): RPA v3 has no TPU v4 case; Rohith "unable to run inference on v4-8s." A tpu-inference fix was in flight 2026-07-10; landing unconfirmed.
2. **Parallel vLLM worker bring-up fails** — Rohith, 2026-07-10 in `#infra`, [verbatim](https://discord.com/channels/1354881461060243556/1364827114670657616/1525035072414023750): "Failed to download and build `vllm @ git+...@912fb118...` ╰─▶ Timeout (300s) when waiting for lock… `ModuleNotFoundError: No module named 'zephyr'`". Romain: "I'll take a look once I get to the office." **No resolution appears through the corpus edge** — this is the clearest live blocker.
3. **`WorkerProc initialization failed`** on split-environment CoreWeave runs (both FLASH_ATTN and TRITON_ATTN). Classified as "remote GPU-wheel/source bootstrap drift rather than a FLASH-specific correctness mismatch" — a guess, never root-caused ([#6891](https://github.com/marin-community/marin/pull/6891)).
4. **RL is blocked on fork sync.** Benjamin Feuer, 2026-07-14: ["I won't be able to do RL until we sync up on the vLLM branch + fork"](https://discord.com/channels/1354881461060243556/1385733711013871729/1526394430637080576). Romain agreed to prioritise porting. A second blocker — a training `GrugMoeForCausalLM` PyTorch class for MarinSkyRL — was [filed as #7164](https://discord.com/channels/1354881461060243556/1385733711013871729/1526568131328479242).
5. **TPU-vs-GPU logprob divergence** — #7183, raised by romain 2026-07-15; Russell Power: ["hopefully the golden run with some longer outputs will be helpful in determining whether this is expected or something we should worry about"](https://discord.com/channels/1354881461060243556/1385733711013871729/1526781045880520745). A live risk against #6869's "*correctly*" bar.
6. **The brokered-serving "wedge"** — #6983 (Will Held). Romain's instrumented rerun overturned the original lease-leak theory: the plateau was fast vLLM HTTP 400s once the prompt crossed marin-8b-base's true 4096-token position limit (`Out of range float values are not JSON compliant: nan`, masked by `VLLM_ALLOW_LONG_MAX_MODEL_LEN`). **Net: the original field wedge remains genuine but unreproduced**, and the silent NaN-400 behaviour past max position was flagged as possibly deserving its own issue ([week of Jul 6 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).
7. **All four July epic issues open** — #6867, #6868, #6869, #6870 carry no close-out comments.
8. **#6965/#6966 unmerged in-corpus** — no review, merge, or post-restack CoreWeave validation recorded at the freeze.

**Resolved (for contrast):** the `ncclAlltoAll` failure [#5377](https://github.com/marin-community/marin/issues/5377) was closed 2026-06-25 ("NCCL version conflict fixed in PR #5379 by pinning torch dep") — note this was **training-side**, not vLLM serving. [#6940](https://github.com/marin-community/marin/issues/6940) (`NCCL_SOCKET_IFNAME` breaking multi-node bootstrap) has a fix in #6941 whose CI verification box was still unchecked.

**Open question, unanswered:** whether MarinSkyRL will need vLLM's native RL APIs to roll out GrugMoE. Romain asked in `#reinforcement-learning`; rjpower's earlier read ([#6041](https://github.com/marin-community/marin/issues/6041), 2026-06-24) was "we want vLLM eventually (and unfortunately it's probably a hard requirement for SkyRL) but it's not a blocker for eval per se."

---

## 5. Corpus coverage caveat — please read before relying on the 7xxx items

The frozen corpus has **asymmetric coverage**. Its GitHub side stops carrying issue/PR bodies at **#6967 (2026-07-05)**, while Discord runs to 2026-07-16 and weekly summaries to 2026-07-12.

Consequently, **every 7xxx item cited above** (#7025, #7085, #7094, #7097, #7106, #7107, #7111, #7113, #7116, #7117, #7133, #7137, #7164, #7183, #6983, #7052) is sourced from the **weekly summary narrative or Discord references, not from the primary issue body**. Their open/closed status cannot be confirmed here. Anything requiring live status on those needs GitHub directly.

Where a summary claim conflicted with primary sources, I preferred the primary: e.g. the `ncclAlltoAll` errors verifiable in-corpus are training-side (#5377, #6940), not vLLM serving-path failures.

---
<!--provenance-->
> *Data: marinmirror — 68026 chunks, built 48h ago · summaries through 2026-07-06_2026-07-12. Frozen eval corpus (MARIN_EVAL_FREEZE=2026-07-16, retrieval pool k=20); refresh disabled, no refresh triggered this run. GitHub bodies stop at #6967 (2026-07-05); Discord extends to 2026-07-16.*
>
> *Query: "Who is doing inference work and where is it at? What's the current performance and are there any open issues?"*
>
> *Sub-queries: "July Grug Inference epic #6867 — scope, DoD, owners, status of sub-issues #6868/#6869/#6870" · "TPU vLLM serving stack — fork pins #7025/#7094/#7097/#6733, RPA v3 v4-8 NotImplementedError #7085, uv git-lock and zephyr worker failures" · "GrugMoE vLLM on GPUs — EP8 parity on 8×H100 CoreWeave (#6891/#6966/#6042/#6869), XSA sharding bug (#6964/#6965), WorkerProc and NCCL failures" · "marin-serve CUDA/GPU serving path — #7111/#7133/#7106/#7107, torch 2.11/CUDA 13 conflicts, packaged-library route" · "GrugMoE inference performance — throughput, latency, the 'fast enough' bar for RL rollouts (#6870), rollout-speed epic, vLLM wedge / 4096-token NaN, MarinSkyRL rollout throughput" · "vLLM regression harnesses and checkpoint export — #7117/#7116/#7137, QB router bias, brokered logit mixing #7113"*
