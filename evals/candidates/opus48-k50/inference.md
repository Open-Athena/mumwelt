# Inference at Marin: who, where it stands, performance, and open issues

**Short version.** Inference work is led by **Romain Yon (yonromai)**, who owns the July inference epic and both the TPU and GPU vLLM serving paths; **Russell Power (rjpower)** owns the serving *infrastructure* underneath it (`marin-serve`, endpoint ingress, federation); **Isaac Hodes (ihodes)** owns the RL-rollout-speed epic; **Benjamin Feuer (penfever)** and **Rohith Kuditipudi (rohithck)** are the main consumers pushing on it; **Will Held (willheld)** is doing serving-side plumbing and reliability work. The state of play is that **correctness has largely landed and performance has barely started**. A v0 of 67B GrugMoE inference on GPUs landed in main on 2026-07-14, explicitly hedged as rough. The headline finding on performance is a negative one: **there is no measured H100 rollout-serving baseline anywhere in the corpus**, and the "fast enough" bar is still a literal `TODO`.

---

## 1. Who is doing the work

| Person | Scope | Evidence |
|---|---|---|
| **Romain Yon** (yonromai) | Owns the July inference epic and all three sub-issues; authored the TPU GrugMoE support, the GPU EP8 validation, the fork pins, and the 67B export/regression harnesses | [#6867](https://github.com/marin-community/marin/issues/6867), [#6664](https://github.com/marin-community/marin/pull/6664), [#6891](https://github.com/marin-community/marin/pull/6891), [#6733](https://github.com/marin-community/marin/issues/6733) |
| **Russell Power** (rjpower) | Serving infrastructure: `marin-serve`, endpoint auth/ingress, cross-cluster federation | [#6556](https://github.com/marin-community/marin/pull/6556), [#6857](https://github.com/marin-community/marin/pull/6857), [#7034/#7064](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html) |
| **Isaac Hodes** (ihodes) | Opened and owns the RL rollout-speed epic; logged the #6042 close-out | [#6709](https://github.com/marin-community/marin/issues/6709), [weekly](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html) |
| **Benjamin Feuer** (penfever) | Main RL/agentic consumer — 131k MoE RL on CoreWeave, TPU SWE-bench parity, off-cluster sandbox ingress asks | [#7052 via Discord](https://discord.com/channels/1354881461060243556/1524748716190339122/1524749076355088485), [#6958](https://github.com/marin-community/marin/issues/6958), [#6847](https://github.com/marin-community/marin/issues/6847) |
| **Will Held** (willheld) | Serving reliability (the "wedge" investigation) and brokered logit mixing | [#6983 via weekly](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html), [#7113 via weekly](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html) |
| **Rohith Kuditipudi** (rohithck) | TPU inference consumer; filed the v4-8 blocker and carries fork patches | [#7085 via Discord](https://discord.com/channels/1354881461060243556/1366632114316906506/1525023853258997821) |

One attribution conflict worth flagging: the weekly credits **Isaac Hodes** with logging the #6042 close-out, while all the underlying EP8 engineering visible in primary threads is **Romain Yon's** ([#6891](https://github.com/marin-community/marin/pull/6891)). Both readings are consistent — Hodes closed the issue on Yon's evidence.

## 2. Where it's at

### The governing epic

**[#6867 "[Epic] July Grug Inference tasks"](https://github.com/marin-community/marin/issues/6867)** (Yon, 2026-07-02). The entire body is two lines:

> "DoD: Support full size GrugMoE model on both TPUs and GPUs. Stretch: Inference is fast enough on GPUs."

Note the shape of that: **correctness is the commitment, speed is only a stretch**. Three sub-issues, **all still open**:

- **[#6868](https://github.com/marin-community/marin/issues/6868)** — correct vLLM inference on the largest GrugMoE checkpoint, **on TPUs**. No recorded progress.
- **[#6869](https://github.com/marin-community/marin/issues/6869)** — same, **on GPUs**.
- **[#6870](https://github.com/marin-community/marin/issues/6870)** — GPU perf "decent". Body verbatim: *"Fast enough: TODO: Within X% of Jax on GPUs vs. within Y% of reference model (e.g. Qwen?)"* — **both variables are unfilled and even the reference model is unchosen.**

Its two predecessor epics both closed in early July: **[#6041](https://github.com/marin-community/marin/issues/6041)** (TPU) closed 2026-07-02 on the strength of [#6664](https://github.com/marin-community/marin/pull/6664), and **[#6042](https://github.com/marin-community/marin/issues/6042)** (GPU) closed a week later on the EP8 result. Both had explicitly scoped out *"Perf good enough for serious inference (e.g. RL)"*.

### GPU track — furthest along, freshest, and self-described as rough

The best-evidenced achieved result is **eight-way expert-parallel serving on 8×H100 CoreWeave**, from [#6891](https://github.com/marin-community/marin/pull/6891) (2026-07-03):

> "PR-head CoreWeave validation passed. … Remote pytest: 3 passed in 274.29s — vLLM observed TP=1, DP=8, EP=8; max_num_seqs=16; TRITON_ATTN — Levanter/JAX mesh: data=1, expert=8, model=1 — … vLLM/Levanter batch match: true"

Critically, that ran on the **`moe_may_compute_opt_d512` step-10980 checkpoint — not the 67B model** — and the author bounds it himself: *"This validates short greedy parity and routed expert coverage for one real checkpoint and one prompt batch, **not performance**, sampling parity, logprob parity, broad context windows, or long-running stability."* It was superseded by [#6966](https://github.com/marin-community/marin/pull/6966), which depends on the XSA/GQA sharding fix in [#6965](https://github.com/marin-community/marin/pull/6965).

The **most current** state — and the single most useful line for "where is it at" — is Romain Yon in `#inference`, [2026-07-14](https://discord.com/channels/1354881461060243556/1385733711013871729/1526388201915289642):

> "FYI a v0 of Grug 67b a2b inference on GPUs has landed in main. … I'm sure there'll be many rough edges, and I'm still actively working on it."

Two 67B regression harnesses landed behind it, per the [weekly](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html): **#7116** (merged) added an 8×H100 BF16 test loading step-18000 and asserting the next token after "The United States Of" is "America" with a top-25 logprob golden check; **#7125** (Rafal Wojdyla) fixed the CI redness it exposed.

### TPU track — correctness landed on a small checkpoint, consolidating forks

[#6041](https://github.com/marin-community/marin/issues/6041) closed with Yon writing:

> "Closing as completed. The GrugMoE TPU vLLM path landed in #6664, including the export path, Marin-owned vLLM/tpu-inference fork pins, and a real-checkpoint TPU e2e comparing vLLM serving against the Levanter/JAX reference path."

The underlying evidence is narrow: [#6664](https://github.com/marin-community/marin/pull/6664) validated on a **v6e-4 in europe-west4**, `Pytest: 2 passed, 1 deselected in 215.03s`, with vLLM and Levanter/JAX both emitting `" The Ultimate. The Ultimate. The Ultimate"` — a string-equality smoke assertion. Yon's own caveat: *"It does not validate throughput/latency, tensor/pipeline parallel serving, broad context windows, router replay, or vLLM routed-expert/logprob exposure."*

Fork-pin consolidation dominated the last two weeks: [#6733](https://github.com/marin-community/marin/issues/6733) moved to tpu-inference v0.23.0 on jax 0.10.1 / libtpu 0.0.41, then **#7025** advanced pins to landed marin-community main SHAs and **#7094** made TPU builds skip the default Rust artifacts ([weekly](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).

### Serving infrastructure

`marin-serve` ([#6556](https://github.com/marin-community/marin/pull/6556), Power, 2026-06-22) is the one-liner front door: boots vLLM on a single-host Iris slice behind a dashboard and an OpenAI-compatible API. Endpoint auth landed via [#6857](https://github.com/marin-community/marin/pull/6857), giving each Iris endpoint a PRIVATE/PUBLIC/BEARER access mode with controller-minted scoped tokens — motivated by [#6847](https://github.com/marin-community/marin/issues/6847), where off-cluster Daytona sandboxes were reaching on-cluster vLLM through *one unauthenticated paid pinggy tunnel per job*. Federation now mirrors child-cluster endpoints up through the parent's `/proxy`, so **#7109** makes "a GPU serving job pinned to CoreWeave usable without tunneling to the child" ([weekly](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).

## 3. Current performance

**The central fact: no measured serving/rollout throughput baseline for GrugMoE exists on any hardware in this corpus.** Every GPU and TPU result to date is greedy-parity correctness. [#6709](https://github.com/marin-community/marin/issues/6709) names measuring it as task one — *"First task: measure and document the current baseline(s) before setting a target"* — and the issue has **zero comments**.

What performance numbers *do* exist, each tagged to its actual hardware and workload:

**RL step cost at 131k context (measured, most decision-relevant).** Benjamin Feuer, [2026-07-09](https://discord.com/channels/1354881461060243556/1524748716190339122/1524749076355088485):

> "Qwen3 30B A3B and Qwen3.5 35B A3B both fit on 64xH100 (8 nodes). ~2.5hr / step at the moment which is slower than I would like but not insane."

with *"A **very** ballpark estimate would be ~10M tokens generated per step"*. He adds that Qwen3 Next 80B A3B — *"the most comparable in scale to the incoming Marin 67B A2B"* — will likely need 96 or 128×H100 to fit, and that *"131k is probably the limit of what we can achieve on this hardware without additional cleverness."* This is the demand signal the rollout-speed epic exists to address.

**Generation throughput, all TPU, all predating the GPU epic.** On **v5p-8, Llama 8B, MATH**: Levanter 762 tok/s vs vLLM 4,749 (c=64) and **7,940 tok/s (c=256)** — *"Levanter is 6-10x slower than vLLM. The gap is structural"* ([#3826](https://github.com/marin-community/marin/issues/3826#issuecomment-4115721362), 2026-03-18). On **v4-8, Llama-3.2-1B in-RL**: **~3000 tok/s** after a measurement correction, against a ~5300 tok/s vLLM-in-isolation ceiling ([#1823](https://github.com/marin-community/marin/issues/1823#issuecomment-3478748680), [#1827](https://github.com/marin-community/marin/issues/1827)). That 6-10x gap was partly closed on TPU three months later — David Hall, [2026-06-01](https://discord.com/channels/1354881461060243556/1385733711013871729/1511034914059976784): *"i threw goal mode at making levanter inference faster and it made it ~10x faster … which is within ~12% of vllm tpu."*

**TPU/GPU eval parity (achieved).** [#6958](https://github.com/marin-community/marin/issues/6958#issuecomment-4886598435): a terminus-2 agent on SWE-bench-Verified random-100, N=300, scored **0.240 resolved on a preemptible v6e-4** against **0.237 on the H100 SLURM reference** — TPU serving is eval-grade for that one model. Qwen3-32B and Qwen3.5-9B parity was never tested.

**Do not carry these across:** the ~101k–169k tokens/s and 2.75–4.59% MFU figures on 8×H100 in [#6237](https://github.com/marin-community/marin/issues/6237#issuecomment-4641909554) are **GrugMoE training** canaries, not generation.

## 4. Open issues

**Blocking the epic's own definition of done:**

1. **[#6868](https://github.com/marin-community/marin/issues/6868) / [#6869](https://github.com/marin-community/marin/issues/6869) — full-size GrugMoE correctness** is not discharged on either accelerator by validated harnesses. The GPU EP8 evidence used the d512 step-10980 checkpoint; the 67B GPU path is a self-described "v0 … many rough edges."
2. **[#6870](https://github.com/marin-community/marin/issues/6870) — the perf bar is undefined.** You cannot pass a DoD whose threshold reads `TODO`.
3. **[#6709](https://github.com/marin-community/marin/issues/6709) has no baseline and no comments.** Its own open questions are unanswered: *"What are we targeting — rollout tok/s, end-to-end RL step time, or cost/token?"*

**Concrete breakage (all per the [weekly](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html) unless noted):**

4. **#7117 NCCL `ncclAlltoAll` error.** The 134 GB / 39-shard BF16 HF export passed SHA256 verification, but *"two reruns of the existing inference assertion hit an NCCL ncclAlltoAll error on the JIT path."* PR still open.
5. **`marin-serve --gpu` cannot boot vLLM.** #7111 hit a conflict between the TPU `vllm` extra and `gpu`; #7133 sidesteps it with a throwaway `uvx --from vllm[runai]==0.25.0 --torch-backend cu128` env, but *"It still needs a real H100 boot test."*
6. **#7085 — RPA v3 raises `NotImplementedError: Unsupported tpu_version=4` on v4-8**, so Rohith Kuditipudi *"could not run inference there at all."* Yon posted a fix PR on [2026-07-10](https://discord.com/channels/1354881461060243556/1366632114316906506/1525169761955221525); the weekly digest bot reports it resolved, but the written weekly still describes the fix as pending — treat as in-flight.
7. **The original "wedge" is genuine but unreproduced.** In **#6983**, Will Held saw brokered clients stall >30 min against an idle, healthy engine. The lease/slot-leak theory gave way to httpx pool exhaustion, then Yon's instrumented rerun overturned *that*: the plateau was fast HTTP 400s once the prompt crossed marin-8b-base's true 4096-token limit (an `Out of range float values are not JSON compliant: nan` body masked by `VLLM_ALLOW_LONG_MAX_MODEL_LEN`), with `dropped_responses=0, rejected_requests=0`. Held withdrew the reproduction claim. **The field wedge itself remains unexplained**, and the silent NaN-400 behavior was flagged as possibly deserving its own issue.
8. **Fork fragmentation blocks RL.** Power opened **#7097** to unify to one vLLM fork. Feuer, [2026-07-14](https://discord.com/channels/1354881461060243556/1385733711013871729/1526394430637080576): *"I won't be able to do RL until we sync up on the vLLM branch + fork."*
9. **`--swap-space` hard-errors on the TPU serve path** — a GPU-only flag the tpu-inference `api_server` rejects, presenting as a silent health-check timeout ([#6958](https://github.com/marin-community/marin/issues/6958)). No fix recorded.
10. **[#6754](https://github.com/marin-community/marin/issues/6754)** — an Iris `EndpointService/ListEndpoints` 404 blocked the fully brokered HumanEval smoke; a control-plane mismatch, not a serving regression.
11. **Ingress follow-ups open:** the `quick_serve` BEARER integration and `iap_gclb.py` stages of [#6847](https://github.com/marin-community/marin/issues/6847)/[#6857](https://github.com/marin-community/marin/pull/6857) remain, so some agentic RL still bridges via paid unauthenticated tunnels.

## 5. Caveats on this answer

The frozen corpus has **asymmetric coverage**: its GitHub ingest ends ~2026-07-05 (highest ref #6967), while Discord runs to ~2026-07-16 and weekly summaries to 2026-07-12. Every claim about a 71xx-numbered issue above therefore rests on the **weekly narrative or Discord**, not on primary thread text — I have quoted rather than paraphrased those, but merge SHAs, comment timestamps, and current open/closed status for #7085, #7106, #7111, #7113, #7116, #7117, #7125, #7133 are not independently verifiable here. #6983 and #7052 are likewise past the GitHub edge.

---
<!--provenance-->
> *Data: marinmirror — 68026 chunks, built 48h ago · summaries through 2026-07-06_2026-07-12 (frozen eval corpus, freeze 2026-07-16; no refresh this run).*
>
> *Query: "Who is doing inference work and where is it at? What's the current performance and are there any open issues?"*
>
> *Sub-queries: "July inference epic #6867 — DoD, owner, sub-issues #6868/#6869/#6870, closure of #6041 and #6042" · "GPU vLLM serving of GrugMoE on CoreWeave H100 — EP8 validation, #7116/#7117/#7125 regressions, marin-serve --gpu #7111/#7133" · "TPU vLLM serving stack — #6041/#6664, fork pins #6732/#6733/#7025/#7094, RPA v3 on v4-8 #7085, marin-serve #6556, SWE-bench on Iris v6e #6958" · "Rollout-speed epic #6709 — serving baseline, RL step time, generation throughput, the 'fast enough' bar" · "vLLM serving reliability under load — the brokered wedge #6983, the 4096-token NaN 400s, logit mixing #7113" · "Serving infra and ownership — endpoint ingress #6847/#6857/#6937, federated /proxy mirroring #7108/#7109, who owns what"*
