# Marin inference: who, where it stands, performance, and open issues

**Short version.** Inference is essentially one person's workstream — **Romain Yon** owns the vLLM / tpu-inference fork stack and the July inference epic — with Russell Power on the `marin-serve` platform layer, David Hall on the competing Levanter-native path, and Rohith Kuditipudi and Benjamin Feuer as the heaviest downstream consumers. The state as of the corpus edge (2026-07-15) is **correctness-first and perf-unmeasured**: a v0 of 67B-A2B GPU inference landed in `main` on July 14, but the July epic's three sub-issues are all still open, and — the single most important finding — **there is no measured GPU serving throughput or latency number anywhere in the corpus**, even though establishing that baseline is the stated first task of the rollout-speed epic.

---

## 1. Who is doing the work

| Person | Area | Evidence |
|---|---|---|
| **Romain Yon** (`yonromai`) | De-facto owner: vLLM + tpu-inference forks, GrugMoE serving path, brokered serving, the July epic | Authored the epic [#6867](https://github.com/marin-community/marin/issues/6867) and all three children [#6868](https://github.com/marin-community/marin/issues/6868) / [#6869](https://github.com/marin-community/marin/issues/6869) / [#6870](https://github.com/marin-community/marin/issues/6870); TPU serving [PR #6664](https://github.com/marin-community/marin/pull/6664); fork refresh [#6733](https://github.com/marin-community/marin/issues/6733); brokered vLLM [PR #5887](https://github.com/marin-community/marin/pull/5887) |
| **Russell Power** (`rjpower`) | `marin-serve` (dev-facing one-liner server + dashboard on Iris); reviewer/architect | Authored [PR #6556](https://github.com/marin-community/marin/pull/6556) closing [#6545](https://github.com/marin-community/marin/issues/6545#issuecomment-4770411340); strategy call "Yeah we want vLLM eventually" on [#6041](https://github.com/marin-community/marin/issues/6041#issuecomment-4792823877) |
| **David Hall** (`dlwh`) | Levanter/JAX native inference — the alternative to vLLM | [#6229 parity benchmarks](https://github.com/marin-community/marin/issues/6229), [#6185 multi-prefill admission](https://github.com/marin-community/marin/issues/6185#issuecomment-4636119006), [#6736 GPU latency numbers](https://github.com/marin-community/marin/issues/6736) |
| **Rohith Kuditipudi** (`rohithck`) | Heaviest TPU-vLLM *consumer* (Delphi / downstream-scaling evals); surfaces the TPU blockers | Owns [#5672 RPA VMEM regression](https://github.com/marin-community/marin/issues/5672); reported the RPA v3 v4-gap (#7085) |
| **Benjamin Feuer** (`penfever`) | RL-side consumer; driving fork consolidation | Maintains `mlfoundations/vllm` `penfever/working`; filed [#7164](https://github.com/marin-community/marin/issues/7164); authored the role spec [#6500](https://github.com/marin-community/marin/issues/6500) |
| **Will Held** (`willheld`) | Brokered-serving reliability + logit mixing — but largely *moved off* inference | Filed [#6983](https://github.com/marin-community/marin/issues/6983); [PR #7113 brokered logit mixing](https://github.com/marin-community/marin/issues/7113). His April RFC [#4401](https://github.com/marin-community/marin/issues/4401) was auto-closed for inactivity on 2026-07-04; by July he is on tokenizers and training MFU |
| **Isaac Hodes** (`ihodes`) | Roadmap/program owner, not an implementer | Filed [#6709](https://github.com/marin-community/marin/issues/6709) and [#6051](https://github.com/marin-community/marin/issues/6051); no implementation PRs in the inference stack |

> **Correction to the weekly narrative.** The 2026-07-06→07-12 summary states that "#6042 closed when Isaac Hodes logged that eight-way expert-parallel (EP8) serving of a real GrugMoE checkpoint had passed on CoreWeave/H100." Against primary sources this is a **mis-attribution**: the EP8 CoreWeave pass was logged by **Romain Yon**, on [#6891](https://github.com/marin-community/marin/issues/6891#issuecomment-4872341972), not by Isaac Hodes. The only `ihodes` comment on [#6042](https://github.com/marin-community/marin/issues/6042) is a 2026-05-29 scheduling note ("Needed no earlier than when #6044 lands"). The *closure* of #6042 is unsupported by primary data in this corpus.

---

## 2. Where it's at

### The July epic: [#6867](https://github.com/marin-community/marin/issues/6867) — 0 of 3 sub-issues closed

The epic's definition of done is one line: *"Support full size GrugMoE model on both TPUs and GPUs. Stretch: Inference is fast enough on GPUs."* As of the latest weekly it stands at **0/3 sub-issues closed** — [#6868](https://github.com/marin-community/marin/issues/6868) (TPU), [#6869](https://github.com/marin-community/marin/issues/6869) (GPU), [#6870](https://github.com/marin-community/marin/issues/6870) (GPU perf).

**GPUs — furthest along.** On 2026-07-14 Romain Yon announced in `#inference`: *"FYI a v0 of Grug 67b a2b inference on GPUs has landed in main… I'm sure there'll be many rough edges, and I'm still actively working on it"* ([Discord](https://discord.com/channels/1354881461060243556/1385733711013871729/1526388201915289642)), pointing at `tests/vllm/e2e/test_june_67b_a2b_vllm_s3_inference.py`. This is the first evidence the **full-size** GPU path exists — but it is self-labelled v0, and no metrics accompany it.

The prior hard GPU evidence is narrower than the epic's DoD. The EP8 validation on [#6891](https://github.com/marin-community/marin/issues/6891#issuecomment-4872341972) passed on **8×H100 / CoreWeave** with *"vLLM observed TP=1, DP=8, EP=8"* and *"vLLM/Levanter batch match: true"* — but it ran against `s3://marin-us-east-02a/marin/grug/moe_may_compute_opt_d512_ep1-05c39b/checkpoints/step-10980`, a **d512 compute-optimal MoE, not the 67B-A2B**. Its own caveat is explicit: it validates *"short greedy parity and routed expert coverage for one real checkpoint and one prompt batch, not performance, sampling parity, logprob parity, broad context windows, or long-running stability."* That PR was closed 2026-07-05 as superseded by [#6966](https://github.com/marin-community/marin/pull/6966), which restacks on the XSA/GQA sharding fix [#6965](https://github.com/marin-community/marin/pull/6965).

**TPUs — behind, and blocked.** What is proven is a *small* GrugMoE checkpoint, one prompt, short decode, a single `v6e-4` slice ([PR #6664](https://github.com/marin-community/marin/pull/6664), which closed [#6041](https://github.com/marin-community/marin/issues/6041#issuecomment-4869501446)). Between that and #6868's DoD sit two live blockers (§4). The week's TPU work was fork-pin consolidation — #7025 advancing pins to landed `marin-community` main SHAs, #7094 skipping Rust artifacts, and Russell Power's [#7097](https://github.com/marin-community/marin/issues/7097) to get Grug down to a single set of patches.

**Regression harnesses.** #7116 (merged) vendored the June 67B cooldown launcher plus an 8×H100 BF16 inference test asserting the token after "The United States Of" is " America" on all eight devices, with a top-25 logprob golden check; #7125 fixed the CI redness it exposed. #7117 (open) adds the export side — the **134 GB vLLM-compatible BF16 HF export passed** its whole-tree SHA256 over 39 shards, though two reruns of the inference assertion hit an `ncclAlltoAll` error on the JIT path (never root-caused in-corpus).

### The rollout-speed epic: [#6709](https://github.com/marin-community/marin/issues/6709) — still at the baseline stage

Opened by Isaac Hodes on 2026-06-26 to *"speed up rollout / generation inference so RL is not inference-bound"*, focused on H100s. Its body says: *"First establish the baseline(s), then optimize against an agreed target"*, and the open questions include *"First task: measure and document the current baseline(s) before setting a target."* **It has zero comments and no sub-issues, PRs, or design docs landed against it.** The target itself is still an open question in the issue: *"What are we targeting — rollout tok/s, end-to-end RL step time, or cost/token?"*

---

## 3. Current performance

**The headline is a gap, not a number.** There is **no measured GPU/H100 inference or serving throughput or latency figure anywhere in this corpus** — verified by an adversarial sweep across four query phrasings plus direct reads of #6042, #6709, #6867–#6870, #6891, #6966, #6736 and #6218. This is exactly what the tracking issues predict: #6709's baseline was named as the first task and never delivered, and **#6870's threshold is still literally unfilled**: *"Fast enough: TODO: Within X% of Jax on GPUs vs. within Y% of reference model (e.g. Qwen?)"*.

Beware two adjacent traps. [#6736 "Experiment: GPU latency numbers"](https://github.com/marin-community/marin/issues/6736#issuecomment-4825320696) has real 8×H100 measurements (NCCL all-reduce ~20–25 µs, 4096³ BF16 matmul ~172 µs) but these are **hardware microbenchmarks**, and the audit explicitly involved *"removing unsupported model-inference… rows"*. And the abundant H100 numbers around Grug (23.8% MFU, B200 17.8% MFU) are **training**, not inference.

What *is* measured — all **TPU**, all *achieved* unless noted:

| Result | Hardware / config | Number |
|---|---|---|
| vLLM vs Levanter decode parity ([#6229](https://github.com/marin-community/marin/issues/6229#issuecomment-4641430082)) | Qwen3-8B, **v6e-8**, TP=8 | vllm-tpu **1264.03** decode tok/s vs levanter:auto **957.72** → ratio **0.758, target FAIL** |
| vLLM fork migration A/B ([#4357](https://github.com/marin-community/marin/issues/4357#issuecomment-4201080002)) | Llama-3.1-8B, **v6e-4**, TP=4, GRPO rollouts | 3,772.88 → **5,194.00** output tok/s (**+37.7%**) |
| vLLM isolated ceiling ([#1827](https://github.com/marin-community/marin/issues/1827#issuecomment-3465171897)) | Llama-3.2-1B, **v4-8**, TP=8 | ~**5,300** tok/s; drops to ~3,100 in the RL pipeline once prompt preprocessing matches ([#1823](https://github.com/marin-community/marin/issues/1823#issuecomment-3484188060)) |
| Levanter multi-host vs vLLM ([#3826](https://github.com/marin-community/marin/issues/3826)) | Llama-8B, **v5p-8** | Levanter 762 tok/s vs vLLM 7,940 tok/s (**6–10× slower**) — closed as a decision |

The #6229 row is the load-bearing one and it is the **latest**: no later chunk in the corpus supersedes it. Note it directly contradicts David Hall's casual June claim that a Levanter speedup landed *"within ~12% of vllm tpu"* ([Discord, 2026-06-01](https://discord.com/channels/1354881461060243556/1385733711013871729/1511034914059976784)) — the instrumented benchmark a week later measures 18–24% behind. Prefer the benchmark.

**The one end-to-end RL cost figure**, from Benjamin Feuer in the 131k-MoE-RL thread ([Discord, 2026-07-09](https://discord.com/channels/1354881461060243556/1524748716190339122/1524749076355088485)): *"Qwen3 30B A3B and Qwen3.5 35B A3B both fit on 64xH100 (8 nodes). ~2.5hr / step at the moment which is slower than I would like but not insane."* Asked for tokens per step he answered *"A \*very\* ballpark estimate would be ~10M tokens generated per step"* — **treat 10M as an estimate, not a measurement**. He judges 131k *"probably the limit of what we can achieve on this hardware without additional cleverness."* Qwen3 Next 80B A3B — the closest public analog to the incoming Marin 67B-A2B — is **projected** to need 96–128×H100. That 2.5 h/step generation cost is precisely the lever #6709 exists to pull.

---

## 4. Open issues

**Blocking RL on the new model** (two things, both open at the corpus edge):
1. **vLLM fork sync.** Feuer, 2026-07-14: *"I won't be able to do RL until we sync up on the vLLM branch + fork"* — he needs `mlfoundations/vllm` `penfever/working` merged into the Marin fork. Romain: *"I'll prioritize porting the changes in your fork, that needs to happen anyway."* No message in the corpus confirms the port completed. ([Discord](https://discord.com/channels/1354881461060243556/1385733711013871729/1526393502387277884))
2. **[#7164](https://github.com/marin-community/marin/issues/7164)** — MarinSkyRL needs a PyTorch `GrugMoeForCausalLM` training class to load Grug via `from_pretrained`. Filed 2026-07-14.

**TPU serving blockers:**
- **#7085 — RPA v3 raises `NotImplementedError: Unsupported tpu_version=4`** on v4-8 workers, so Rohith Kuditipudi *cannot run inference on v4 at all*. This matters because the 67B-A2B hero run trains on **v4-2048** ([#6704](https://github.com/marin-community/marin/issues/6704)) while all validated GrugMoE TPU serving to date is v6e-4. He notes the minimal fix is adding a `case 4` to the default heuristic.
- **[#5672 — Delphi vLLM RPA VMEM regression](https://github.com/marin-community/marin/issues/5672)**, open since May and still live: *"today I hit the issue again with the 1e22 model on a v5-8"* and *"I don't think it (1e22) was ever working for me without monkey patching"* (rohithck, [2026-07-15](https://discord.com/channels/1354881461060243556/1503803923876675635/1527021967851393025)). Romain re-engaged: *"this thread had totally gotten LRU cache evicted from my brain."* No fix yet.
- **[#7097 "one vllm fork"](https://github.com/marin-community/marin/issues/7097)** — open; the vllm and tpu-inference repos *"go in pair - and are therefore rebased together"*, which is what makes consolidation awkward. Collateral damage from the pin churn: uv git-lock timeouts spinning up parallel vLLM workers, plus a missing `zephyr` module ([Discord, 2026-07-10](https://discord.com/channels/1354881461060243556/1364827114670657616/1525035072414023750)).

**Serving reliability:**
- **[#6983](https://github.com/marin-community/marin/issues/6983) — brokered vLLM degrades under sustained load.** A good story about diagnosis discipline: Will Held's initial lease-leak theory, then an httpx connection-pool theory with a reproducer that *"wedged reliably in about 24 minutes"* — both **overturned** by Romain Yon's instrumented rerun, which showed the plateau was fast vLLM **HTTP 400s** once the prompt crossed marin-8b-base's true **4096-token position limit** (body: `Out of range float values are not JSON compliant: nan`), masked by `VLLM_ALLOW_LONG_MAX_MODEL_LEN` and an oversized `max_model_len`; the broker path was clean. Held withdrew the repro claim. **Net: the original field wedge remains genuine but unreproduced**, and the silent NaN→400 was flagged as possibly deserving its own issue — which I find no evidence was ever filed. The same NaN bit Rohith independently on humaneval on [2026-07-14](https://discord.com/channels/1354881461060243556/1356487738840318002/1526513864697450526).

**`marin-serve` gaps:** [#7111](https://github.com/marin-community/marin/issues/7111) (`--gpu` can't boot vLLM; TPU extra conflicts with `gpu`) → [#7133](https://github.com/marin-community/marin/issues/7133) sidesteps resolution with a throwaway `uvx --from vllm[runai]==0.25.0 --torch-backend cu128` env, **still needs a real H100 boot test**, and found #7111's `torch==2.7.0` premise stale. [#7106](https://github.com/marin-community/marin/issues/7106)/[#7107](https://github.com/marin-community/marin/issues/7107): cryptic failure outside a workspace checkout.

**Correctness / in review:** [#7117](https://github.com/marin-community/marin/issues/7117) (NCCL AlltoAll on rerun), [#7137](https://github.com/marin-community/marin/issues/7137) (June export parity from object storage), [#7113](https://github.com/marin-community/marin/issues/7113) (logit mixing, reviewed by rohithck 2026-07-13), [#6965](https://github.com/marin-community/marin/pull/6965)/[#6964](https://github.com/marin-community/marin/issues/6964) (XSA/GQA sharding), and [#7183](https://github.com/marin-community/marin/issues/7183) — Levanter TPU-vs-GPU logprob differences, raised 2026-07-15, with Russell Power hoping *"the golden run with some longer outputs will be helpful in determining whether this is expected."* Nobody calls #7183 an RL blocker.

**Unresolved architecturally:** vLLM vs Levanter-native inference is still an open question in-corpus — Power's *"we want vLLM eventually"* ([#6041](https://github.com/marin-community/marin/issues/6041#issuecomment-4792823877)) against Hall's Levanter speedup work, with #6229 currently favoring vLLM by 24%.

---

## Caveats on this answer

The corpus's **GitHub coverage ends around #6967 / 2026-07-05**, while **Discord runs to 2026-07-16**. Everything numbered above ~#7000 is therefore sourced from the 2026-07-06→07-12 weekly narrative plus post-summary Discord, not from the issue threads themselves — so I cannot report their open/closed state as of today, only as of that window. Two items I deliberately do not report: the contents of rohithck's review on #7113, and the root cause of the `ncclAlltoAll` error (observed twice, never diagnosed in-corpus).

---
<!--provenance-->
> *Data: marinmirror — 68026 chunks, built 45h ago · summaries through 2026-07-06_2026-07-12 (frozen eval corpus; no refresh this run).*
>
> *Query: "Who is doing inference work and where is it at? What's the current performance and are there any open issues?"*
>
> *Sub-queries: "TPU vLLM serving of full-size GrugMoE checkpoints — #6868, fork pins #7025/#7094, one-vllm-fork #7097, RPA v3 on TPU v4 #7085" · "GPU vLLM serving of GrugMoE on H100 — EP8 validation #6042/#6966, #6869, XSA/GQA sharding #6964/#6965" · "inference / rollout performance numbers — throughput, decode latency, RL step time, #6870, epic #6709, MoE sharding grid #7052" · "who owns Marin inference work — Yon, Kuditipudi, Held, Power, Hodes, Hall, Feuer across vLLM / tpu-inference / marin-serve / brokered serving" · "open inference bugs and serving reliability — #6983, marin-serve --gpu #7111/#7133, #7106/#7107, logit mixing #7113, export regressions #7116/#7117/#7137" · "latest state mid-July 2026 — 67B-A2B GPU inference in main, penfever/working fork port, #7164, #7183, what blocks RL" · adversarial verification pass on 9 load-bearing claims.*
