# Inference at Marin: who, where it stands, performance, and open issues

**Short version.** Inference work is led by **Romain Yon (yonromai)**, who owns the July inference epic and both the TPU and GPU vLLM serving stacks; **Rohith Kuditipudi** drives TPU-side serving for evals; **Russell Power** and **rjpower** own tooling (`marin-serve`, fork consolidation); **Will Held** adds serving-side plumbing; **Benjamin Feuer** is the main downstream consumer (RL rollouts). As of the corpus edge, GPU inference for the 67B-A2B GrugMoE model has just reached a **v0 "landed in main, many rough edges"** state, TPU serving is mid-consolidation, and **there is no measured serving-performance number for GrugMoE on either platform** — the perf goal is literally still a `TODO`.

---

## 1. The epic and who owns it

The organizing issue is **[#6867 "[Epic] July Grug Inference tasks"](https://github.com/marin-community/marin/issues/6867)**, filed by yonromai on 2026-07-02. Its definition of done, verbatim:

> DoD: Support full size GrugMoE model on both TPUs and GPUs. Stretch: Inference is fast enough on GPUs.

It splits into two sub-issues, both filed the same day:

- **[#6869 — "[Inference] vLLM supports fully size GrugMoE checkpoints on GPUs"](https://github.com/marin-community/marin/issues/6869)** — correctness. *"DoD: vLLM is able to correctly do inference on largest GrugMoE checkpoint to date, on GPUs."*
- **[#6870 — "[Inference] vLLM GrugMoE inference performance is decent on GPUs"](https://github.com/marin-community/marin/issues/6870)** — performance. Its bar is **undefined**: *"Fast enough: TODO: Within X% of Jax on GPUs vs. within Y% of reference model (e.g. Qwen?)"*

That `TODO` is load-bearing for the rest of this answer: **the perf sub-goal of the epic has no numeric target, so "decent performance" is currently unfalsifiable.**

A second, forward-looking epic is **[#6709 "Inference speed (for RL rollouts)"](https://github.com/marin-community/marin/issues/6709)**, filed by **ihodes** 2026-06-26 — *"picking up after the July Commitment and Hero Run work... Focus on GPUs, H100s in particular, where we intend do most of our RL this year."* Its stated first task is *"measure and document the current baseline(s) before setting a target"*, with the open question *"What are we targeting — rollout tok/s, end-to-end RL step time, or cost/token?"* Per the [week of 2026-07-06](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html), *"No sub-issues, PRs, or design docs landed against it this week"* — **it is still at the baseline-establishing stage and has not started.**

## 2. Where GPU inference is at

**Status: v0 landed, days old, explicitly rough.** On 2026-07-14 Romain Yon posted in `#inference` ([message](https://discord.com/channels/1354881461060243556/1385733711013871729/1526388201915289642)):

> FYI a v0 of Grug 67b a2b inference on GPUs has landed in main. See example here: ...`tests/vllm/e2e/test_june_67b_a2b_vllm_s3_inference.py`. I'm sure there'll be many rough edges, and I'm still ac[tively working on it]

The engineering path there: the GPU tracking issue **[#6042 "Inference: GrugMoE support in vLLM GPU"](https://github.com/marin-community/marin/issues/6042)** (yonromai, 2026-05-29) set a deliberately modest bar — correct inference on GPU, *"perf ok enough for evals"*, with RL-grade performance explicitly out of scope. Per the [week of 2026-07-06 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html), **#6042 closed** that week *"when Isaac Hodes logged that eight-way expert-parallel (EP8) serving of a real GrugMoE checkpoint had passed on CoreWeave/H100, moving the remaining full-size and reinforcement-learning-grade work onto sub-issues #6869 and #6870."*

The underlying validation is **[PR #6891 "[codex] Validate GrugMoE GPU expert parallel serving"](https://github.com/marin-community/marin/pull/6891)** (yonromai, 2026-07-03). Its [result comment](https://github.com/marin-community/marin/issues/6891#issuecomment-4872341972) reports, **achieved on CoreWeave**:

> Remote pytest: 3 passed in 274.29s · vLLM observed TP=1, DP=8, EP=8; max_num_seqs=16; TRITON_ATTN · Levanter/JAX mesh: data=1, expert=8, model=1

**Read this carefully — it is a sharding-parity correctness check, not a performance or quality result.** The PR itself scopes out *"performance, sampling parity, logprob parity, broad context windows, or long-running stability"*, and the same comment logs a degenerate completion: `" The Ultimate. The Ultimate. The Ultimate"`. The 274s is wall-clock for three tests, **not** a throughput figure. #6891 was then [closed as superseded](https://github.com/marin-community/marin/issues/6891#issuecomment-4887307764) by **[#6966](https://github.com/marin-community/marin/pull/6966)**, restacked on **[#6965 (Fix Grug MoE XSA GQA sharding)](https://github.com/marin-community/marin/pull/6965)** so the sharding fix for [#6964](https://github.com/marin-community/marin/issues/6964) lands in a precursor.

**GPU tooling** is a parallel track. Per the [week of 2026-07-06 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html), **#7111** asked for `marin-serve --gpu` to actually boot vLLM (the TPU `vllm` extra conflicts with `gpu`, and stock PyPI vLLM's torch pin collided with Marin's torch 2.11 / CUDA 13 stack); Weaverbot, at Russell Power's direction, opened **#7133**, which provisions CUDA vLLM per-job in a throwaway `uv`-tool env so vLLM's torch/CUDA tree never enters `uv.lock` — **it still needs a real H100 boot test.** Separately **#7106** (cryptic failure outside a workspace checkout) is fixed by **#7107**. This matters because the in-corpus predecessor state shows the conflict was deliberate: [PR #4663](https://github.com/marin-community/marin/pull/4663) noted *"marin only ships vllm-tpu (no vllm-cuda variant)"* and added the `vllm`/`gpu` mutual exclusion; it went stale unmerged, and [#6481](https://github.com/marin-community/marin/issues/6481) explicitly chose to *"Leave GPU-vs-CPU/TPU/vLLM conflicts in place."*

## 3. Where TPU inference is at

**Status: consolidating forks, with one hard platform gap.** Marin serves TPU vLLM from its own `vllm` + `tpu-inference` forks rather than a package install ([PR #6288](https://github.com/marin-community/marin/pull/6288)), with an agentic refresh skill ([PR #6467](https://github.com/marin-community/marin/pull/6467)) to rebase and smoke-test them on upstream releases. Per the [week of 2026-07-06 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html), **#7025** advanced both fork pins to the landed `marin-community` main SHAs and **#7094** pinned the build so TPU builds skip the default Rust artifacts.

Fork proliferation is the live problem. **Russell Power opened [#7097 "one vllm fork"](https://github.com/marin-community/marin/issues/7097)** to synchronize Rohith Kuditipudi's overlay patches so Grug needs only a single patch set. Romain Yon explained the mechanics in `#infra` on 2026-07-15 ([message](https://discord.com/channels/1354881461060243556/1527030290835570819/1527036805042405488)):

> The is a bit of complexity due to the fact that vllm and tpu-inference repos go in pair - and are therefore rebased together... Pick a tpu-inference release (exactly v0.23.0 currently used in marin's fork)

`marin-serve` itself — **[PR #6556](https://github.com/marin-community/marin/pull/6556)** (rjpower, 2026-06-22), a one-liner inference server + dashboard on Iris — is **TPU-only** at the corpus edge, verified live on Qwen3-0.6B (v5litepod-4) and a Delphi 9.7B SFT canary (v6e-4, auto TP=2, 4k context).

## 4. Current performance

**There is no measured GrugMoE serving-throughput number, on either GPU or TPU, anywhere in this corpus.** #6870's target is a `TODO`; #6709's baseline is unmeasured; #6891 reports only test wall-clock. Saying otherwise would be inventing a number.

What *does* exist are throughput measurements from adjacent stacks. Attribute each carefully:

| Measurement | Hardware / model | Kind |
|---|---|---|
| **623 / 736 gen tok/s** (mean/peak) at client concurrency 8, 32k·bf16 | v6e-4, TP=4 DP=1, `Qwen3.6-35B-A3B-FP8` | Achieved — *datagen* stack |
| **504.2 / 729.6 gen tok/s** (mean/peak), 131k ctx, seqs=8, fp8 KV, concurrency 16 — "131k winner, +115% over R2" | v5p-32, `Qwen3.5-122B-A10B-FP8` | Achieved — *datagen* stack |
| Levanter inference **~10× faster**, "within ~12% of vllm tpu" | **TPU** | Achieved, 2026-06-01 |
| ~**2.5 h/step**, ~10M generated tokens/step, 131k context | 64×H100 (8 nodes), Qwen3 30B-A3B / Qwen3.5 35B-A3B | Achieved — *RL step*, not serving |
| 96–128×H100 needed for Qwen3 Next 80B A3B | H100 | **Projected**, not measured |

Sources and caveats:

- The v6e-4 and v5p-32 tables are from **[#6133 "[datagen] Datagen throughput tracking on TPU"](https://github.com/marin-community/marin/issues/6133)** (AlienKevin and penfever, June 2026). **These are not Marin's serving stack** — they run vLLM-TPU 0.20.0 with `--load-format runai_streamer` inside the `openthoughts-agent:tpu` image, launched from `open-thoughts/OpenThoughts-Agent`. Treating them as `marin-serve` numbers would be a misattribution. Note also the R3-v3 131k config runs at client concurrency **16** (seqs=8), not 8.
- The Levanter figure is dlwh in `#inference`, 2026-06-01 ([message](https://discord.com/channels/1354881461060243556/1385733711013871729/1511034914059976784)): *"i threw goal mode at making levanter inference faster and it made it ~10x faster (mostly some dumb things we were doing) which is within ~12% of vllm tpu."* The comparator is **vLLM on TPU** — do not read it as a GPU result, and it predates all the July GPU work.
- The RL step-time figures are Benjamin Feuer, 2026-07-09 ([message](https://discord.com/channels/1354881461060243556/1524748716190339122/1524749076355088485)), backed by **[#7052](https://github.com/marin-community/marin/issues/7052)**: *"Qwen3 30B A3B and Qwen3.5 35B A3B both fit on 64xH100 (8 nodes). ~2.5hr / step at the moment which is slower than I would like but not insane"*, with the 10M tokens/step self-labeled *"a very ballpark estimate"* and 131k judged *"probably the limit of what we can achieve on this hardware without additional cleverness."* This is RL step time, **not** serving throughput — but it is precisely the cost that epic #6709 exists to attack.

## 5. Open issues

**Blocking someone right now:**

1. **Fork sync blocks RL.** Benjamin Feuer, 2026-07-14 ([thread](https://discord.com/channels/1354881461060243556/1385733711013871729/1526388201915289642)): *"I won't be able to do RL until we sync up on the vLLM branch + fork"* (wanting `mlfoundations/vllm` `penfever/working` merged). Romain agreed to prioritize the port. Filed as **[#7164](https://github.com/marin-community/marin/issues/7164)** — per Feuer the real gap is *"a training GrugMoeForCausalLM PyTorch modeling class."* Worth noting Romain pushed back that this may not be the only blocker (*"Aren't you also blocked on having Grug compatible training code in MarinSkyRL?"* → *"Probably"*).
2. **RPA v3 unusable on TPU v4.** Per the [week of 2026-07-06 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html), **#7085**: ragged paged attention v3 raises `NotImplementedError: Unsupported tpu_version=4` on v4-8 workers, *"so Rohith Kuditipudi could not run inference there at all."* The minimal fix is adding a `case 4` to the default heuristic, since the v3 tuned-block-sizes table already carries a conservative v4 fallback.
3. **[#5672 "[TPU] Fix Delphi vLLM RPA VMEM regression"](https://github.com/marin-community/marin/issues/5672)** (RohithKuditipudi, 2026-05-12) — **still open after two months.** Live on 2026-07-15 ([thread](https://discord.com/channels/1354881461060243556/1503803923876675635/1527022402909765692)): rohithck *"my monkey patching has worked so far... but today I hit the issue again with the 1e22 model on a v5-8"*; romain: *"this thread had totally gotten LRU cache evicted from my brain."* Neither could establish whether it ever worked unpatched.

**Open engineering items:**

4. **#7117** (per the [summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)) — still open; adds a vLLM-compatible BF16 Hugging Face export regression with a whole-tree SHA256 over 39 shards. The full **134 GB export passed**, but a rerun of the inference assertion hit an **NCCL AlltoAll error** on the JIT path. (Its sibling **#7116**, an 8×H100 BF16 inference regression on the step-18000 checkpoint, merged; **#7125** fixed the CI redness it exposed.) Note the summary is internally inconsistent on whether one or two reruns failed.
5. **#7133 needs a real H100 boot test** before CUDA `marin-serve` can be trusted.
6. **[#6983](https://github.com/marin-community/marin/issues/6983) — brokered vLLM wedge: still unexplained.** This one is easy to get wrong. Will Held filed it after clients timed out against an idle, healthy engine; his isolated reproducer was then **overturned** by Romain Yon's instrumented rerun, which showed the reproducer's plateau was really fast HTTP 400s once the prompt crossed marin-8b-base's true 4096-token position limit (a NaN body masked by `VLLM_ALLOW_LONG_MAX_MODEL_LEN`). Will Held withdrew the reproduction claim — **but the original field wedge "remains genuine but unreproduced."** The 4096-token NaN bug explains the *reproducer*, not the *bug*. The silent NaN-400 behavior past max position was flagged as possibly deserving its own issue.
7. **[#7183](https://github.com/marin-community/marin/issues/7183)** — Levanter TPU vs GPU logprob differences, raised by Romain 2026-07-15; Russell Power responded that a golden run with longer outputs should determine *"whether this is expected or something we should worry about."*
8. **#7113** (Will Held) — brokered logit mixing: a `RunningModel` serving a logit-mixed subset of the OpenAI completions API over two brokered vLLM systems (teacher and student), per the [summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html).

**Meta-issue.** Two of the three things one would want to state about performance are undefined rather than unmet: **#6870 has no numeric bar and #6709 has no measured baseline.** Until one of those is filled in, "how is inference performing?" has no answerable form for GrugMoE specifically.

## Coverage gap — please read

This corpus is **asymmetric**. Its raw GitHub leg stops at **~#6967 / 2026-07-05**; Discord runs to **2026-07-16** and weekly summaries to **2026-07-12**. Every item numbered **#7025 and above** — #7025, #7052, #7085, #7094, #7097, #7106/#7107, #7111, #7113, #7116, #7117, #7133, #7164, #7183 — is sourced **only** from the weekly summary or Discord, never from the issue text itself. I have flagged those inline. The close of #6042 is likewise summary-only, and carries an attribution wrinkle: the EP8 validation was run and posted by **yonromai** on #6891, while the summary credits **Isaac Hodes** with the closing log. Anything claiming to quote the bodies of those issues directly should be treated as unverified.

---
<!--provenance-->
> *Data: marinmirror (frozen eval corpus) — 86367 chunks, built 0h ago · summaries through 2026-07-06_2026-07-12 · GitHub leg ends ~#6967 / 2026-07-05, Discord to 2026-07-16. No refresh (disabled for this run).*
>
> *Query: "Who is doing inference work and where is it at? What's the current performance and are there any open issues?"*
>
> *Sub-queries: "July inference epic #6867 — scope, owner, sub-issues #6869/#6870" · "TPU vLLM serving stack: fork pins #7025/#7094/#7097, ragged paged attention v3 #7085" · "marin-serve GPU/CUDA vLLM path: #7111, #7133, #7106/#7107, GrugMoE EP8 on CoreWeave H100 (#6042)" · "inference/serving performance numbers: tokens/sec, throughput, decode latency, #6870" · "inference regression harnesses #7116/#7117/#7125 and the NCCL ncclAlltoAll error" · "rollout-speed epic #6709, brokered vLLM degradation #6983, brokered logit mixing #7113" · adversarial verification pass over 14 load-bearing claims.*
