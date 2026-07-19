# Our current Muon approach, and how we got here

**Short version.** Marin trains on **MuonH** — Nesterov-momentum Muon whose orthogonalized direction is mapped through a scale-invariant *Frobenius-hyperball* reparam instead of Muon's usual `√(out/in)` scaling. It is the production matrix optimizer on TPU, landed in the MoE template as `grug_moe_muonh_v1` ([PR #6153](https://github.com/marin-community/marin/pull/6153)), and it is what the 67B-A2B hero run is training with on v4-2048 today ([#6044](https://github.com/marin-community/marin/issues/6044)). Plain Muon lost; MuonH-with-an-Adam-mask won. On GPU, Muon was **shelved in late June** for cost reasons ([#6693](https://github.com/marin-community/marin/issues/6693)) — but that is the single most stale part of this picture, and July evidence has already moved against it (see §5).

---

## 1. What "our Muon approach" concretely is

### MuonH, not Muon

The distinguishing piece is the **hyperball step**, not the orthogonalization. From [PR #6634](https://github.com/marin-community/marin/pull/6634): the orthogonal direction `X_t` is mapped through the "**scale-invariant (hyperball) constant-norm reparam** instead of the `√(Out/In)` scaling — each matrix's Frobenius norm is held at its initialization value." The same PR pins the definition by construction: at `λ = 0` its curvature-corrected optimizer "is **exactly MuonH** (Nesterov Muon + hyperball)."

Per-leaf, the step is `lr · ||W||_F · update` ([#5794](https://github.com/marin-community/marin/issues/5794)). Two consequences the team has had to reason about explicitly: hyperball "washes out the `sqrt(fan_out/fan_in)` in Muon" ([Discord #moe, 2026-05-14](https://discord.com/channels/1354881461060243556/1365044508546568372/1504541888626163712)), and the split `w_gate`/`w_up` MoEMLP layout is *not* optimizer-neutral, because "MuonH sees two `(e, d, i)` leaves and uses each leaf's own `||·||_F` independently" ([#5794](https://github.com/marin-community/marin/issues/5794)).

### Three parameter groups, not two

This is the detail most often gotten wrong. Per the [#6044 launch job summary (2026-06-27)](https://github.com/marin-community/marin/issues/6044#issuecomment-4820179240), routing is by parameter path via `experiments/grug/moe/optimizer.py:create_mask`:

| group | parameters | update |
|---|---|---|
| **muonh** | "2D weight matrices and stacked MoE expert weights" | Newton-Schulz + hyperball |
| **adamh** | "`lm_head` / `output_proj`" | "Adam moments + Frobenius hyperball (`scale_by_adamh`); **no Newton-Schulz**." Same LR schedule as muonh |
| **adam** | "biases, RMSNorm scales, router weights, gated-norm scales, and (via `rmsnorm_to_adam=True`) stacked RMSNorm scales" | plain Adam |

**`lm_head` has never been on Muon**, and keeping it off was the fix that made the swap work at all (§3). Note one internal disagreement: the earlier [2026-06-02 plan comment](https://github.com/marin-community/marin/issues/6044#issuecomment-4607416665) put GatedNorms in the *muonh* group; the June-27 launched config lists gated-norm scales under **adam**. Prefer the launched version.

### Hyperparameters as shipped

Learning rate comes from the refit heuristic in [#5951](https://github.com/marin-community/marin/issues/5951), wired into `heuristic_v2.py`. [#6153](https://github.com/marin-community/marin/pull/6153) states the shipped rule: `muonh_lr = 18.31 · tokens^-0.395 · dim^-0.150 · √B`, with `adam_lr = muonh_lr / (13/3)` and `adamh_lr = muonh_lr` ([#6044 job summary](https://github.com/marin-community/marin/issues/6044#issuecomment-4820179240)). Reported fit quality is **R² = 0.996 on 17 cells**, from ~150 runs over d ∈ {512, 768, 1024, 1280} on **v5p-32 / v5p — small-scale, not the hero hardware**.

*Caveat worth carrying:* the #5951 issue body's own fit is `26.9 · tokens^-0.395 · dim^-0.121 · √B`, while #6153 and #6044 both cite the dim exponent as **−0.150** with coefficient 18.31. The #5951 comment stream is not in this corpus, so the −0.121 → −0.150 correction is visible only via the downstream citations. Treat −0.150 / 18.31 as shipped.

`β₂` is a **clamp floor, not a pinned constant**: `clip(0.999^(tokens_per_batch/131072), 0.95, 0.9999)`, which at the hero run's batch sizes lands at 0.95 both before and after the ramp ([#6044](https://github.com/marin-community/marin/issues/6044#issuecomment-4820179240)). Larry's rationale: "I swept beta2 [in #4567] previously on AdamH. Performance was not that sensitive. We floor beta2 at 0.95, so all big runs end up using 0.95" ([Discord #optimizers, 2026-06-28](https://discord.com/channels/1354881461060243556/1382240679765217342/1520675022354513991)). `β₁ = 0.9062` fixed; `weight_decay = 0`; `max_grad_norm = None`.

The **13/3 MuonH:Adam LR ratio was inherited from AdamH and never re-tuned for MuonH** — [#5621](https://github.com/marin-community/marin/issues/5621) flagged exactly this ("that ratio was tuned for AdamH's geometry in #4225 and was never re-tuned for MuonH") and its Phase 3 was launched at 13/3 anyway rather than waiting for the ratio decision. No close-out moving it exists in the corpus; the hero run shipped 13/3. **This is an open loose end, not a settled choice.**

> **Gap:** I could not tie a **Muon momentum value** or a Newton-Schulz step count for the production MoE config to any quoted source. The only 0.95-momentum evidence is `muonh_qwen3_512_momentum_0_95` wandb runs on a **d512 dense qwen3**, not the MoE recipe. Omitted rather than guessed.

---

## 2. Where it's running: the 67B-A2B hero run

The production run is `moe_67b_a2b_d2560_..._v4_2048_muon_10T`, launched 2026-06-27 on **v4-2048 (1,024 chips), us-central2** ([#6044](https://github.com/marin-community/marin/issues/6044#issuecomment-4820179240)): d=2560, 26 layers, 256 experts at k=4, **67.1B total / 2.01B active**, 157,500 steps, ~10.07T tokens, ~1.69e23 FLOPs, EP=1. `muonh_lr` 0.003733 → 0.005279 (×√2) at step 15,000 alongside the 33.55M → 67.1M tokens/step batch ramp.

Hardware choice was **absolute throughput over per-chip MFU** ([#6044](https://github.com/marin-community/marin/issues/6044#issuecomment-4812848295)): v4-1024 held the best per-chip MFU at **21.54%** (1.49M tok/s) but projected **~78 days**; v4-2048 with replica sharding and doubled batch reached **18.62% MFU / ~2.57M tok/s → ~45 days**. Larry's *initial* read was the opposite ("This seems like a bad trade"); dlwh's replica-sharding suggestion flipped it. Report the later state.

**Latest state (wandb, ~2026-07-17):** the resumed run is **still training on Muon** — `..._muon_resume15k_v2_10T`, state=running, ~3.17T of 10.07T tokens (~32%), train loss 1.39. The first intermediate cooldown off the step-39k checkpoint **finished** (`..._muon_cooldown_step39k`, 2.829e12 tokens, eval bpb 0.6665); willheld confirmed 2026-07-12: "Final loss - first cooldown is done!"

> **Target vs achieved:** the preregistered stage-1 goal is **2.269 Paloma macro loss at the 8T mark, evaluated at seqlen 8192 on 1024 sequences** ([#6044](https://github.com/marin-community/marin/issues/6044#issuecomment-4820008980)) — a *target*, from a weak 3-anchor fit that Larry himself flagged (vs 20+ anchors for the prior 1e23 run). The run is only ~32% in. **Nothing in the corpus states the run has hit or missed it**, and wandb `eval/loss` is not the same metric.

### The bug that nearly ate the run

The most operationally consequential Muon event: the **sharded Newton-Schulz was silently dropping**. Larry, [2026-06-27](https://github.com/marin-community/marin/issues/6044#issuecomment-4819379417):

> "the sharded NS calc in Muon was silently dropping when num_matrices didn't evenly divide into num_chips. This was originally written on v4-1024 where 256*26 is divisible by 512 chips. However it isn't divisible by 1024 chips."

(6,656 matrices = 256 experts × 26 layers; 6656/512 = 13 exactly, 6656/1024 = 6.5.) For a day this was **misdiagnosed as a large-batch instability** — fp32 masked the divergence without fixing it. Larry: "Lack of orthogonalization explains why the loss curve started better and then got worse over time, similar to how adam can beat muon in first couple steps." It still shaped the production config: the run starts at 33.55M tokens/step for the first 5% out of caution, even though "we likely could do the full run at 67 million token batch size."

Two honest qualifications: the root-cause writeup posted at **16:54 UTC, ~1.5h *after* the production job launched at ~15:42** — the fix was in the launched code, but this was a same-afternoon catch during bring-up, not a pre-launch gate. And the identical silent-fallback pattern had been **documented on the GPU side nine days earlier** ([#6493, 2026-06-18](https://github.com/marin-community/marin/issues/6493#issuecomment-4738902480): a param whose leading stack dim isn't divisible "silently falls back to plain `vmap`"). It was in the corpus before it bit TPU. Larry's own postmortem asks for the PR-review and prod-hardware-replication process that was skipped.

---

## 3. How we got here — the adoption arc

**2025-09 → 2025-11 · The dense precondition.** [PR #1558](https://github.com/marin-community/marin/pull/1558) tested whether scaling batch size widens the Muon-vs-AdamW gap. Confirmed in [#1565](https://github.com/marin-community/marin/issues/1565#issuecomment-3567362589): "while both optimizers degrade with increasing batch size, **Muon degrades significantly more slowly than AdamW**." This is the origin of the institutional belief that Muon is the right bet *at large batch* — which is exactly the regime the hero run lives in.

**2026-03 · Muon had never touched MoE.** [#4034](https://github.com/marin-community/marin/issues/4034) opened with "Muon + MoE has not been tried in this codebase" — all MoE work was AdamW; `GrugMuonConfig` was proven only on grug *dense*.

**2026-04 · Plain Muon lost.** [#5167](https://github.com/marin-community/marin/issues/5167) ran a Vizier LR/beta search for Muon plus an AdamH-2×-batch control. Verdict (v5p-8, Paloma macro): d512 baseline 3.81035 → best Muon 3.80732 (−0.003), but **d768 3.43387 → 3.46759 (+0.034, worse)**. "Current gate-1 evidence does not support promoting Muon over AdamH broadly; AdamH remains the stronger scaling choice at d768." The AdamH-2× controls also *underperformed*, killing the "it's just an effective-batch effect" explanation.

**2026-05-07/09 · The OOD diagnosis.** Larry on Discord: "MuonH gives a 30% speedup over AdamH in nanogpt… but looks roughly flat in our MoE recipe" ([#moe](https://discord.com/channels/1354881461060243556/1365044508546568372/1501750602731094056)). Walking nanoGPT feature-by-feature to the Marin recipe: "Muon holds the gain all the way to our recipe. However **Muon is doing terrible on OOD evals like code**… The worst is github/cpp, where Muon gets 2.55 and AdamH gets 2.2."

**2026-05-09 · The run that worked — [#5596](https://github.com/marin-community/marin/issues/5596) / [PR #5597](https://github.com/marin-community/marin/pull/5597).** MuonH on matrix params, **`lm_head`/`output_proj` kept on AdamH**, and — the actual fix — the baseline Adam group left on Adam with identical hyperparameters (suffix `baseline-adam-mask`). Kaiyue Wen: "We only need to swap every AdamH with MuonH except lm head… This seems to speedup uniformly on every dataset **including cpp**." Gate results vs the v16 AdamH baseline (v5p-8, Paloma macro):

| scale | budget | baseline → MuonH | wall-clock speedup |
|---|---|---|---|
| d512 | 2.19e17 | 3.8104 → 3.7542 | **1.33×** |
| d768 | 1.70e18 | 3.4339 → 3.3988 | **1.26×** |
| d1024 | 9.00e18 | 3.1605 → 3.1357 | **1.22×** |
| d1280 | 2.83e19 | 3.0065 → 2.9888 | **1.19×** |

**This is the honest crux of the whole story: gate-2 did not cleanly pass.** MuonH's refit exponent α = 0.0906 vs baseline 0.0941, so the curves **cross at C ≈ 1.65e21** and the **1e23 projection is +0.0106 *worse***. The thread's own words: "in the measured 2.19e17–2.83e19 range MuonH is uniformly better in both quality and wall-clock; the projection caveat at 1e23 is real under this fit but rests on 4 points." **Marin adopted MuonH on measured-range evidence while its extrapolation to hero-run scale was unfavorable.** That is a judgment call, and it is worth naming as one — the hero run at ~1.69e23 FLOPs sits on the wrong side of that crossover.

**2026-05-14/17 · GatedNorms and the canonical baseline.** [#5750](https://github.com/marin-community/marin/issues/5750)/[#5726](https://github.com/marin-community/marin/issues/5726#issuecomment-4455160403) routed GatedNorms to MuonH on equal-TPS speedups (1.362/1.340/1.190× at d512/768/1024). [#5763](https://github.com/marin-community/marin/issues/5763#issuecomment-4471778083) became the canonical baseline — also an explicit judgment call, "a borderline pass — d768 is 0.991… **Adopted as the canonical baseline anyway**."

**2026-05-26 → 06-04 · LR refit and the merge.** [#5951](https://github.com/marin-community/marin/issues/5951) refit the LR heuristic (with a self-corrected `intermediate_dim` units bug; net effect at compute-optimal was **0.88–0.98× the old LR, not a bump**). [#6074](https://github.com/marin-community/marin/issues/6074#issuecomment-4587463772) fit the May Recipe scaling law `loss(C) = 1.6 + 88.90·C^-0.0941` — same exponent as the v16 baseline, "**uniform 2.12× equal-TPS speedup at every budget**." Then [PR #6153](https://github.com/marin-community/marin/pull/6153) merged the May Recipe onto main's MoE template: "MuonH optimizer (`GrugMoeMuonHConfig`, registered as `grug_moe_muonh_v1`) **replaces AdamH on the weight-matrix + GatedNorm group**."

> **Scope the 2.12× carefully:** it is the **entire May Recipe** (architecture + optimizer) vs v16, fit on four small **v4-32** points, and it is *equal-TPS compute-equivalent*, not a hero-run measurement. It is not "Muon gives 2.12×."
>
> A commonly-repeated **"~4.2× effective speedup, best frontier point is a MuonH run at d1280" could not be corroborated** anywhere in this corpus despite targeted searching. The largest MuonH-attributable figures I can quote are 1.19–1.33× (#5596) and 1.19–1.36× for GN-routing variants (#5726/#5750). Treat 4.2× as unverified.

**2026-06-06 · Closure.** [#5167](https://github.com/marin-community/marin/issues/5167) closed "Closing, we went with MuonH here"; [#4034](https://github.com/marin-community/marin/issues/4034) closed "Integrated MuonH."

---

## 4. The GPU detour — why Muon got shelved there

Muon is cheap on TPU and expensive on GPU, and the reason is structural, not a lowering bug.

**The measurement.** On an identical single-node 8×H100 shape ([#6493, 2026-06-18](https://github.com/marin-community/marin/issues/6493#issuecomment-4738899640)): MuonH5 at **4.75% MFU / 1.492 s per step** vs an SGD A/B at **16.19% MFU / 0.438 s per step**. Isolated Newton-Schulz dot kernels run at ~70–77% of nominal dense bf16 peak, so dlwh concluded "this no longer looks like a bad expression-form/XLA-lowering problem. The Muon train-step gap is mostly real algorithmic/comm work."

**The structural argument** ([dlwh, #gpu, 2026-06-21](https://discord.com/channels/1354881461060243556/1399998407657001062/1518073937186259134), cross-posted to [#6493](https://github.com/marin-community/marin/issues/6493#issuecomment-4760616407)):

> "Muon is not super friendly to FSDP-ish setups (except at very large batch size), because the optimizer-side compute and comms are very large relative to the forward/backward pass. Muon NS compute itself will scale away, but the comms are large and do not scale away with node count. The comms are tricky to hide. **If we are to become a muon shop, I think we will need to embrace PP.**"

Newton-Schulz "wants each 2D matrix to be local," so converting grouped-Muon updates back into FSDP layout is an expert-weight-sized event — **~121.9 GiB bf16 globally** — that doesn't shrink with node count and sits after the backward where it can't be overlapped. On a 2-node run the roofline dashboard ([#6573](https://github.com/marin-community/marin/issues/6573)) put it at "**~500ms of the 1500ms step time in this particular 2 node run**" ([Discord, 2026-06-23](https://discord.com/channels/1354881461060243556/1399998407657001062/1518855948100177991)).

**The contrast on TPU** ([#6493, v4-1024 profile](https://github.com/marin-community/marin/issues/6493#issuecomment-4787096060)): step 22.4 s at 21.4% MFU, "Optimizer NS + bookkeeping ~206 ms, **~0.9% of step**." Two corrections worth carrying: this is **v4-1024, not the v4-2048 hero shape**; and the popular "XLA hides the cost" framing is *not* what the profile shows — NS is simply **small** (~50 ms compute, ~70 ms observed comm). The original "not sure how it is hiding the cost" / "xla tpu is magic" exchange was a **measurement Larry retracted the next day** ("this metric was, in fact, not accurate"); 0.9% is the post-correction figure.

**The decision** ([dlwh, #gpu, 2026-06-26](https://discord.com/channels/1354881461060243556/1399998407657001062/1520200189690773564)):

> "Outcome of OA meeting on GPU progress… We're currently at ~19.9 MFU on 4 nodes w/ SGD/Adam… **Muon needs to go for now. 10% step-count improvement isn't worth atm. We'll need much faster interconnect or PP.**"

[#6693](https://github.com/marin-community/marin/issues/6693) records the outcome — one workstream row, "Stabilize grouped-bank Muon harness | MuonH | **SKIP** | [no-op for now] | Blocker: No muon" — but the *rationale* lives in Discord and the *evidence* in #6493.

> **The "~10% step-count" figure is not grounded in any experiment in this corpus.** It brackets on both sides: Kaiyue Wen projected the speedup "shrinks to 4%" at 1e23 ([Discord, 2026-05-13](https://discord.com/channels/1354881461060243556/1504268862126952551/1504269560306602165)); #5619's gate-2 gave 1.17× → 1.04× wall-clock with the 1e23 margin "within the ~0.007 residual noise floor"; and #5596's gate-2 *failed* its 1e23 criterion. The nearest "10%" string is a Larry aspiration from 2026-04-28 ("My hope is that we can get… at least a 10-15% compute efficiency gain") — a target, not a measurement. **The decision is sound; the number cited for it understates the case if anything.**

**The escape hatch was investigated and parked.** Russell Power's PP work ([#6532](https://github.com/marin-community/marin/issues/6532), [PR #6534](https://github.com/marin-community/marin/pull/6534)) validated the thesis — "PP makes Muon viable at scale by sharding the optimizer per stage" — with real numbers: on 8×H100, FSDP's Muon tax is **+0.76 s (−43% throughput)** vs **+0.09 s** for PP-1F1B; at 2 nodes the FSDP tax grows to +0.92 s while PP-1F1B is **3.2× smaller**. But PP's *base* throughput lost: **0.78× of FSDP** on a single v6e-8 host, only pulling ahead (**1.22×**) at v6e-32 once FSDP's all-gather crosses DCN. Both #6423 and #6534 were **closed unmerged** and consolidated to `rjpower/marin-experiments`. The related staleness study ([#6431](https://github.com/marin-community/marin/issues/6431#issuecomment-4723408436)) found **Muon is 2.6–5.3× more delay-robust than AdamH** on a d512 MoE on v6e-8, with a Muon-specific weight-prediction corrector closing ~46% of the gap — a real argument *for* Muon under PP, but on a ~290M-param toy.

---

## 5. Where this is actually heading — and the staleness caveat

**Read this section before repeating "TPU keeps Muon, GPU dropped it."** That framing was accurate in late June and is materially stale by mid-July.

This corpus has **asymmetric cutoffs**: GitHub artifacts end **2026-07-05**, while Discord runs to 07-16, wandb to 07-17, and summaries cover 07-06→07-12. So **every GitHub item from the final week (#7118, #7024, #6979, #7012) exists only as weekly-summary prose or an incidental Discord link** — second-hand by construction.

With that caveat, the July signals:

- **Muon is back inside the measured GPU stack.** Larry's GPU MFU Learning Path (#6979) builds a full 8×H100 MoE up to **26.7% MFU for the full `model.py`**, with "**real Muon's Newton-Schulz orthogonalization the single biggest give-back at ~3.7 points**," and weak-scaling essentially flat to **26.5% at 64 H100s** ([summary 2026-07-06→07-12](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)). That is a Muon-inclusive stack comfortably **above the 20% bar** Muon was shelved to reach — and Muon costing 3.7 points, not the 11.4 of the June A/B.
- **The named precondition is advancing.** dlwh, [2026-07-10](https://discord.com/channels/1354881461060243556/1399998407657001062/1525217063830949888): "codex is seemingly making good progress on jaxpp… **Hopper is back on the menu I think**." PP was explicitly the condition for Muon on GPU.
- **But no formal reversal exists.** A sweep of the July #gpu channel turned up no re-adoption statement, #7024's best point is still under 20 MFU, and #6998 still lists "Muon carries high overhead at high sparsity" as a live Hopper obstacle.

**Fair statement of the current position:** MuonH is settled and in production on TPU; on GPU it remains formally shelved from the June decision, but the empirical basis for that shelving has substantially eroded and the decision looks ripe for revisiting rather than settled.

### Variant research: active, none landed

| effort | status |
|---|---|
| [#6388](https://github.com/marin-community/marin/issues/6388) MuonH magnitude-direction decoupling (Kaiyue Wen) | Best candidate. Real loss win at both scales, but **gate-1 passes at d512 only** (1.034×); d768 ~0.99. The apparent throughput "sink" was a **checkpointing artifact**, true overhead ~2.6%. Not landed. |
| [#6634](https://github.com/marin-community/marin/pull/6634) curvature-corrected Muon | Curvature penalty **alone is neutral**; the Mudam warm start gives ~0.003. qwen3-130m, single scale — author's own next step is "more control seeds." |
| [#6589](https://github.com/marin-community/marin/pull/6589) Mudam | Beats Muon 1.1653 vs 1.1673 — **0.0020 at 130m, one scale, one seed**. |
| [#6588](https://github.com/marin-community/marin/pull/6588) left-preconditioned Muon | **Negative**, badly (worst variant +0.087 vs Muon). Byproduct insight: few-step NS *saturates* the noise amplification that exact inverse-sqrt blows up — "that cap is the entire regularization." |
| [#6506](https://github.com/marin-community/marin/issues/6506) gain-gated, [#6538](https://github.com/marin-community/marin/pull/6538) activation-aware | Both **negative** at 130m; #6538 after a self-corrected metric error. |
| [#6301](https://github.com/marin-community/marin/pull/6301) PRISM | Not a Muon variant — a separate optimizer benchmarked *against* Muon. Ties on bpb to 520m but is **1–4% slower, ~16% slower at 1_2b**. Not adopted. |
| [#6527](https://github.com/marin-community/marin/issues/6527) Prime Intellect FSDP-friendly Muon | **Never actually evaluated** — the only response is a bot triage that couldn't fetch the source ("the sandbox blocked all network access"). Open lead. |

One July item, **#7118**, adds two error-aware Muon feedback policies; the summary reports Hessian-corrected Muon beating a fresh control at four of five learning rates by a mean **−0.000587 C4-en bpb**. **The primary document is not retrievable in this corpus** — this rests entirely on the weekly summary, which carries its own hedge: "a single observation, not yet a replicated optimizer win."

Note nearly all variant work is **qwen3-130m single-scale**, where #6634 documented that merely moving hardware (v6e-8 vs v5p-8) shifts the MuonH anchor by ~0.002 — **larger than most of the reported wins**.

---

## 6. Open questions worth flagging

1. **The 1e23 crossover.** MuonH's fitted exponent is *worse* than AdamH's, and the hero run sits past the projected crossover. That projection rests on 4 points and may well be wrong — but no one has retired it, and the hero run is the natural test.
2. **The 13/3 ratio** was tuned for AdamH's geometry and never re-derived for MuonH ([#5621](https://github.com/marin-community/marin/issues/5621)).
3. **Does the GPU shelving still hold** given #6979's Muon-inclusive 26.7% and jaxpp progress? No one has written the answer down.
4. **Silent-fallback class of bug.** The same divisibility footgun was documented on GPU nine days before it cost a day of TPU hero-run debugging. The postmortem's process asks are not visibly implemented.

---
<!--provenance-->
> *Data: marinmirror — 86367 chunks, built 0h ago · summaries through 2026-07-06_2026-07-12. Frozen eval corpus (2026-07-18-chunked); no refresh this run. Note asymmetric cutoffs: GitHub ends 2026-07-05, Discord 07-16, wandb 07-17.*
>
> *Query: "what is our current muon approach, and how did we end up here?"*
>
> *Sub-queries: "MuonH in the May Recipe production MoE config — parameter groups, hyperparameters, LR refit #5951, what hero run #6044/#6153 launched with" · "Why Muon was shelved on GPU — Newton-Schulz cost under FSDP, #6493 profiling, roofline dashboard #6573, OA sync decision #6693" · "Pipeline parallelism as escape hatch for Muon's FSDP tax — #6532/#6534 zero-bubble, #6431/#6423 delayed-gradient staleness" · "Active Muon variant research — #6388 decoupling, #6301 PRISM, #7118 error-aware, #6527 Prime Intellect FSDP Muon" · "How Marin came to adopt Muon — #5596/#5167/#4034 optimizer studies, effective-speedup frontier, when MuonH replaced AdamH" · "Muon on TPU for the 67B-A2B hero run — sharded Newton-Schulz bug, v4-1024 vs v4-2048, current status" · adversarial skeptic pass re-testing six load-bearing claims.*
