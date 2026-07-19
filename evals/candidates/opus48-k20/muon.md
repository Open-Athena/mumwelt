# Our Muon approach, and how we got here

## The short version

We are a **MuonH shop on TPU and an Adam shop on GPU.** That split is not a matter of taste — it was forced by a measurement, and it is the single most important fact about our current optimizer stance.

**MuonH** — Newton–Schulz orthogonalization followed by a **Frobenius-hyperball** projection ("H" = hyperball) — is the optimizer running the production 67B-A2B Grug MoE 10T pretrain on TPU v4-2048 ([#6044](https://github.com/marin-community/marin/issues/6044#issuecomment-4820179240)). It got there via ~14 months of bake-offs, one spectacular failure, several rejected variants, and a gate that it passed only on the second attempt after its learning-rate law was refit.

On the **GPU/H100 path, Muon was dropped on 2026-06-26** and has not been re-enabled as of the freeze. dlwh, reporting the outcome of the OA meeting ([#gpu](https://discord.com/channels/1354881461060243556/1399998407657001062/1520200189690773564)):

> "We're currently at ~19.9 MFU on 4 nodes w/ SGD/Adam. Some low hanging fruit remains. **Muon needs to go for now. 10% step-count improvement isn't worth atm. We'll need much faster interconnect or PP.**"

---

## 1. What MuonH actually is

MuonH is **not** plain Muon. It is a three-group optax chain routed by parameter path, per the authoritative close-out job summary in [#6044](https://github.com/marin-community/marin/issues/6044#issuecomment-4820179240):

> "**MuonH (Muon + Frobenius hyperball)**: Newton-Schulz orthogonalization on weight matrices followed by a scale-invariant Frobenius-hyperball projection. Three LR groups:
> - **muonh** — 2D weight matrices and stacked MoE expert weights (NS + hyperball)
> - **adamh** — `lm_head` / `output_proj`. AdamH = Adam moments + Frobenius hyperball (`scale_by_adamh`); no Newton-Schulz. Uses the same LR schedule as muonh.
> - **adam** — biases, RMSNorm scales, router weights, gated-norm scales, and (via `rmsnorm_to_adam=True`) stacked RMSNorm scales"

The "H" is load-bearing and its meaning is often mis-guessed. It is **Frobenius hyperball**, not "hybrid" or "heuristic" — pinned by the four-way GatedNorm routing table in [#5750](https://github.com/marin-community/marin/issues/5750#issuecomment-4456961403), which distinguishes `muonh` ("NS + Frobenius hyperball") from `muon` ("NS + Keller `√(max(1, fan_out/fan_in))`, NO hyperball"). That comparison's verdict ([#5750](https://github.com/marin-community/marin/issues/5750#issuecomment-4465767626)):

> "`gn → muonh` (NS + Frobenius hyperball) is the clear winner of the 4-way GatedNorm routing comparison. The NS direction + hyperball pair is what matters — using either piece alone … loses to the combined recipe."

**Why `lm_head` stays on AdamH** is not arbitrary either — it traces to a measured Muon weakness on out-of-distribution code data (§3.4).

### Production hyperparameters (v4-2048, d=2560, 67B total / 2.01B active)

All from the [#6044 job summary](https://github.com/marin-community/marin/issues/6044#issuecomment-4820179240). Values change at the step-15,000 batch-size ramp (33.5M → 67.1M tokens/step):

| knob | value |
|---|---|
| `muonh_lr` | 0.003733 → **0.005279** at step 15,000 (×√2 with the BS doubling) |
| `adam_lr` | 0.000861 → **0.001218** (= muonh_lr / (13/3); also ×√2) |
| `adamh_lr` | = `muonh_lr` (same schedule) |
| `beta1` | 0.9062 (unchanged across the ramp) |
| `beta2` | 0.95 (clamped both before and after the ramp) |
| `weight_decay` | 0 |
| `max_grad_norm` | None — no clipping |
| warmup | 0.01 (1% ≈ 3,000 steps) |
| `min_lr_ratio` | 0.05, linear schedule |
| NS iterations | `backend_steps: 5` in the run config |

`min_lr_ratio` moved off zero deliberately ([#6044](https://github.com/marin-community/marin/issues/6044#issuecomment-4820298253)): *"LR decay to 0.05 instead of 0. This is because we will extend out from 64k->262k, and do subsequent posttrain. decay to zero is only optimal if we have no subsequent weight updates."*

The LRs come from the refit MuonH heuristic in [#5951](https://github.com/marin-community/marin/issues/5951). **Quote the correction, not the headline** — the first published fit had a units bug ([#5951](https://github.com/marin-community/marin/issues/5951#issuecomment-4549326686)):

> "`muonh_lr = 18.31 · tokens^-0.395 · dim^-0.150 · bs^0.5    (R² = 0.996, 17 fit cells)`"

and, importantly:

> "So at compute-optimal the new recipe is **slightly below** the old … The earlier '1.6–1.7× bump' claim was an artifact of the units bug."

The Newton–Schulz coefficients default to a **quintic** five-tuple set (a different triple per iteration), made configurable alongside `simple`/`polar_express`/`aol` in [PR #2545](https://github.com/marin-community/marin/pull/2545). Worth knowing: in June 2026 nobody could immediately say where those coefficients came from — Larry posted the values and then said *"im not actually sure where these coef came from"* and *"hmm maybe our coef are not optimal"* ([#optimizers](https://discord.com/channels/1354881461060243556/1382240679765217342/1516208109683605644)); provenance was traced to a nanoGPT-speedrun writeup, and an ablation was opened as [#6404](https://github.com/marin-community/marin/issues/6404) — **which has no verdict comment in the freeze.**

---

## 2. Where it runs — the hardware split, and the number behind it

This is the crux. The same optimizer is nearly free on TPU and ruinous on H100.

| platform | measurement | source |
|---|---|---|
| **TPU v4-1024**, d=2560, BS=4096, seq=8192 | optimizer NS + bookkeeping ≈ **206 ms of a ~22.4 s step (~0.9%)** | [#6493](https://github.com/marin-community/marin/issues/6493#issuecomment-4787096060) |
| **H100, 2 nodes** | Muon ≈ **500 ms of a ~1500 ms step (~33%)** | [dlwh, 2026-06-23](https://discord.com/channels/1354881461060243556/1399998407657001062/1518855948100177991), dashboard [#6573](https://github.com/marin-community/marin/issues/6573#issuecomment-4775876844) |
| **H100, single node 8×**, d2560 May recipe | MuonH5 **1.492 s/step, 21,962 tok/s, 4.75% MFU** vs SGD **0.438 s/step, 74,848 tok/s, 16.19% MFU** → ≈**1.05 s/step** attributable to Muon | [#6493](https://github.com/marin-community/marin/issues/6493#issuecomment-4738899640) |

dlwh's diagnosis is that **the bottleneck is communication, not the Newton–Schulz FLOPs** ([#6493](https://github.com/marin-community/marin/issues/6493#issuecomment-4760616407)):

> "**Muon is not super friendly to FSDP-ish setups (except at very large batch size)**, because the optimizer-side compute and comms are very large relative to the forward/backward pass. Muon NS compute itself will scale away, but the comms are large and do not scale away with node count. The comms are tricky to hide. If we are to become a muon shop, I think we will need to embrace PP."

> "For the expert weights alone that payload is about **121.875 GiB bf16 globally**. That cost does not shrink much just because we add more nodes… So we are basically **doubling the all-gather cost per step** compared to a non-Muon setup. But it's a not-easily-overlappable doubling, which is the problem."

Two escape routes were tried and both failed:

**Grouped/stacked Newton–Schulz ([#6493](https://github.com/marin-community/marin/issues/6493))** — negative:

> "The main result so far is **negative for simple grouping**: unbounded grouping OOMed during autotune, bounded grouped/padded MuonH did not improve full training throughput, and harness runs show the **isolated NS/dot kernels are already reasonably efficient**."

The NS kernels themselves hit 63.4% of nominal bf16 dense peak standalone; it is the slice/restore boundary that erases the win.

**Pipeline parallelism ([#6532](https://github.com/marin-community/marin/issues/6532) / [PR #6534](https://github.com/marin-community/marin/pull/6534))** — PP *did* amortize Muon exactly as predicted. Measured Muon tax on 2 nodes of H100 at ~6.1B: FSDP **+0.92 s** (−60% throughput) vs PP-1f1b **+0.29 s**, i.e. **3.2× smaller** ([#6532](https://github.com/marin-community/marin/issues/6532#issuecomment-4794455018)). But PP lost on absolute throughput and was parked by Russell Power on 2026-06-25 ([close-out](https://github.com/marin-community/marin/issues/6532#issuecomment-4802268924)):

> "**Parked.** … bubble-free PP is grad-exact vs the oracle and unlocks the memory-scaling win …, but it doesn't beat FSDP on throughput at the 100M–1B / v6e-8 scale … PP only wins once the FSDP all-gather crosses DCN (~1.22× FSDP on v6e-32), and the residual gap is MFU + recompute, not pipeline bubble."

Two corrections worth carrying, because the weekly-summary narrative compresses them:
- **The 0.78× / 1.22× PP-vs-FSDP figures are TPU v6e-8 / v6e-32, not H100.** On H100 the gap was far worse (FSDP 57,056 tok/s vs 8-way PP 8,077 tok/s, ~7× slower — [#6532](https://github.com/marin-community/marin/issues/6532#issuecomment-4790773618)).
- **The PP "structural memory wall" was a bug, not a limit.** Per-microbatch weight-grads were held M-wide and summed only at the end; fixed, an 8M-token batch dropped to 22.8 GiB/device ([#6532](https://github.com/marin-community/marin/issues/6532#issuecomment-4800731253)).

With both routes closed, Muon left the GPU path. It is encoded in the [#6693](https://github.com/marin-community/marin/issues/6693) ">20 MFU at 128 GPUs" workstream tracker as a row reading `SKIP … No muon`. The GPU effort is now a hand-written fused expert-parallel MoE kernel ([#6597](https://github.com/marin-community/marin/issues/6597)), and the Jul 6–12 digest's GPU section covers that kernel and B200 MFU with **no Muon item either way**.

**Has it been revisited?** No source in the freeze re-enables it. The nearest signal is dlwh on 2026-07-10: *"codex is seemingly making good progress on **jaxpp** … **Hopper is back on the menu** I think"* ([#gpu](https://discord.com/channels/1354881461060243556/1399998407657001062/1525217063830949888)) — notable because the stated condition for Muon's return was "much faster interconnect **or PP**", and jaxpp is PP infrastructure. But the linked issue #7024 is not in the corpus, and **nothing states Muon is back.** Treat this as a precondition being rebuilt, not a reversal.

---

## 3. How we got here

### 3.1 Entry: a bake-off where Muon won only a middle band (May 2025)

Muon enters Marin not as a favorite but as 1 of 10 entrants in [#1290 "Fantastic Pretraining Optimizers And Where to Find Them"](https://github.com/marin-community/marin/issues/1290) (WhenWen, 2025-05-17), alongside Scion, SOAP and Kron. Dense LLaMA-style, 130M–1.2B, TPU:

> "**Max Speedup ≤ 1.4×** … No optimizer achieved the 2× step-wise speedup from prior claims; the best was ≈ 1.4× over AdamW."
> "**Regime-Dependent Winner** — Muon wins in 1×–4× Chinchilla; Soap/Kron takes over at ≥ 8× and in over-trained (16×) settings."

That last clause matters — it predicted Muon would *lose* in the overtrained regime. It was later contradicted by our own MoE data (§3.6).

### 3.2 The first big swing, and its failure (June–July 2025)

[#1380](https://github.com/marin-community/marin/issues/1380) swapped a 32B mid-run onto Muon ("let's use it to see if we can yolo muon without blowing everything up"). dlwh's close-out, 2025-07-02, is four words:

> "**was great until it wasn't**"

### 3.3 MuonClip: evaluated, never built (July 2025)

After Kimi K2, MuonClip was seriously discussed and explicitly rejected on engineering-cost grounds — willheld, [2025-07-24](https://discord.com/channels/1354881461060243556/1382240679765217342/1397745652238389299):

> "**Main problem is that MuonClip is actually kind of a pain to implement because we'd need to start tracking logits and passing them to the optimizer which is somewhat invasive.**"

There is **no evidence in the corpus that MuonClip was ever implemented.** A related muP effort ([#1570](https://github.com/marin-community/marin/issues/1570)) was implemented in haliax/levanter and **failed its goal**: *"None of the muP stuff leads to muTransfer from 128 to 1024 though..."*

### 3.4 The hyperball, and the OOD-code problem that shaped the group split (Oct 2025 – May 2026)

AdamH/MuonH enter via the hyperball speedrun [PR #1774](https://github.com/marin-community/marin/pull/1774) (2025-10-14).

On the MoE track the first gate was **negative**. [#5167](https://github.com/marin-community/marin/issues/5167)'s Vizier search close-out (2026-04-28):

> "d512 best Muon Vizier r2: `3.80732` … delta `-0.00303` vs AdamH … d768 best Muon Vizier r2: `3.46759` … delta `+0.03372` … **Current gate-1 evidence does not support promoting Muon over AdamH broadly; AdamH remains the stronger scaling choice at d768.**"

The diagnosis came from [#5585](https://github.com/marin-community/marin/issues/5585)'s 18-step feature walk from modded-nanogpt to our MoE recipe. Larry's own close-out — worth quoting because he is correcting the agent-written commentary above it ([#5585](https://github.com/marin-community/marin/issues/5585#issuecomment-4411229059)):

> "Main conclusion is the full **30% Muon gain in distribution** holds all the way through to our recipe… However, **Muon does absolutely terrible on OOD data, mostly code.** The most likely factors driving this are that AdamH had logit z loss and muon did not, AdamH had regularized param sizing (in particular lm_head and gated_norm)."

> "Many tokens the AdamH model assigns a ~2% chance to, and Muon assigns a 0.0000001% chance to."

**This is why `lm_head`/`output_proj` remained on AdamH in the shipped recipe.**

### 3.5 The gate that MuonH passed — and the one it failed

[#5596](https://github.com/marin-community/marin/issues/5596) / [PR #5597](https://github.com/marin-community/marin/pull/5597) swapped MuonH for AdamH on matrix-shaped params at four scales, v5p-8, v16 MoE baseline held fixed. Gate-2 close-out ([#5596](https://github.com/marin-community/marin/issues/5596#issuecomment-4422416900)) — measured wall-clock effective speedup vs v16 AdamH:

| scale | budget | baseline macro_loss | MuonH macro_loss | wall-clock speedup |
|---|---|---|---|---|
| d512 | 2.19e17 | 3.8104 | **3.7542** | **1.33×** |
| d768 | 1.70e18 | 3.4339 | **3.3988** | **1.26×** |
| d1024 | 9.00e18 | 3.1605 | **3.1357** | **1.22×** |
| d1280 | 2.83e19 | 3.0065 | **2.9888** | **1.19×** |

> "- ✅ **Wall-clock speedup > 1 at all four scales** — passes the all-scales criterion.
> - ❌ **1e23 projection ~0.011 higher** than baseline — fails the 1e23 projection criterion"

The failure was a **projection, never a measurement** — MuonH's fitted exponent (α = 0.0906) was shallower than the baseline's (0.0941), so the curves crossed near C ≈ 1.65e21. Kaiyue Wen's read was that this was an artifact of not having refit the LR for Muon. **That turned out to be right.** After the [#5951](https://github.com/marin-community/marin/issues/5951) LR refit, the [#6074](https://github.com/marin-community/marin/issues/6074) isoFLOP sweeps recovered **α = 0.0941, identical to the v16 baseline**, dissolving the crossover:

> "Dropping 1e18: `loss = 1.6 + 88.90 · C^-0.0941` (same α as v16, uniform 2.12× equal-TPS speedup)"

**Do not conflate those two speedups.** The 1.19–1.33× is *MuonH alone* at fixed v16 architecture (v5p-8). The **2.12×** is the *entire May Recipe* (256 experts, PKO, half-RoPE, split w_gate/w_up, routing renorm 2.5, MuonH) vs the v16 scaling law, on v5p-32.

### 3.6 Adoption (June 2026)

Adoption was a **borderline** call, and honest about it — [#5763](https://github.com/marin-community/marin/issues/5763#issuecomment-4471778083) records d512 speedup 1.078, d768 **0.991**, d1024 **0.966**:

> "Strictly by the agent.md two-scale gate-1 rule …, this is a borderline pass — d768 is 0.991 … Adopted as the canonical baseline anyway because: 1. d512 quality lead is large and reproducible. 2. d768/d1024 deltas … well within run-to-run noise. 3. The recipe is **simpler**."

ClassicLarry closed the Great-10T Muon trackers on 2026-06-06 with two words — **"Integrated MuonH"** ([#4033](https://github.com/marin-community/marin/issues/4033#issuecomment-4640135827), [#4034](https://github.com/marin-community/marin/issues/4034#issuecomment-4640128969)) — and the historical AdamH-vs-Adam gate [#4042](https://github.com/marin-community/marin/issues/4042) now carries a golem TL;DR reading *"the optimizer path has moved toward MuonH, with Adam used where it is safer for embeddings."* The recipe landed on `main` via [PR #6153](https://github.com/marin-community/marin/pull/6153).

One genuine reversal of the 2025 prior: at **5000× tokens-per-param**, Larry found *"MuonH maintained the advantage gap the whole way. I would have thought that in extreme overtrain, model will saturate its capacity and details of optimizer will matter less"* ([#moe](https://discord.com/channels/1354881461060243556/1365044508546568372/1516145091734016162)) — the opposite of #1290's "Soap/Kron takes over at ≥8×".

### 3.7 The v4-2048 scare — a sharding bug wearing a numerics costume (June 2026)

Before the hero run could launch at full slice, it diverged. The symptom looked like a precision problem and fp32 appeared to fix it. It was not. Root cause ([#6044](https://github.com/marin-community/marin/issues/6044#issuecomment-4819379417)):

> "The root cause of the issue is that the sharded NS calc in Muon was **silently dropping when num_matrices didn't evenly divide into num_chips**. This was originally written on v4-1024 where 256*26 is divisible by 512 chips. However it isn't divisible by 1024 chips. The fix was to explicitly replicate the NS calc over the same size as the replica axis, and also patch the silent failure."

> "Increasing precision to fp32 was helping the non orthogonalized approach not diverge, but **not addressing the root cause**. Lack of orthogonalization explains why the loss curve started better and then got worse over time, similar to how adam can beat muon in first couple steps."

The consequence is that a **conclusion we had drawn was wrong**: the apparent instability at 67M-token batch was an artifact, *"it is likely that the instability at 67 million token batch size is only due to failing to orthogonalize the gradient. So, we likely could do the full run at 67 million token batch size."* The batch ramp was kept anyway for token efficiency. Larry's own retro is the process lesson: *"this issue should have errored loudly, but I did not review this agent written resharding close enough."*

---

## 4. Where the run stands, and what's still open

**Production run (TPU v4-2048, MuonH).** The canonical config is `moe_67b_a2b_d2560_ep1_rep16_bs4096_seq8192_sw2k_v4_2048_muon_10T`. Smoke-test MFU at the canonical d=2560 config ([#6044](https://github.com/marin-community/marin/issues/6044#issuecomment-4812848295)): v4-1024 has the best per-chip MFU (**21.54%**) because d=2560 shards cleanly at 512 chips, but v4-2048 + replica=8 + BS=8192 wins absolute throughput (**18.62% MFU, 2.57M tok/s**), projecting ~45 days vs v4-1024's ~78. **These are smoke/short runs — no sustained production MFU percentage is quotable from the freeze.** A real regression is on record: after a crash/restart at step 31,962, step time went 26.15 s → 30.05 s (~15% slower) and *"never recovered"* ([#moe-hero-run](https://discord.com/channels/1354881461060243556/1365044508546568372/1524108412894318603)), best guess *"we got a less good topology."*

Per the [Jul 6–12 weekly summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html), the run crossed step 39,000 (~2.114T tokens, ~21% of horizon) holding MFU near 18.6%, and the first intermediate cooldown ([#6811](https://github.com/marin-community/marin/issues/6811)) finished at step 42,149 at Paloma macro loss **2.2772** — essentially matching the run's *preregistered* 8T-token stage-1 target of 2.269 ([#6044](https://github.com/marin-community/marin/issues/6044#issuecomment-4820008980)) from a checkpoint at only ~2.1T tokens. **That is a favorable early signal, not a retired preregistration** — the cooldown fully decays LR, switches data mix, and evaluates at 65,536 context rather than the preregistration's 8,192.

**Active optimizer research (all small-scale, none adopted).** A June–July sweep of Muon variants at qwen3-130m (`eval/paloma/c4_en/bpb`, Muon control ≈ 1.1673) is mostly negative: gain-gated Muon *"does not beat Muon"* ([#6506](https://github.com/marin-community/marin/pull/6506)), activation-aware Muon *"monotonically hurts"* ([#6538](https://github.com/marin-community/marin/issues/6538#issuecomment-4770528871)), left-preconditioned Muon *"does not beat Muon"* ([#6588](https://github.com/marin-community/marin/issues/6588#issuecomment-4791400232)). The one positive is **Mudam / curvature-corrected Muon** ([#6634](https://github.com/marin-community/marin/issues/6634#issuecomment-4796074066)): a Mudam warm start + curvature refinement beats same-compute MuonH by **~0.003 bpb** — author-flagged as near the noise band, single scale, not replicated, not scaled. Earlier variants tried and rejected: **MuonRemez** (*"still seems to be worse"*, [#2284](https://github.com/marin-community/marin/issues/2284)), **MuonHT** (*"almost identical training curves"*, [#2434](https://github.com/marin-community/marin/issues/2434)), **HybridMuon** (*"simply a weaker optimizer here"*, [#6119](https://github.com/marin-community/marin/issues/6119)), and **Dion**, closed 2026-06-01 as *"primarily useful for tensor-sharded models, which doesn't appear to be used in any grug configs"* ([PR #5625](https://github.com/marin-community/marin/pull/5625)). `muon_epsilon` was swept and found insensitive from 1e-6 to 1e-16 ([#5933](https://github.com/marin-community/marin/issues/5933)).

**The strongest replicated Muon-favorable result** in the recent window is not about loss at all: under injected gradient staleness, **Muon is 2.6–5.3× more delay-robust than AdamH** (d512 grug MoE, 3000 steps, 8 arms, seed-replicated — [#6431](https://github.com/marin-community/marin/issues/6431#issuecomment-4723408436)). That property is what made PP attractive as a Muon amortizer, and it is why a working jaxpp would matter.

**Muon in RL is currently a confound, not a result.** Ashwinee Panda's cross-framework repro ([#6915](https://github.com/marin-community/marin/issues/6915)) originally reported large gains, then he self-corrected on 2026-07-03 that the run used Muon where the reference used AdamW: *"welp i just looked at the issue for the first time and i apparently used the wrong optimizer 💀"*. Per the [Jul 6–12 summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html), the parity run lands at roughly **+0.8 to +1.4 MATH500 against the reference's +8.4**, with AIME regressing under the parity contract — and Muon-vs-AdamW remains one of five un-isolated candidate causes.

---

## 5. Gaps and caveats

- **#7118 ("error-aware Muon feedback policies", Hessian-corrected Muon) is NOT retrievable in this freeze.** I verified this directly — both the issue and pull URLs return `not found in corpus`. It exists only as a description in the [Jul 6–12 weekly summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html): *"Kaiyue Wen opened a speedrun PR #7118 adding two error-aware Muon feedback policies; in a single-seed 40-run sweep, Hessian-corrected Muon beat a fresh Muon control at four of five learning rates by a mean 0.000587 C4-en bits-per-byte — a single observation, not yet a replicated optimizer win."* I cannot verify that against a primary source. The GitHub side of the freeze thins out above ~#6789.
- **The "10% step-count improvement" figure** in dlwh's shelving decision is stated without an attached measurement. It sits alongside much larger, differently-scoped claims — #5596's measured 1.14–1.32× step-wise, and Larry's "30% in distribution". Treat 10% as the GPU-era working number at production scale, not a re-measurement.
- **No sustained production MFU** for the post-fix v4-2048 run is quotable; only smoke tests and step times.
- **[#6404](https://github.com/marin-community/marin/issues/6404)'s NS-coefficient ablation has no verdict** in the freeze — so "are our quintic coefficients optimal?" is formally still open.
- A widely-held earlier belief that Muon overhead was **~1% vs Adam** was TPU-derived and was overturned on GPU by the measurements in §2. Both numbers are correct *for their own hardware*; the error was carrying one across.
- Corpus is 49h old at time of writing and refresh was disabled for this run, so anything after 2026-07-16 is invisible.

---
<!--provenance-->
> *Data: marinmirror — 68026 chunks, built 49h ago · summaries through 2026-07-06_2026-07-12. Frozen eval corpus (MARIN_EVAL_FREEZE=2026-07-16), retrieval pool pinned at k=20; no refresh performed.*
>
> *Query: "what is our current muon approach, and how did we end up here?"*
>
> *Sub-queries: "MuonH in the 67B-A2B Grug MoE production run #6044 — current optimizer config, hyperparameters, Newton-Schulz sharding, TPU v4-2048" · "why Muon was shelved on GPU/H100 — Newton-Schulz cost roofline dashboard #6573, grouped Newton-Schulz negative result #6493, FSDP Muon tax" · "MuonH vs AdamH optimizer studies #5596 #5167 #4034 — scaling-law evidence, compute-equivalent speedup, why Muon was adopted in the May Recipe" · "history of Muon adoption in Marin — first Muon experiments, Muon variants tried (Muon, MuonH, MuonClip, Scion, Dion), learning-rate and weight-decay tuning, muP/scaling transfer" · "error-aware Muon feedback policies speedrun PR #7118 — Hessian-corrected Muon, current open Muon research and Muon numerical instability" · "pipeline parallelism zero-bubble #6532 PR #6534 as a fix for the Muon tax under FSDP — throughput verdict and why it was parked"*
