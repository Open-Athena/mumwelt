# Our current Muon approach, and how we got here

## The short answer

Marin runs **MuonH** — Muon's Newton–Schulz-orthogonalized update wrapped in a Frobenius **hyperball** constraint — as the pretraining optimizer for matrix-shaped parameters, but **only on TPU**. On GPU, Muon was deliberately **cut** in late June 2026 because its optimizer-side communication does not amortize under FSDP.

The result is a clean hardware split, stated plainly in the June 22–28 weekly summary:

> "The net is a clean hardware split: TPU keeps Muon (the best effective-speedup frontier point is still a MuonH run), GPU drops it."
> — [summary 2026-06-22_2026-06-28](https://mws.oa.dev/summaries/summary-2026-06-22_2026-06-28.html)

The 67B-A2B "Grug" hero run — the centerpiece 10T-token pretraining job — is a MuonH run on TPU v4-2048, tracked in [#6044](https://github.com/marin-community/marin/issues/6044) and [#6704](https://github.com/marin-community/marin/issues/6704).

---

## 1. What "our Muon approach" concretely is

### MuonH = Muon + hyperball

**AdamH** is Adam plus a *hyperball* constraint: weights are held inside a Frobenius-norm ball whose radius is set from the **initial** weight norm, instead of being regularized by weight decay ([Discord #questions, 2026-03-21](https://discord.com/channels/1354881461060243556/1357080963472949428/1484995860763770980)). **MuonH** applies that same hyperball wrap to the Newton–Schulz orthogonalized direction.

The hyperball wrap is load-bearing, not decoration. The consolidated GatedNorm-routing ablation [#5750](https://github.com/marin-community/marin/issues/5750) tested `gn → muon` ("NS direction, Keller post-scale, no hyperball") against `gn → muonh` (same plus hyperball) and concluded:

> "`gn → muonh` (NS + Frobenius hyperball) is the clear winner of the 4-way GatedNorm routing comparison."

At d512 that is Paloma macro 3.7471 (muonh) vs 3.7557 (no hyperball). A *grad-aware* hyperball variant was also tried and lost at both d512 and d768 ([#5251](https://github.com/marin-community/marin/issues/5251)), so the plain post-hoc projection was kept.

### Parameter-group routing (the 10T hero run)

Per the job summary in [#6044](https://github.com/marin-community/marin/issues/6044) (ClassicLarry, 2026-06-27, run `moe_67b_a2b_d2560_ep1_rep16_bs4096_seq8192_sw2k_v4_2048_muon_10T`), there are three groups, routed by parameter path via `experiments/grug/moe/optimizer.py:create_mask`:

- **muonh** — 2D weight matrices and stacked MoE expert weights (NS + hyperball)
- **adamh** — `lm_head` / `output_proj`; Adam moments + Frobenius hyperball, **no** Newton–Schulz, same LR schedule as muonh
- **adam** — biases, RMSNorm scales, router weights, gated-norm scales, and (via `rmsnorm_to_adam=True`) stacked RMSNorm scales

One reading trap worth defusing: an earlier job summary in the same issue (2026-06-02) lists "GatedNorms" under **muonh**, while the June 27 one lists "gated-norm scales" under **adam**. These are not in conflict — GatedNorm is defined in-thread as "rank-128 low-rank gating after RMSNorm", i.e. rectangular `w_up`/`w_down` **matrices** (→ muonh), whereas the *scales* are 1-D leaves (→ adam). The mask table in [#5735](https://github.com/marin-community/marin/issues/5735) corroborates: muonh = "all attn matrices, all MoE MLP matrices, all 4 GatedNorms". *(This resolution is an inference from "2D weight matrices", not a direct quote.)*

### Learning rates: the May Recipe refit

Optimizer hyperparameters derive from the refit heuristic in [#5951](https://github.com/marin-community/marin/issues/5951) (ClassicLarry, 2026-05-26), fitting `LR = m · tokens^(−a) · hidden_dim^(−b) · batch_size^0.5` over ~150 runs (mostly v5p-32):

> `muonh_lr = 18.31 · tokens^-0.395 · dim^-0.150 · bs^0.5    (R² = 0.996, 17 fit cells)`

Note this is the **corrected** fit. The first fit (`26.9 · …`) had a units bug — the script assumed `intermediate_dim = d*4` while `build_model_config` uses `intermediate_dim ≈ d/2`, inflating `m` by ~1.74×. Larry's correction is explicit: **"The earlier '1.6–1.7× bump' claim was an artifact of the units bug."** Runs launched at the buggy LR were killed and resubmitted. Final values: `lr_coeff = 0.06602`, `lr_tokens_exp = -0.395`, `lr_dim_exp = -0.150`, `adamh_ratio = 13/3`.

At the hero-run config that yields `muonh_lr ≈ 0.003733` → **0.005279** at step 15,000 (×√2 with the batch doubling), `adam_lr = muonh_lr/(13/3)`, `adamh_lr = muonh_lr`, beta1 0.9062, beta2 clamped 0.95, weight_decay 0, no grad clipping, 1% warmup.

---

## 2. How we ended up here — the timeline

### 2025: Muon arrives from outside, as a batch-size argument

The first tracked Muon-vs-AdamW work is an **external** contribution: PR [#1558](https://github.com/marin-community/marin/pull/1558) (leloykun, 2025-09-05), testing that scaling batch size widens the Muon–AdamW gap — the premise being Muon's larger critical batch size. WhenWen picked it up internally in [#1565](https://github.com/marin-community/marin/issues/1565) (2025-09-07), where opooladz raised a fairness-of-tuning objection (a finer search space should be granted to *all* optimizers, not just Muon).

The hypothesis held — WhenWen, 2025-11-23: *"The results confirm this hypothesis across multiple model sizes and training budgets… Performance Gap Systematically Widens with Batch Size."* By 2025-09-25 it was already the de-facto speedrun default ([Discord](https://discord.com/channels/1354881461060243556/1419076916526452807/1420587388366749819): "Seems like you're using muon by default"). PR #1558 itself was auto-closed for inactivity in January 2026; the result lives in #1565.

### Spring 2026: Muon meets MoE, and MuonH wins

dlwh opened [#4033](https://github.com/marin-community/marin/issues/4033) "[moe] Great 10T: Muon perf on MoE" on 2026-03-23 — does the Muon family change the promoted MoE recipe enough to matter?

The answer arrived in two stages, and the distinction matters:

- **Plain Muon ≈ tied AdamH.** The Vizier optimizer ablation [#5167](https://github.com/marin-community/marin/issues/5167) found d512 AdamH baseline Paloma macro **3.81035** vs best Muon Vizier **3.80732** — essentially a wash. Larry closed it 2026-06-06: **"Closing, we went with MuonH here."**
- **MuonH won.** [#5596](https://github.com/marin-community/marin/issues/5596) / PR #5597 (WhenWen, 2026-05-09) swapped MuonH onto every matrix-shaped parameter except the LM head. Against the AdamH v16 baseline (gate-2 on preemptible v5p-8), Paloma macro_loss:

  | scale | budget | AdamH | MuonH | Δ | tok/s Δ |
  |---|---|---|---|---|---|
  | d512 | 2.19e17 | 3.8104 | **3.7542** | −0.056 | +1.5% |
  | d768 | 1.70e18 | 3.4339 | **3.3988** | −0.035 | +2.8% |
  | d1024 | 9.00e18 | 3.1605 | **3.1357** | −0.025 | +2.6% |

  MuonH is both better *and* faster in tokens/sec. But the verdict was called "mixed" for an honest reason: **the gap shrinks with scale, and the two fitted scaling curves cross near C ≈ 1.65e21**. Wall-clock effective speedup was **1.33× / 1.26×** at d512/d768 (#5596).

Larry recorded adoption on [#4033](https://github.com/marin-community/marin/issues/4033) on 2026-06-06: **"Integrated MuonH."**

### June 2026: the GPU divorce

Three weeks of H100 bring-up established that Muon's cost on GPU is structural, not a kernel-tuning problem.

**The measurement.** [#6493](https://github.com/marin-community/marin/issues/6493) "Speed up Grug MoE Muon on single-node GPU" (dlwh, 2026-06-18), single-node 8×H100: *"May140 MuonH5 ran at 1.492s/step … while May141 SGD reached about 0.438s/step"* — Muon costing roughly 1s/step. Its own close-out is negative: *"The main result so far is negative for simple grouping: unbounded grouping OOMed during autotune, bounded grouped/padded MuonH did not improve full training throughput."*

**The diagnosis.** dlwh, [2026-06-21](https://discord.com/channels/1354881461060243556/1399998407657001062/1518073966030487734):

> "Muon is not super friendly to FSDP-ish setups (except at very large batch size)… **Muon NS compute itself will scale away, but the comms are large and do not scale away with node count.** … If we are to become a muon shop, I think we will need to embrace PP."

The mechanism, with the load-bearing number: after computing updates, an expert-weight-sized payload — **~121.875 GiB bf16 globally** — must move back into the FSDP layout, roughly doubling the per-step all-gather cost, and it sits after backward/optimizer compute where it is hard to hide.

**The confirmation.** The roofline dashboard [#6573](https://github.com/marin-community/marin/issues/6573) put a number on it — dlwh, [2026-06-23](https://discord.com/channels/1354881461060243556/1399998407657001062/1518855948100177991): *"muon is still killing us. it's like **~500ms of the 1500ms step time** in this particular 2 node run"* — and, crucially, *"at ~100% efficiency it would still be a bug chunk of time without further sharding."* Even a perfect NS kernel doesn't save it.

**The escape hatch that didn't hold.** Pipeline parallelism genuinely amortizes Muon — [#6532](https://github.com/marin-community/marin/issues/6532#issuecomment-4794455018) (rjpower, 2026-06-24, 2-node H100) measured the Muon tax at **+0.92 s under FSDP (−60% throughput) vs +0.29 s under PP-1F1B, a ~3.2× smaller tax**, with zero-bubble hiding it entirely on a single node. But PP lost on absolute throughput. Close-out, [2026-06-25](https://github.com/marin-community/marin/issues/6532#issuecomment-4802268924): *"Parked. … marin PR #6534 is closed unmerged. … best PP ≈ 22.9k tok/s vs FSDP ≈ 145k tok/s; ~0.78× single-host"*, only winning (~1.22×) once FSDP's all-gather crosses DCN at v6e-32.

**The decision.** dlwh, #gpu, [2026-06-26 22:52](https://discord.com/channels/1354881461060243556/1399998407657001062/1520200189690773564):

> "Outcome of OA meeting on GPU progress: … We're currently at ~19.9 MFU on 4 nodes w/ SGD/Adam. Some low hanging fruit remains. **Muon needs to go for now. 10% step-count improvement isn't worth atm. We'll need much faster interconnect or PP.**"

This is recorded in the workstream table of [#6693](https://github.com/marin-community/marin/issues/6693), where the Muon row reads owner `SKIP`, command `[no-op for now]`, blocker `No muon`.

Two things this decision was **not**: it was not a quality judgment (the trade quoted was a "10% step-count improvement", i.e. Muon still wins on loss-per-step), and it was not framed as permanent — the stated reopen conditions are "much faster interconnect or PP".

**Why TPU is the opposite.** Larry posted the contrast datapoint into the same issue ([#6493, 2026-06-24](https://github.com/marin-community/marin/issues/6493#issuecomment-4787096060)), profiling the identical Grug MoE MuonH path on **v4-1024**: wall-clock ~22.4 s/step, MFU 21.4%, and

> "| Optimizer NS + bookkeeping | ~0 per token | ~6.8e15 FLOPs | **~206 ms | ~0.9%** |"

**Muon is ~33% of a 2-node H100 step and <1% of a v4-1024 TPU step.** XLA absorbs the Newton–Schulz cost. That single contrast is the whole reason for the hardware split.

### June–July 2026: the hero run, and one real bug

The production 67.1B-total / 2.01B-active Grug MoE was staged onto TPU v4-2048 after stalling on CoreWeave H100s. Getting it stable surfaced a genuine Muon bug — ClassicLarry, [#6044, 2026-06-27](https://github.com/marin-community/marin/issues/6044#issuecomment-4819379417):

> "The root cause of the issue is that the sharded NS calc in Muon was **silently dropping when num_matrices didn't evenly divide into num_chips**. This was originally written on v4-1024 where 256*26 is divisible by 512 chips. However it isn't divisible by 1024 chips. The fix was to explicitly replicate the NS calc over the same size as the replica axis, and also patch the silent failure."

The optimizer had been applying un-orthogonalized updates — which, per Larry, *"explains why the loss curve started better and then got worse over time, similar to how adam can beat muon in first couple steps."* Note the second-order lesson: increasing precision to fp32 had *masked* the divergence without addressing the cause. Larry's retro asks for human review of prod-scale diffs and for always replicating a small-scale run on production hardware — *"I did not review this agent written resharding close enough."*

---

## 3. Where it stands at the freeze (2026-07-16)

**TPU — running, healthy.** As of the July 6–12 week the hero run had passed ~2.3T tokens (train loss 1.41, 216k chip-hours, EP held at 1), holding MFU near 18.6%. Larry branched the first intermediate cooldown from step 39,000 ([#6811](https://github.com/marin-community/marin/issues/6811)) — 3,150 steps of LR decay that also stretches context 8× (seqlen 8,192 → 65,536) and switches data mix. It finished at **Paloma macro loss 2.2772**.

That number invites a mistake worth flagging explicitly: the run's **pre-registered target of 2.269** is a *stage-1, 8T-token* prediction evaluated at seqlen 8,192 before final LR cooldown. The 2.2772 comes from a ~2.1T checkpoint that fully decayed LR, switched mix, and evaluated at 65,536 context. Per the summary, the comparison **"is deliberately not apples-to-apples … it doesn't retire the 8T preregistration; it is a strongly favorable early signal."** A second cooldown at 5T is planned for early August; the full 10T checkpoint targets ~August 30. *(Hero-run July numbers are weekly-summary-sourced — see the corpus caveat below. The cooldown W&B run card itself reports TPU v4, 1024 chips, 19% MFU, BPB 0.666.)*

**GPU — still no Muon.** I found **no source stating Muon has been re-enabled on GPU**. The GPU thrust is elsewhere: a Pallas/Mosaic fused MoE kernel ([#6597](https://github.com/marin-community/marin/issues/6597), ~2.09× over the ragged all-to-all baseline), MXFP8 on Blackwell, and a 20–25% MFU target on 128–256 H100s. The GPU MFU learning path (#6979, summary-sourced) re-confirms the cost from the bottom up on a single 8×H100 node: as correctness features are switched on, throughput walks from 34.4% down to 26.7% MFU, with **"real Muon's Newton-Schulz orthogonalization the single biggest give-back at ~3.7 points"** versus ~0.4–1 point for each norm.

The blocker is, however, being cleared. dlwh, [2026-07-10](https://discord.com/channels/1354881461060243556/1399998407657001062/1525217063830949888): *"codex is seemingly making good progress on jaxpp … **Hopper is back on the menu I think**"*. Read precisely: that is about pipeline parallelism, not an announced Muon revival. The chain *Muon needs PP → PP now works → Muon can return* is stated intent, not a landed change.

**Why TPU keeps it.** The scaling-recipe justification is that the best point on the effective-speedup frontier remains a MuonH run — *"the muonh-may-recipe d1280 configuration, roughly 4.2x"* ([summary 2026-06-22_2026-06-28](https://mws.oa.dev/summaries/summary-2026-06-22_2026-06-28.html)). This 4.2× figure appears **only in the weekly summary**; I could not tie it to a primary issue, and it should not be confused with the 1.26–1.33× *wall-clock* speedups measured in #5596, which are a different quantity.

---

## 4. The active research line: variants, mostly negative

Kaiyue Wen ran a burst of Muon-variant prototypes in late June (all on qwen3-130m, 4959 steps, `eval/paloma/c4_en/bpb`, Muon control **1.1673**):

| Variant | Result |
|---|---|
| **Activation-aware Muon** [#6538](https://github.com/marin-community/marin/pull/6538) | **Negative.** *"the activation-whitening monotonically hurts"* — best real cell 1.1751; matches Muon only with whitening off. Cause proposed: inputs are RMSNorm outputs, already well-conditioned. |
| **Left-preconditioned Muon** [#6588](https://github.com/marin-community/marin/pull/6588) | **Negative as specified** — *"idea 4 does not beat Muon"* (outer form 1.2538–1.2665). |
| **Mudam** (Shampoo–Muon) [#6589](https://github.com/marin-community/marin/pull/6589) | Ported, **not re-validated in-tree**; the 8B result (1.0180 vs Muon 1.0320) is self-flagged as *"suggestive, but the baseline is a single run at a different LR/decay."* |
| **Curvature-corrected Muon** [#6634](https://github.com/marin-community/marin/pull/6634) | **Partial positive, open.** ~1.1652 vs same-compute MuonH 1.168 (~0.003 bpb), but explicitly not separated from the noise band. |

The most interesting output of this line was mechanistic rather than a win: #6588 proved offline that `inner_only` and Mudam are numerically the *same rule* (cosine 1.0000000000), and that the ~0.02 gap was pure solver fidelity — a coarse 5-step Newton–Schulz (cosine 0.80 to the exact polar) **trains better** than an exact `eigh` (cosine 0.9999), because it saturates the amplification of tiny eigenvalues (at eigenvalue 1e-6: 1000× exact vs 5.8× coarse). *A saturating, under-converged whitening regularizes better than exact whitening.*

Skeptical read on all of it: the ~1.165 figure replicates across three independent routes, but **the margin over Muon does not** — it is ~0.002–0.003 bpb at one scale, one seed per cell, and baselines shift ~0.002 across hardware (1.1661 on v5p-8 vs 1.1680 on v6e-8), the same order as the claimed win.

The most recent entry, [PR #7118](https://github.com/marin-community/marin/pull/7118) (Kaiyue Wen, July), adds two error-aware Muon feedback policies; Hessian-corrected Muon beat a fresh Muon control at four of five learning rates by a mean 0.000587 C4-en bpb — which the weekly summary itself labels *"a single observation, not yet a replicated optimizer win."*

## 5. One place Muon is a confound, not a choice

In RL post-training, Muon shows up as a **source of error**, not a result. Ashwinee Panda's cross-framework reproduction [#6915](https://github.com/marin-community/marin/issues/6915) carries an in-body self-correction (2026-07-03): *"this section originally said AdamW; the run's server config is `optimizer: muon`"*. Per the July 6–12 summary, a July 10 fidelity audit found the headline numbers were not a same-ruler comparison, and after closing the gaps the gains **remain confounded across five candidate causes — Muon vs AdamW among them — "none of which has isolated causal evidence yet."** There is no standing claim in this corpus that Muon beats AdamW in RL.

---

## Gaps and caveats

- **The GitHub half of the corpus ends 2026-07-05**, while Discord/W&B run to 2026-07-16 and weekly summaries to 2026-07-12. So #6811's comments, #6979, #6998, #7024, and #7118 are **summary-sourced only** — no primary backing here. Any GPU-Muon change discussed only in July GitHub issues would be invisible to me; my "Muon is still off on GPU" claim is solid for Discord through 07-16 and GitHub only through 07-05.
- The **4.2× effective-speedup frontier** figure is weekly-summary-only and I could not ground it in a primary source.
- The GatedNorm → muonh routing in the June 27 job summary is an **inference** from "2D weight matrices", corroborated by #5735/#5750 but not directly quoted there.
- Larry has reportedly opened the question of what to do with the cooldown checkpoints, ranging from checkpoint averaging to **swapping MuonH for Adam** — summary-sourced, and worth watching as the one live signal that the TPU default could itself change.

---
<!--provenance-->
> *Data: marinmirror (frozen eval corpus 2026-07-16) — 68,026 chunks, built 45h ago · summaries through 2026-07-06_2026-07-12 · GitHub content ends 2026-07-05, Discord/W&B through 2026-07-16. No refresh this run (disabled).*
>
> *Query: "what is our current muon approach, and how did we end up here?"*
>
> *Sub-queries: "MuonH current TPU recipe — parameter groups, May Recipe #5951 LR refit, Newton-Schulz replication fix" · "why Muon was shelved on GPU — roofline #6573, FSDP tax, OA sync #6693, grouped-bank #6493, PP #6532/#6534" · "Muon variant prototypes — #6538 activation-aware, #6588 left-preconditioned, #6589 Mudam, #6634 curvature-corrected, #7118 error-aware" · "how Marin adopted Muon — earliest experiments, MuonH vs AdamH ladder, effective-speedup frontier" · "Muon vs AdamW in RL — Ashwinee Panda fidelity audit, MarinSkyRL confounds" · "latest Muon state July 2026 — hero run #6044/#6704, cooldown #6811, #6979 GPU MFU path, jaxpp revival" · [verification pass] "resolve GatedNorm routing conflict; confirm 'Muon needs to go' quote; confirm TPU 0.9% step share; refute any GPU re-enablement; establish corpus end dates"*
