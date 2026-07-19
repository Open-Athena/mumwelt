# Marin's Muon approach: where we are, and how we got here

## The short version

Marin's production optimizer is **MuonH** — Newton–Schulz-orthogonalized momentum wrapped in a Frobenius "hyperball" step-size rule — applied to the weight-matrix and GatedNorm parameter groups, with everything else (token embeddings, routers, router biases, attention gates, 1-D norms) left on plain Adam and the `lm_head`/`output_proj` left on AdamH. That configuration landed on main as part of the May Recipe in [PR #6153](https://github.com/marin-community/marin/pull/6153) (2026-06-04) and is what the 67B-A2B hero run is training under today.

The single most important fact about the current state is that **Muon is now a hardware-split decision**: TPU keeps it, GPU dropped it. On TPU the Newton–Schulz work is ~1% of step time and effectively free; on H100 it was measured eating ~33–70% of a step depending on config, and the June 26 OA meeting shelved it on GPU. The June hero run was consequently staged onto TPU rather than the CoreWeave H100 cluster.

---

## 1. What MuonH actually is

The variant naming is set out explicitly in the GatedNorm-routing study [#5750](https://github.com/marin-community/marin/issues/5750#issuecomment-4456961403):

| variant | update geometry | per-step Frobenius norm |
|---|---|---|
| `adam` | plain Adam at small `adam_lr` | LR-scaled by Adam |
| `adamh` | Frobenius hyperball at `learning_rate` | `lr · ‖W‖_F` |
| **`muonh`** | **Newton–Schulz + Frobenius hyperball** | `lr · ‖W‖_F` |
| `muon` | NS + Keller post-scale `√max(1, fan_out/fan_in)`, no hyperball | `lr · √fan_out` |

Both halves are load-bearing: dropping the hyperball wrap (`gn → muon`) regressed at every scale (+0.0086 / +0.0054 / +0.0014 paloma at d512/768/1024) — [#5750](https://github.com/marin-community/marin/issues/5750#issuecomment-4465767626). Implementation is `GrugMoeMuonHConfig` in `experiments/grug/moe/optimizer.py`, registered as `grug_moe_muonh_v1`.

**Parameter routing**, verbatim from [PR #6153](https://github.com/marin-community/marin/pull/6153):

> "**MuonH optimizer** (`GrugMoeMuonHConfig`, registered as `grug_moe_muonh_v1`) replaces AdamH on the weight-matrix + GatedNorm group."

with the surrounding recipe settings from the same PR: **1% warmup**, **no gradient clipping** (`max_grad_norm = None`), router z-loss and final-logit z-loss both disabled.

**Learning rate.** MuonH originally inherited AdamH's LR through a fixed `13/3` ratio that had been tuned for AdamH geometry and never re-tuned ([#5621](https://github.com/marin-community/marin/issues/5621), [PR #5622](https://github.com/marin-community/marin/pull/5622)). The refit is [#5951](https://github.com/marin-community/marin/issues/5951) (~150 runs over d512–d1280 × six token-per-param budgets × five LR multipliers). Worth knowing that **the first posted fit was wrong** — a units bug set `intermediate_dim = d*4` where the heuristic uses `≈ d/2`, inflating token counts ~4× and producing a spurious "1.6–1.7× LR bump" ([retraction](https://github.com/marin-community/marin/issues/5951#issuecomment-4549326686)). The corrected law, carried into #6153, is:

```
muonh_lr = 18.31 · tokens^-0.395 · dim^-0.150 · √B      (R² = 0.996, 17 cells)
```

At compute-optimal points this is *slightly below* the old LR (0.98× / 0.93× / 0.90× / 0.88× at d512/768/1024/1280); the apparent bump was purely the units artifact. Runs launched under the buggy LR were killed and resubmitted.

---

## 2. The hardware split — the defining feature of the current approach

### TPU: Muon is nearly free

The decisive measurement is a TPU profile posted into the GPU issue — [#6493](https://github.com/marin-community/marin/issues/6493#issuecomment-4787096060), on **v4-1024, d2560, EP=1, BS=4096, seq 8192**:

| phase | wall time | % of step |
|---|---|---|
| Forward | ~7.4 s | ~33% |
| Backward | ~14.8 s | ~66% |
| **Optimizer NS + bookkeeping** | **~206 ms** | **~0.9%** |

Against a ~22.4 s step at 21.4% MFU, the whole optimizer phase is rounding error. End-to-end this shows up as a wash in throughput: AdamH 1.51M tok/s vs MuonH ~1.47–1.49M tok/s on v4-1024 at 32M batch ([Discord, 2026-06-23/24](https://discord.com/channels/1354881461060243556/1518073966030487734/1519259765627289681)), with Larry's own caveat that his first reading of that metric "was, in fact, not accurate" before he re-measured. His conclusion: "TPU Muon performance seems good, but replicating on GPU may not be feasible with same approach."

*Caveat worth stating:* that 0.9% figure is **v4-1024, pre-fix era**, where NS distributed cleanly at 13 matrices per chip. After the sharding fix (below), NS is *replicated* over the replica axis on v4-2048, which adds redundant compute. No post-fix v4-2048 NS-share-of-step breakdown appears in the corpus — treat 0.9% as v4-1024-specific.

### GPU: Muon was shelved

The A/B on **8×H100, single node, Grug MoE d2560 / L26 / E=256 / top-4 / seq4096 / B8, EP8** ([#6493](https://github.com/marin-community/marin/issues/6493#issuecomment-4738899640)):

| run | optimizer | MFU | tok/s | s/step |
|---|---|---|---|---|
| May140 | MuonH5 (real NS) | **4.75%** | 21,962 | 1.492 |
| May141 | SGD | **16.19%** | 74,848 | 0.438 |

≈11.4 MFU points, ~1.05 s of a 1.49 s step attributable to the Muon path at that config. A separate 2-node profiling dashboard put it at **~500 ms of a 1500 ms step** ([#6573](https://github.com/marin-community/marin/issues/6573#issuecomment-4775876844)), and rjpower's independent PP harness (6.1B, P=8, batch 64) measured a Muon tax of **+0.76 s single-node (−43%)** and **+0.92 s at 2 nodes (−60%)** — growing with node count ([#6532](https://github.com/marin-community/marin/issues/6532#issuecomment-4794455018)). These are three different shapes; don't conflate the percentages.

**The mechanism is comms, not FLOPs.** dlwh's roofline writeup ([#6493](https://github.com/marin-community/marin/issues/6493#issuecomment-4760616407), 2026-06-21) is the definitive account: NS wants whole matrices resident on one device, so the natural layout groups over the expert-stack axis — which is hostile to the FSDP weight layout. Converting grouped updates back costs an expert-weight-sized payload (**121.875 GiB bf16 globally** at L=26, E=256, D=2560, I=1280), lands *after* backward so it can't be overlapped, and does not shrink with node count:

> "Muon NS compute itself will scale away, but the comms are large and do not scale away with node count."

The measured harness (~0.370 s at R4) sits essentially at its own compute+comms floor (~0.400 s) — "the problem is not a huge inefficiency there; the problem is that the floor itself is large."

**Every attempt to bridge that boundary failed.** [#6493](https://github.com/marin-community/marin/issues/6493) states it plainly: "unbounded grouping OOMed during autotune, bounded grouped/padded MuonH did not improve full training throughput, and harness runs show the isolated NS/dot kernels are already reasonably efficient." Bounded grouped/padded MuonH (May143) hit 21,595 tok/s — indistinguishable from ungrouped May140's 21,962. Direct `jax.sharding.reshard`, packed slice-first gather-apply, and target-FSDP reshard were each tried and each rejected; all four packed variants lowered to the ideal 2 all-gathers but GPU compilation re-expanded them ([comment](https://github.com/marin-community/marin/issues/6493#issuecomment-4756837084)).

**The decision**, dlwh reporting the OA meeting outcome, 2026-06-26 ([Discord](https://discord.com/channels/1354881461060243556/1399998407657001062/1520200189690773564)):

> "We're currently at ~19.9 MFU on 4 nodes w/ SGD/Adam. Some low hanging fruit remains. **Muon needs to go for now. 10% step-count improvement isn't worth atm. We'll need much faster interconnect or PP.** Targeting a fancy pallas kernel for moe mlp."

This is encoded in the tracking issue [#6693](https://github.com/marin-community/marin/issues/6693), whose workstream table carries a row: *"Stabilize grouped-bank Muon harness | MuonH | **SKIP** | #6493 harness | [no-op for now] | ... | **No muon**"* (verified directly against the issue body).

Note the "10%" is a **sample-efficiency / step-count** figure, not throughput. The original bar Larry set was a *hope* for "at least a 10–15% compute efficiency gain" ([Discord, 2026-04-28](https://discord.com/channels/1354881461060243556/1365044508546568372/1498818986887090236)) — a target, not a result.

### The escape hatch that didn't open

Pipeline parallelism was the named fix, and it demonstrably *does* solve the Muon tax. On 8×H100, 6.1B, P=8 ([#6532](https://github.com/marin-community/marin/issues/6532#issuecomment-4792205591)): FSDP pays +0.76 s for Muon, PP-1f1b pays +0.09 s, and **zero-bubble PP hides Newton–Schulz entirely (~0.00 s)**. But PP's *base* throughput is 2–3.7× worse, so FSDP still wins net on NVLink. rjpower parked the work on 2026-06-25 — best PP ≈ 22.9k tok/s vs FSDP ≈ 145k tok/s (~0.78× single-host), only pulling ahead (~1.22×) once FSDP's all-gather crosses the slower data-center interconnect at v6e-32. PR #6534 was closed unmerged and moved to an external repo ([close-out](https://github.com/marin-community/marin/issues/6532#issuecomment-4802268924)). A notable diagnosis: PP's ceiling was **Python dispatch, not silicon** — 3085 ms of a 3140 ms wavefront was single-thread enqueue.

---

## 3. How we got here — the chronology

**March 2026 — trackers filed, nothing run.** dlwh split the question into [#4033 "Muon perf on MoE"](https://github.com/marin-community/marin/issues/4033) and [#4034 "Muon loss on MoE"](https://github.com/marin-community/marin/issues/4034). The starting state was explicit: Muon + MoE "has not been tried in this codebase"; `MuonConfig`/`MuonHConfig` existed but were proven only on dense LLaMA/Qwen. Both trackers were eventually closed by ClassicLarry on 2026-06-06 with the one-word verdict **"Integrated MuonH"**.

Relevant prior art on the Adam side: [#4024](https://github.com/marin-community/marin/issues/4024) concluded "do not blanket-promote AdamH across the MoE recipe; keep router on Adam" — which is why every later Muon mask leaves the router on plain Adam.

**April — plain Muon on MoE is a wash.** [#5115](https://github.com/marin-community/marin/issues/5115) (Muon with AOL Newton–Schulz coefficients) and [#5134](https://github.com/marin-community/marin/issues/5134) (MuonH with 2× batch) both came out flat. Pranshu Chaturvedi, 2026-04-25: "Looks like there isn't any real difference between AdamH baseline, Muon AOL, MuonH base batch, and MuonH 2x batch" ([Discord](https://discord.com/channels/1354881461060243556/1365044508546568372/1497412143144894584)). The Vizier LR/beta search [#5167](https://github.com/marin-community/marin/issues/5167#issuecomment-4333247150) was worse than flat — d512 best Muon 3.80732 vs AdamH 3.81035 (−0.003), but d768 3.46759 vs 3.43387 (**+0.034, worse**), concluding "current gate-1 evidence does not support promoting Muon over AdamH broadly."

**Early May — localizing the failure.** [#5585](https://github.com/marin-community/marin/issues/5585#issuecomment-4411229059) rebuilt every architectural feature one knob at a time on a dense nanogpt baseline. Larry's finding: "the full 30% Muon gain in distribution holds all the way through to our recipe… However, Muon does absolutely terrible on OOD data, mostly code." Quantified: paloma `dolma_100_programing_languages` −0.25 nats and `github_python` −0.33 nats in AdamH's favor, with Muon winning 0.01–0.04 on wikitext/arxiv/c4_en. On Discord: github/cpp Muon 2.55 vs AdamH 2.2.

**2026-05-09/10 — the unlock.** [#5596](https://github.com/marin-community/marin/issues/5596) / [PR #5597](https://github.com/marin-community/marin/pull/5597) (Kaiyue Wen). The fix versus every earlier attempt was the **"baseline-adam-mask"**: every parameter the AdamH baseline routed to Adam *stays* on Adam with identical hyperparameters, only the AdamH matrix groups switch to MuonH, and `lm_head` stays AdamH. Kaiyue: "swap every AdamH with MuonH except lm head… roughly the speedup is 20-30% step-wise compared to AdamH with no regression on token per second" ([Discord](https://discord.com/channels/1354881461060243556/1365044508546568372/1502824423936491550)). Note this also resolved #4033's perf question *positively* — MuonH was faster in tok/s, not slower.

**Gate-2 results ([#5596](https://github.com/marin-community/marin/issues/5596#issuecomment-4422416900)), and this is where measured and projected must be kept apart:**

| scale | AdamH baseline | MuonH | wall-clock speedup |
|---|---|---|---|
| d512 (2.19e17) | 3.8104 | 3.7542 | 1.33 |
| d768 (1.70e18) | 3.4339 | 3.3988 | 1.26 |
| d1024 (9.00e18) | 3.1605 | 3.1357 | 1.22 |
| d1280 (2.83e19) | 3.0065 | 2.9888 | 1.19 |

Speedup > 1 at all four **measured** scales — but the fitted scaling exponent was *shallower* (α = 0.0906 vs baseline 0.0941), so the **projections** had the curves crossing at C ≈ 1.65e21 and MuonH landing **0.0106 worse at 1e23**, a strict gate-2 failure. The fit rested on four points.

**Mid-May — the sweep that fixed the projection.** [#5619](https://github.com/marin-community/marin/issues/5619) removed warmup (the heuristic had used 10%) and was the single biggest lever: d512 3.7404, d768 3.3834 → 1.43 / 1.38 wall-clock vs AdamH, +7–9% over MuonH-with-warmup, at unchanged tok/s. Kaiyue two days later: "**After removing warmup, we now predict a small but positive win at 1e23 comparing MuonH and AdamH**" ([Discord](https://discord.com/channels/1354881461060243556/1365044508546568372/1504268569947410512)) — that is the moment the gate-2 failure cleared.

Other arms in the same fortnight: GatedNorm → MuonH won its 4-way bake-off ([#5750](https://github.com/marin-community/marin/issues/5750)); K/V → AdamH was **rejected** (d512 +0.0140 regression — [#5719](https://github.com/marin-community/marin/issues/5719#issuecomment-4462192491)); NorMuonH was essentially tied with MuonH ([#5598](https://github.com/marin-community/marin/issues/5598)); `muon_epsilon` proved insensitive across 1e-6→1e-16 ([#5933](https://github.com/marin-community/marin/issues/5933)); per-expert LR scaling by √sparsity mostly failed gate 1 ([#5799](https://github.com/marin-community/marin/issues/5799)).

**2026-05-17 — adoption, on borderline evidence.** [#5763](https://github.com/marin-community/marin/issues/5763#issuecomment-4471778083) (`may_arch` + GN→MuonH + 1% warmup + no grad-clip) posted d512 3.6427 (−0.0170, 1.078×), d768 3.3040 (+0.0018, **0.991×**), d1024 3.0610 (+0.0036, **0.966×**). It was adopted anyway — reasoning: the d512 lead is large and reproducible, d768/d1024 deltas are inside run-to-run noise, and the recipe is simpler. Those three numbers became the comparison anchors for the whole downstream stack. This is worth flagging honestly: **the canonical baseline was a borderline gate-1 pass**, not a clean win.

**2026-05-30 → 06-04 — scaling laws and the merge.** [#6074](https://github.com/marin-community/marin/issues/6074) refit isoFLOP laws for the MuonH May Recipe; [PR #6153](https://github.com/marin-community/marin/pull/6153) merged it onto main with the fitted law `loss(C) = 1.6 + 88.90 · C^-0.0941` — same exponent as the v16 baseline, **~2.12× equal-throughput compute-equivalent speedup at every budget**. Two caveats on that headline number: it is a **fit-based projection**, and the largest *measured* budget anywhere in the thread is 3e19. Larry himself: "I am not trusting the 1e18 data point as much. It seems like Muon does disproportionately well at the very small size with only 1pct warmup."

**Late June — the GPU verdict** (§2 above), and the TPU staging of the hero run.

---

## 4. Muon in the live hero run

The 67.1B-total / 2.01B-active Grug MoE is training on **TPU v4-2048 under MuonH**, tracked in [#6044](https://github.com/marin-community/marin/issues/6044). Two Muon-specific things from that run are worth carrying forward.

**A silent Newton–Schulz sharding bug nearly derailed the launch.** Larry's root-cause, 2026-06-27 ([#6044](https://github.com/marin-community/marin/issues/6044#issuecomment-4819379417)):

> "The root cause of the issue is that the sharded NS calc in Muon was silently dropping when num_matrices didn't evenly divide into num_chips. This was originally written on v4-1024 where 256*26 is divisible by 512 chips. However it isn't divisible by 1024 chips."

256 experts × 26 layers = 6,656 matrices; 6656/512 = 13 exactly, but 6656/1024 = 6.5, so orthogonalization was silently skipped on the larger slice. The tell was diagnostic: "Lack of orthogonalization explains why the loss curve started better and then got worse over time, similar to how adam can beat muon in first couple steps." Larry had initially written the divergence up as a numerical-precision problem and retracted it — forcing fp32 masked the symptom without fixing it. A downstream consequence: the apparent instability at 67M-token batch size **was also this bug**, not a real critical-batch-size limit. The fix replicates the NS calc over the replica axis and makes the silent failure error loudly. Larry's own retro notes that agent-written resharding code reached production without human review under perceived hardware-idle pressure.

**Hardware choice, measured** ([#6044](https://github.com/marin-community/marin/issues/6044#issuecomment-4812848295), d2560 / seq8192 / MuonH):

| hardware | tokens/batch | EP | replica | MFU | tok/s |
|---|---|---|---|---|---|
| v4-1024 | 33.55M | 1 | 1 | **21.54%** | 1.49M |
| v4-2048 | 67.11M | 1 | 8 | 18.62% | **2.57M** |
| v4-2048 | 33.55M | 2 | 1 | 10.13% | 1.40M |

The team deliberately accepted **lower per-chip MFU on v4-2048 for absolute throughput** (~45 days projected vs ~78 on v4-1024). As of the 2026-07-06/12 week the run was near step 39,000 (~2.114T tokens, ~21% of the 10.07T horizon) holding MFU near 18.6% ([weekly summary](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html)).

---

## 5. What's open, and what's only exploratory

**Swapping MuonH → Adam for cooldown is an open question, not a plan.** After the first intermediate cooldown [#6811](https://github.com/marin-community/marin/issues/6811) completed, Larry posted to the midtraining group, 2026-07-11: *"We used MuonH for training. How much does swapping to Adam hurt/help? How much does maintaining the optimizer state help?"* ([Discord](https://discord.com/channels/1354881461060243556/1525374850053705760/1525377288139575356)), alongside questions about averaging the eight cooldown checkpoints and relaxing the hyperball norm constraint at a later stage. **That thread has no replies in the corpus** — no ablation launched, no decision. Do not read it as scheduled work.

**Muon variant research continues, but nothing has cleared the bar.** Kaiyue Wen ran a series of Muon variants on the qwen3-130m speedrun harness through June, all against a plain-Muon control:

- **Activation-aware Muon** ([#6538](https://github.com/marin-community/marin/pull/6538)) — "performs on par with plain Muon at qwen3-130m (no improvement, no harm)", after a self-correction where the first-reported metric was the wrong one.
- **Gain-gated Muon** ([#6506](https://github.com/marin-community/marin/pull/6506)) — SVD interpolation between Muon and square-root/HJB.
- **Curvature-corrected Muon** ([PR #6634](https://github.com/marin-community/marin/pull/6634)) — the most interesting result. Its [close-out](https://github.com/marin-community/marin/issues/6634#issuecomment-4796074066): "the curvature penalty *by itself* is loss-neutral vs MuonH, but a Mudam `q_k(P^{-1/2})` warm start + curvature refinement beats the same-compute MuonH baseline by ~0.003 bpb" (best 1.1652 vs a same-hardware MuonH control of 1.168). The writeup is careful about the baseline: the submission's 1.1661 was on v5p-8 and is *not* the fair anchor for a v6e-8 comparison. Iterating the fixed point further (K=3) made things *worse*. This is qwen3-130m, single-seed, ~0.003 bpb — a promising signal, not a promotion.

**A caveat on three issue numbers.** The 2026-07-06/12 weekly summary describes a GPU Upskilling Master Plan (#6998), a GPU MFU Learning Path (#6979) — including an itemized H100 MFU progression where "real Muon's Newton-Schulz orthogonalization" is called out as the single biggest give-back at ~3.7 points — and a speedrun PR adding two error-aware Muon feedback policies (#7118) where Hessian-corrected Muon reportedly beat a Muon control at four of five learning rates by a mean 0.000587 C4-en bpb, explicitly flagged as "a single observation, not yet a replicated optimizer win." **None of #6998, #6979, or #7118 are retrievable in this corpus** — I confirmed each returns `not found`. They exist only as summary narrative here, so I am reporting them as second-hand and unverified against primary sources. Source: [weekly summary 2026-07-06_2026-07-12](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html).

**Is Muon coming back on GPU?** Directionally there is one signal and no number. dlwh, 2026-07-10: "codex is seemingly making good progress on jaxpp [#7024] . **Hopper is back on the menu I think**" ([Discord](https://discord.com/channels/1354881461060243556/1399998407657001062/1525217063830949888)). But the ~40 jaxpp H100 runs from 2026-07-10/11 carry `adam_lr` in their configs — **Muon is still off in the PP runs at the freeze** — and Larry's explicit Muon-with-NS vs Muon-no-NS A/B on H100 from 2026-07-06 (`liger-d2560-moe-e256-muon` / `-muon-nons`) all show `state=failed` with no MFU recorded, as does the later `grug-gqa-muonprof2` profiling attempt on 2026-07-15. The #6693 SKIP row has not been revised.

---

## 6. Summary of the reasoning, in one paragraph

Muon looked flat on MoE for a month because it was being applied with the wrong parameter mask and the wrong learning rate; once Kaiyue Wen pinned every non-matrix parameter back to the AdamH baseline's Adam routing and left `lm_head` on AdamH, it produced a 1.19–1.33× wall-clock win across four measured scales, and removing warmup flipped the extrapolated 1e23 verdict from negative to positive. That configuration — MuonH, 1% warmup, no grad clipping, refit LR — merged as the May Recipe and is training the hero run. Then hardware bifurcated the story: the Newton–Schulz orthogonalization's *compute* shards away fine, but the layout conversion back to FSDP costs an expert-weight-sized, unhideable all-gather that doesn't shrink with node count, which is tolerable on TPU's ICI (~1% of step) and ruinous on H100 (~33–70%). Pipeline parallelism provably erases the tax but loses more elsewhere, so GPU shelved Muon and went after fused MoE kernels instead. The honest caveats: the 2.12× compute-equivalent speedup is a fit, the largest measured budget is 3e19, and the canonical May baseline was a borderline gate-1 pass that was adopted for simplicity as much as for its numbers.

---
<!--provenance-->
> *Data: marinmirror — 68026 chunks, built 48h ago · summaries through 2026-07-06_2026-07-12. Frozen eval corpus; no refresh this run.*
>
> *Query: "what is our current muon approach, and how did we end up here?"*
>
> *Sub-queries: "MuonH in the May Recipe — optimizer configuration, parameter-group routing, LR refit" · "Muon Newton–Schulz overhead on H100 under FSDP: roofline, grouped-NS negative result, the shelve decision" · "MuonH on TPU for the 67B-A2B hero run: NS sharding bug, MFU, why TPU absorbs the cost" · "history of Muon/MuonH/AdamH ablations on MoE and what the sweeps concluded" · "July GPU MFU work: Muon's cost in the H100 progression, PP as the fix, is Muon being reconsidered" · "error-aware / activation-aware / curvature-corrected Muon variant research (#7118 and the June optim line)"*
