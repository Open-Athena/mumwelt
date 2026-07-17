# Our Muon approach, and how we got here

## Short answer

Our production optimizer is **MuonH** — a *heuristic-scaled* Muon that applies an AdamH-style width/token/batch learning-rate scaling law to Muon's orthogonalized update. It is the optimizer in the **"May Recipe"** that landed on `main` in [#6153](https://github.com/marin-community/marin/pull/6153) (2026-06-04) and is what the July/August **67B-A2B Grug MoE hero run** trains with on TPU. MuonH replaces AdamH on the matrix-shaped ("weight-matrix") parameter group; embeddings, the lm-head, routers, gates, and vector/scalar leaves stay on Adam/AdamH. It is a **TPU decision** — Muon is expensive enough on GPU that the hero run deliberately runs on TPU v4-2048, not H100.

---

## What the current approach actually is

- **MuonH = Muon + a per-group LR heuristic.** The optimizer thread that produced it set out to "define a provisional MuonH scaling heuristic analogous to the existing AdamH heuristic" and shipped a `MoeMuonHHeuristic` config ([#5167](https://github.com/marin-community/marin/issues/5167)). Under the hood it is still standard Muon: momentum then **Newton-Schulz orthogonalization** of the 2-D update ([#4034](https://github.com/marin-community/marin/issues/4034) notes the `MuonConfig`/`MuonHConfig`/`GrugMuonConfig` lineage, all first proven on dense LLaMA/Qwen).

- **Parameter routing.** MuonH is applied to *every matrix-shaped parameter except the lm-head* (`output_proj` stays AdamH); token embeddings, routers, router biases, attention gates, and all vector/scalar leaves remain on Adam/AdamH ([#5596](https://github.com/marin-community/marin/issues/5596)). Two refinements were tuned on top: **GatedNorm gains are routed to AdamH** (the "GN to AdamH" stack, [#5645](https://github.com/marin-community/marin/issues/5645), [#5719](https://github.com/marin-community/marin/issues/5719)), and an attempt to also move attention **K,V to AdamH regressed** (-0.014 loss at d512), so attention matrices stay on MuonH ([#5719](https://github.com/marin-community/marin/issues/5719)). Expert `w_gate`/`w_up` are stored split for orthogonalization ([#5167](https://github.com/marin-community/marin/issues/5167), [#6153](https://github.com/marin-community/marin/pull/6153)).

- **Its own LR law.** MuonH gets a dedicated LR scaling law of the form `LR = m * tokens^(-a) * dim^(-b) * batch^0.5`, refit for the May Recipe in [#5951](https://github.com/marin-community/marin/issues/5951) (new fit `lr_coeff` 0.0255 to **0.097**, `lr_tokens_exp` -0.281 to **-0.395**, R^2=0.996). This heuristic feeds the July hero-run architecture ([#6701](https://github.com/marin-community/marin/issues/6701)) and loss preregistration (#6702) so follow-ups launch at the heuristic-optimal LR instead of re-sweeping.

- **What it buys.** The MuonH matrix swap passed both gates: at d512 it cut Paloma macro loss to **3.7542 vs the v16 AdamH baseline 3.8104 (-0.056)** at roughly equal throughput ([#5596](https://github.com/marin-community/marin/issues/5596)). The full May Recipe (MuonH plus the other architecture deltas) is credited with a **~2.12x equal-throughput compute-equivalent speedup** over the v16 baseline at every isoFLOP budget ([#6153](https://github.com/marin-community/marin/pull/6153)) — note this is the *whole recipe*, not Muon alone.

## How we ended up here (timeline)

1. **March — question posed.** "Is Muon a net quality win, loss, or wash on the MoE path?" Muon had only been proven on dense models; MoE was untried ([#4034](https://github.com/marin-community/marin/issues/4034), dlwh, 2026-03-23).
2. **April — bake-off.** An AdamH / Muon / MuonH gate-1 comparison led to a follow-up study to (a) define the MuonH heuristic, (b) A/B against AdamH-2x, and (c) run a Vizier Muon LR/beta search ([#5167](https://github.com/marin-community/marin/issues/5167)). Verdict, 2026-06-06: *"Closing, we went with MuonH here."*
3. **May — swap validated and tuned.** MuonH replacing AdamH on matrices passed gate-1 and gate-2 ([#5596](https://github.com/marin-community/marin/issues/5596) / PR [#5597](https://github.com/marin-community/marin/pull/5597), which also swept NorMuonH). A cluster of ablations then settled *which* non-matrix leaves to route where (GatedNorms to AdamH, K,V stay on Muon, etc. — [#5645](https://github.com/marin-community/marin/issues/5645), [#5719](https://github.com/marin-community/marin/issues/5719), #5735).
4. **Late May / June — LR refit + integration.** The MuonH LR law was refit for the May Recipe ([#5951](https://github.com/marin-community/marin/issues/5951)), and the recipe — "MuonH replacing AdamH on the weight-matrix group" plus half-RoPE, PKO, split gate/up, renormalized routing, no grad clipping, 1% warmup — was merged to `main` in [#6153](https://github.com/marin-community/marin/pull/6153) (2026-06-04). [#4034](https://github.com/marin-community/marin/issues/4034) was closed "Integrated MuonH" (2026-06-06).
5. **June/July — into the hero run.** The recipe carries into the 67B-A2B / 10T-token Grug MoE hero run ([#6044](https://github.com/marin-community/marin/issues/6044), tracked to land on TPU in [#6704](https://github.com/marin-community/marin/issues/6704)); run names carry the `_muon_` tag.

## The GPU caveat (why TPU)

Muon is costly on GPU. On a single H100 node, the **MuonH path ran at ~4.75% MFU / 21,962 tok/s / 1.492 s per step, versus ~16.19% MFU / 74,848 tok/s for the same recipe with SGD** — because Newton-Schulz runs **one matrix at a time**; a grouped/stacked Newton-Schulz prototype was built to amortize it ([#6493](https://github.com/marin-community/marin/issues/6493), [#6492](https://github.com/marin-community/marin/issues/6492)). More generally, "Muon carries high overhead at high sparsity," and on the Hopper MFU build-up the Newton-Schulz orthogonalization is the single biggest MFU give-back (~3.7 points) ([#6979](https://github.com/marin-community/marin/issues/6979), per the 2026-07-06 weekly summary). Combined with the high-sparsity Adam HBM tax and a slow MoE block, this is why the June/July run **landed on TPU rather than H100** ([#6704](https://github.com/marin-community/marin/issues/6704)).

## Frontier / not adopted

These were explored around MuonH but are **not** in the current recipe:

- **MuonEq** (lightweight equilibration before orthogonalization): gate-1 **FAIL** at both exponents, closed ([#6066](https://github.com/marin-community/marin/issues/6066)).
- **Left-preconditioned Muon / Mudam** (Shampoo-Muon; Ali Jadbabaie "idea 4"): Mudam ~1.165 vs a Muon control 1.167 c4-en bpb at qwen3-130m — a small edge; the `inner_only` variant reproduces it. Research-stage only ([#6588](https://github.com/marin-community/marin/issues/6588)).
- **Error-aware / Hessian-corrected Muon** speedrun (Kaiyue Wen): beat a fresh Muon control at 4 of 5 LRs by a mean 0.000587 bpb — explicitly flagged as "a single observation, not yet a replicated optimizer win" ([#7118](https://github.com/marin-community/marin/issues/7118)).
- **Open question (latest week):** for the intermediate cooldown checkpoints, Larry has raised **swapping MuonH for Adam** as one option (alongside checkpoint averaging) — a live discussion, not a decision ([#6811](https://github.com/marin-community/marin/issues/6811), per the 2026-07-06 summary).

## Bottom line

The current approach is **MuonH — heuristic-LR-scaled Muon with Newton-Schulz orthogonalization, applied to the matrix parameter group (minus lm-head), with GatedNorm gains kept on AdamH** — shipped in the May Recipe ([#6153](https://github.com/marin-community/marin/pull/6153)) and running the TPU hero run. We got there by a March-to-June bake-off (AdamH to Muon to MuonH, [#5167](https://github.com/marin-community/marin/issues/5167) / [#5596](https://github.com/marin-community/marin/issues/5596)), a per-recipe LR refit ([#5951](https://github.com/marin-community/marin/issues/5951)), and a hardware verdict that Muon's orthogonalization overhead makes TPU the home for the big run ([#6493](https://github.com/marin-community/marin/issues/6493), [#6704](https://github.com/marin-community/marin/issues/6704)). Newer variants (MuonEq, Mudam/left-preconditioned, Hessian-corrected) remain research, not production.

---
<!--provenance-->
> *Data: marinmirror — 68026 chunks, built 1h ago · summaries through 2026-07-06_2026-07-12 (frozen 2026-07-16 eval corpus; no refresh).*
>
> *Query: "what is our current muon approach, and how did we end up here?"*
>
> *Sub-queries: "MuonH replacing AdamH on the weight-matrix group (current recipe)" · "what MuonH is / how it differs from Muon and Adam" · "MuonEq equilibration before Newton-Schulz orthogonalization" · "Muon shelved on GPU / kept on TPU decision" · "tuning LR on the MuonH recipe (#5951)" · "Muon-on-MoE decision history / why chosen" · "NorMuonH / MuonEq / left-preconditioned Muon / Hessian-corrected Muon variants" · "May Recipe merge and July hero-run architecture"*
