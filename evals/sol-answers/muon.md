# What is our current Muon approach, and how did we end up here?

## Short answer

Marin's selected training recipe is **MuonH for the ordinary matrix groups, with Adam retained for the parameter groups for which Muon geometry is inappropriate**. In the corrected Grug-MoE routing, token embeddings, routers and router biases, attention gates, and vector/scalar leaves stay on Adam; the LM head (`output_proj`) also stays off MuonH, while the matrix groups that the old recipe sent to AdamH/AdamH-expert move to MuonH ([#5596](https://github.com/marin-community/marin/issues/5596)). “H” matters: MuonH takes the Muon/Newton–Schulz direction and applies hyperball-style normalization; a direct GatedNorm comparison found the no-hyperball Muon route worse at all reported scales, including +0.0086 Paloma macro loss at d512 ([#5750](https://github.com/marin-community/marin/issues/5750)).

For MoE expert tensors, Marin also extended the implementation rather than pretending each expert stack was an ordinary 2-D leaf: the optimizer batches Newton–Schulz over stacked expert weights and restores the original sharding once at the optimizer boundary ([PR #3902](https://github.com/marin-community/marin/pull/3902)). On GPUs this remains a systems problem as well as an optimizer choice: the June H100 work grouped same-shaped 3-D expert leaves into a 4-D Newton–Schulz layout, but its handoff explicitly said the standalone update target was met while the production training path had not yet been proven fixed ([#6493](https://github.com/marin-community/marin/issues/6493)).

## How the choice was made

The decision was empirical and staged:

1. The April gate-1 program compared AdamH, Muon, and MuonH, then searched Muon learning rate/beta and batch-size variants at d512 and d768. The best plain-Muon Vizier point at d512 was 3.80732 Paloma macro loss versus the 3.81035 AdamH baseline—only a marginal gain—and the thread was ultimately closed with “we went with MuonH” ([#5167](https://github.com/marin-community/marin/issues/5167)).
2. The corrected May matrix-swap test held the Adam mask fixed and compared MuonH with the v16 AdamH baseline at matched budgets. MuonH improved Paloma at d512 (3.7542 vs 3.8104), d768 (3.3988 vs 3.4339), and d1024 (3.1357 vs 3.1605); the d1280 result made the overall gate-2 verdict “mixed,” so this was not evidence of uniform dominance at every scale ([#5596](https://github.com/marin-community/marin/issues/5596)). The same issue's fit put the projected curve crossing near 1.65e21 FLOPs; that is a **projection**, not a measured crossover.
3. Follow-up routing ablations corrected a crucial over-broad interpretation of “Muon everywhere”: embeddings, routing parameters, gates, vectors/scalars, and the LM head stayed on Adam-family updates, while eligible matrices used MuonH ([#5596](https://github.com/marin-community/marin/issues/5596)). GatedNorm-specific experiments favored keeping the hyperball form over a plain NS/Keller-scaled Muon update ([#5750](https://github.com/marin-community/marin/issues/5750)).
4. Productionization then had to address stacked experts and hardware. PR #3902 added batched orthogonalization for expert stacks; June's H100 investigation found that Newton–Schulz wants local 2-D matrices, which conflicts with FSDP-style matrix sharding, and explored grouped/packed master-state layouts rather than changing the chosen optimizer on quality grounds ([PR #3902](https://github.com/marin-community/marin/pull/3902), [#6493](https://github.com/marin-community/marin/issues/6493)).

## Current-state guardrails

- “Current Muon approach” means **selective MuonH**, not MuonH on every parameter.
- The quality evidence above is primarily TPU gate experiments; the H100 work is performance/representation evidence and should not be used to transfer TPU throughput numbers to GPUs.
- The frozen corpus supports selection of MuonH, but it does not support saying the grouped H100 production path is finished. #6493's latest handoff is explicitly more cautious.

**Verified sources / gap note.** Re-opened #5167, #5596, #5750, PR #3902, and #6493. All numeric claims above are measured values in those threads except the 1.65e21 crossing, labeled as a fit projection. The corpus does not provide a single final canonical config dump, so exact current learning rates and Newton–Schulz step count are omitted.

---
<!--provenance-->
> *Data: frozen marinmirror — 68026 chunks, built 2h ago · summaries through 2026-07-06_2026-07-12. No refresh (frozen runner only).*
>
> *Query: "what is our current muon approach, and how did we end up here?"*
>
> *Sub-queries: "current Muon approach / parameter routing" · "MuonH vs AdamH gates" · "Newton–Schulz and hyperball" · "stacked expert orthogonalization" · "learning-rate/batch ablations" · "GPU production status". HyDE vector retrieval was used for the recipe/history facet.*
