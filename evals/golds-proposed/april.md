# How did we do on our April 2026 milestone?

**Short version.** The April 2026 GitHub milestone — [*"Kick-off pre-trained 100B-A13B 1.2T token MoE (pregistered)"*](https://github.com/marin-community/marin/issues/4266) — spanned **three pillars: modeling, infra, and post-training / synthetic data.** Its central scientific bet **paid off** (the pre-registered 1e23 MoE run beat its predicted loss), and most infra epics were substantially delivered. But the milestone was **never formally declared complete or given a completion percentage** in the corpus; it was wound down issue-by-issue in two sweeps (a partial one on **2026-05-03** and a final one on **2026-06-05**), with the marquee run finishing in May and a whole post-training pillar coming in **mixed-to-negative**. Net: a real win on the headline modeling bet, solid-but-tailing-into-May infra delivery, two missed quantitative sub-targets (canary pass rate, TPU MFU — both soft), and a post-training quarter that produced one genuine positive signal amid three misses plus a durable learning.

---

## How the milestone was actually structured (this matters for "how did we do")

The two "April 2026" issues people reach for — **[Infra April 2026 (#4256)](https://github.com/marin-community/marin/issues/4256)** and **[Modeling April 2026 (#4266)](https://github.com/marin-community/marin/issues/4266)** — are **not** the scorecard parents. Both were split into epics, then **deliberately removed from the milestone and closed** so they wouldn't distort the count. ihodes on #4256: *"remove this issue from the milestone, so it doesn't pollute our completion count"* ([#4256](https://github.com/marin-community/marin/issues/4256)); #4266 was likewise split, un-parented (after some reparenting churn ihodes told the agent to *"undo that reparenting"*), and closed ([#4266](https://github.com/marin-community/marin/issues/4266)).

The live milestone therefore holds **the epics themselves plus three additional tracking issues** — roughly a 12-epic roster, not the 9 the split alone produces:

- **6 infra epics** ([#4268](https://github.com/marin-community/marin/issues/4268)–[#4273](https://github.com/marin-community/marin/issues/4273))
- **3 modeling epics** ([#4281](https://github.com/marin-community/marin/issues/4281), [#4282](https://github.com/marin-community/marin/issues/4282), [#4283](https://github.com/marin-community/marin/issues/4283))
- **[#4474](https://github.com/marin-community/marin/issues/4474)** Levanter Store / K8s Logging / Infra Improvements
- **[#3192](https://github.com/marin-community/marin/issues/3192)** Synthetic data (research + critical path for post-training)
- **[#3100](https://github.com/marin-community/marin/issues/3100)** Data sources for pre-training / mid-training

(Note: the frozen corpus stores issue/comment *text* only, not GitHub `state`/`closed_at`/`milestone` fields, so every "closed / open" call below is **inferred from close-out comments and the milestone-scoped weekly summaries**, not from metadata.)

---

## Pillar 1 — Modeling ([#4266](https://github.com/marin-community/marin/issues/4266) split)

### The headline deliverable: the pre-registered 1e23 (compute-optimal ~130B-A16B MoE) run — *scientific win, completion slipped into May (but completion was only a STRETCH goal)*

The committed vs. stretch framing is explicit in the #4266 body:
- **Committed:** *"clean 'best recipe' MoE isoflop up to 1e22 completed, **1e23 launched**"*
- **Stretch (verbatim):** *"**Stretch: 1e23 completed (~16 days on v4-2048)**"*

So the milestone's committed modeling deliverable was **launch + pre-register**, not finish-by-April-30. Against that:

- **Committed goal met on time.** ClassicLarry started the run **Apr 11** on a **v4-1024, EP4 ring dispatch** ([#4697](https://github.com/marin-community/marin/issues/4697)), pre-registered against the isoflop fit in [#4447](https://github.com/marin-community/marin/issues/4447). Both boxes in #4281 are checked: *"[x] Clean 'best recipe' MoE isoflop up to 1e22 completed"* and *"[x] 1e23 … compute optimal launched + pre-registered"* ([#4281](https://github.com/marin-community/marin/issues/4281)).
- **Did not finish inside April.** As of the week of Apr 26, three attempts of `moe_1e23_d5120_bs2048_ep8_ragged_48l` on **1024 v4 chips** had burned **319k chip-hours — 94.5% of the week's TPU spend and 96.5% of HW FLOPs — and all three crashed** ([summary 2026-04-26](https://mws.oa.dev/summaries/summary-2026-04-26_2026-05-02.html)). The recurring crash was bisected to `wandb.init(resume="allow")` on worker 0 creating host-state divergence that surfaced as a TPU launch-id mismatch one save-step after restart; the validated fix was fresh-id-per-attempt with `resume="never"` ([#5319](https://github.com/marin-community/marin/issues/5319)).
- **Stretch goal ultimately achieved — in May.** After migrating to a **v4-2048, EP8 ragged all-to-all** setup in `us-central2` ([#4697](https://github.com/marin-community/marin/issues/4697)), the run completed. On **2026-05-20** ClassicLarry reported **final `paloma/macro_loss = 2.234, "1% better than the initial prediction of 2.252"**; dlwh sealed it 2026-05-21 (final W&B run `…resume112662_clip15_20260518_123236`, report `Marin-1e23-MoE`, checkpoint `gs://marin-us-central2/…/step-120332`).

**On the "beat" — name the registered number and qualify it.** The **pre-registered bet was 2.252**, specifically the leave-future-out isoflop fit with **irreducible loss L∞ fixed at 1.6** ([#4447 c4225593039](https://github.com/marin-community/marin/issues/4447#issuecomment-4225593039)); dlwh's public announcement calls exactly this number "preregistered." The achieved **2.234 beats 2.252 by ~1% (0.018)** — a well-calibrated result (the same machinery predicted 1e21 at 2.598 vs. an actual 2.599). But the beat is **asymptote-assumption-dependent, not robust across every forecast on record.** ClassicLarry's own sensitivity table shows 2.234 lands exactly on the **L∞=1.5 row**, and the *free-asymptote best fit* in the same issue was **2.224–2.226 (L∞≈1.447), which 2.234 does not beat** ([#4447 c4222453538](https://github.com/marin-community/marin/issues/4447#issuecomment-4222453538)). Other forecasts coexisted (Helw150's VPNLS re-forecast **2.295**, registered for Percy's ICLR talk and self-described as conservative; the AdamH-embed projection **2.251** baseline). Honest phrasing: *the run beat its pre-registered conservative estimate and landed mid-range of its own forecast envelope.*

### MoE Scaling ([#4281](https://github.com/marin-community/marin/issues/4281)) — mostly met
1e22 completed (`moe-v7-1e22-d3200` on **v4-512**: ~34.6B total / 4.7B active, 326B tokens, hidden 3200, 32 layers, 64 experts K=4; [#3800](https://github.com/marin-community/marin/issues/3800)); 1e23 launched + pre-registered. The third box — *"Isoflop ablations on key decisions ('Good/Great 10T') up to 1e21"* — **stayed unchecked** ([#4281](https://github.com/marin-community/marin/issues/4281)). No explicit close-out comment in corpus.

### Agentify experimentation ([#4282](https://github.com/marin-community/marin/issues/4282)) — closeable for April; note the goal was ~3 *full* agent-run decisions
The stated goal was *"Agents can run full 'try out a new architecture idea' end to end from smoke → tuned isoflop up to 1e20. **Prove this out on at least ~3 decisions** … coordinate this entirely through GitHub"* ([#4282](https://github.com/marin-community/marin/issues/4282)). ihodes marked it *"closeable for April"* on 2026-05-03, leaning on the **first cleanly Gate-2-promotable agent-driven architecture change — AdamH-on-embed ([#5184](https://github.com/marin-community/marin/issues/5184))**, which passed Gate 2 across all four scales with a 1e23 paloma projection of **2.249 vs 2.251 baseline** and closed the long-running embed-norm-growth investigation ([#4569](https://github.com/marin-community/marin/issues/4569)) — *"real architectural signal, not just process."* Supporting context: an MCP babysitter, the `agent-research` skill, and a new "foraging" skill to seed agent research with prior work ([#5117](https://github.com/marin-community/marin/issues/5117)). **Caveat:** the close-out leaned on that one promotable change plus a stack of negative results; whether ~3 *full* end-to-end agent-run decisions were completed is not demonstrated in-corpus, and hammer's response was to ask ihodes to turn it into a blog post rather than confirm the numeric bar. Treat as *"morally done / closeable,"* not a clean 3-of-3.

### MoE MFU at scale ([#4283](https://github.com/marin-community/marin/issues/4283)) — mixed; TPU MFU below target but signed off "good enough"
On **H100** the JAX 0.9.2 upgrade unlocked a full **Triton fwd+bwd MoE path measuring 1.91× faster than XLA on H100×8** ([#5330](https://github.com/marin-community/marin/issues/5330)). On **TPU v4**, the target was *"25%–30% MFU sustained for 100B-A13B on v4-1024"* ([#4300](https://github.com/marin-community/marin/issues/4300)); the best documented sustained figure was **~16.4% MFU** on the crashed `resume45207_clip15` v4-1024 intermediate (487B tokens; [summary 2026-04-26](https://mws.oa.dev/summaries/summary-2026-04-26_2026-05-02.html)). **Important framing:** dlwh filed the 25–30% band as numbers he *"Made … up because i don't have any TPU v4 MOE intuition yet,"* aimed only at keeping the run *"not too wasteful,"* and **closed #4300 as "good enough" on 2026-05-31** ([#4300](https://github.com/marin-community/marin/issues/4300)). So this is an **accepted-below-target / owner-signed-off** outcome, not a hard failure — and the run it was protecting hit its loss target. (The corpus does not report a separate sustained MFU for the *completed* v4-2048 run; the 16.4% figure belongs to a crashed v4-1024 attempt.)

---

## Pillar 2 — Infra ([#4256](https://github.com/marin-community/marin/issues/4256) split, + [#4474](https://github.com/marin-community/marin/issues/4474))

- **Single way of running jobs — off Ray ([#4269](https://github.com/marin-community/marin/issues/4269)) — met.** ihodes 2026-05-03: *"Ray is functionally gone after the late-April delete spree"* (#5028, #5031, #5076, #5087, #5089, #5131 …); yonromai 2026-05-04: *"Ray is code-complete from Marin's side, with one live-infra tail."* Post-sunset work rolled to the May infra tune-up ([#5369](https://github.com/marin-community/marin/issues/5369)).
- **Improve Usability & Observability ([#4273](https://github.com/marin-community/marin/issues/4273)) — April scope met; closed 2026-06-05.** ihodes 2026-05-03: *"The April scope landed end-to-end — log/stats service split (#5212, #5290, #5370), endpoint proxy + IAP (#5336, #5349), BackgroundTracker (#5332)."* Formally *"Closed, as this was a catch-all"* on 2026-06-05.
- **On-demand & reserved capacity ([#4268](https://github.com/marin-community/marin/issues/4268)) — substantially delivered (no explicit close-out comment).** Goal: *"support reserved capacity."* Concrete evidence it was addressed: the 2026-03-30 codebase audit found *"Iris has a full reservation system,"* and **PR [#4379](https://github.com/marin-community/marin/pull/4379) landed the `CapacityType` enum (PREEMPTIBLE / ON_DEMAND / RESERVED) with the GCP queued-resource API for reserved TPUs on 2026-04-03**, followed by [#4662](https://github.com/marin-community/marin/pull/4662) expanding the v4-reserved pool to full topology sizes. No close-out text exists, so formal sign-off can't be pinned — but the capability shipped.
- **Canonical data pipeline ([#4272](https://github.com/marin-community/marin/issues/4272)) — largely done / rolled forward.** A composable `DataPipeline` exists, sources migrated, dedup module landed; heavy source-expansion and Zephyr-perf work continued into May. No close-out comment; state indeterminate/rolled-forward.
- **Marin-as-a-library ([#4271](https://github.com/marin-community/marin/issues/4271)) — partial.** By 2026-04-15 rjpower had *"2 demo examples … good enough for a 'getting-started'"* via a separate `marin-experiments` repo importing nightly artifacts — external `import marin` demonstrated, but not a full pip-installable curated API. No explicit close.
- **Canary pass rate to 90%+ ([#4270](https://github.com/marin-community/marin/issues/4270)) — not met on the GPU side within the window.** Target = rolling pass rate ≥90%. Week of Apr 26: **TPU ferry 9/9 (100%)**, but the **GPU canary regressed to 60%** after JAX 0.9.2 exposed an NCCL all-to-all crash on CoreWeave H100 ([#5377](https://github.com/marin-community/marin/issues/5377), fix [#5379](https://github.com/marin-community/marin/issues/5379)) and a `pl.load`/`pl.store` breakage ([#5341](https://github.com/marin-community/marin/issues/5341), fix #5347), and the **datakit smoke ferry slipped to ~71%** ([summary 2026-04-26](https://mws.oa.dev/summaries/summary-2026-04-26_2026-05-02.html)). No close-out; the clearest quantitative miss at end of April.
- **Levanter Store / K8s Logging / Infra Improvements ([#4474](https://github.com/marin-community/marin/issues/4474)) — closed "substantially accomplished" 2026-06-05.** ihodes: *"fair to close as substantially accomplished?"* → rjpower: *"Yeah sounds good."* (K8s logging #4477/#4478 done; Levanter store #4445, checkpoint RAM #4475, Nemo v2 #4476 tracked.)

---

## Pillar 3 — Post-training / synthetic data ([#3192](https://github.com/marin-community/marin/issues/3192)) — the pillar most easily missed; **mixed-to-negative**

#3192 was the top-level epic for the post-training pillar (*"critical path for post-training"*). ihodes scored its April scope on 2026-05-03 but did **not** close it; it drifted until 2026-06-05, when Helw150 said *"We should close and split up into something with good definitions of done"* — an admission it never had crisp definitions of done. The weekly rollup scored it **0/4 sub-issues closed**.

- **WIN — SWE-bench dose-response on Marin-8B ([#4898](https://github.com/marin-community/marin/issues/4898)).** SFT of `marin-community/marin-8b-base` on SWE-ZERO (execution-free) trajectories, evaluated on **SWE-bench Verified** (100 tasks, mini-swe-agent v1 harness, vLLM on v6e-4, 3 runs/model): **0.0% baseline → 3.3% (10K) → 4.0% (50K) → 5.3% ± 1.5% (100K SFT)**. This is the one genuine positive signal: SWE-ZERO crossed from *"the pipeline works"* to a **measurable downstream capability curve**, generalizing from diverse non-Python code-edit trajectories to Python SWE-bench. (An earlier single-run "100K regresses to 2%" readout was retracted; with 3 runs 100K is highest, though it shows a `!!!`-repetition degeneration tied to an unemitted `<|eot_id|>` stop token.)
- **MISS (stuck) — the 500K SFT dose-response point ([#4898](https://github.com/marin-community/marin/issues/4898)).** The next dose point never produced a real training step: the launcher iterated v5→v10 fighting wrong working dirs, Levanter cross-region rejections, and v5p-16 capacity that was 0 cluster-wide, then hung on a 3-hour dataset-cache build and ended `.executor_status=FAILED` at **step-38 (init only, out of 125,000 target), listed down ~80 hours.** It was never resurrected (SFT later "held per user direction"). Permanent gap.
- **MISS (~half target) — 140B-token SWE-ZERO scale-out ([#4719](https://github.com/marin-community/marin/issues/4719)).** Target: 12.3M rollouts (122,910 PRs × 100) ≈ 140B tokens. Achieved **~6.5M rollouts / ~52% rollout coverage, ~51% PR coverage** before a user-authorized cutoff, when us-east5 v6e-4 capacity collapsed. Real engineering wins inside the miss: a measured **3.4× per-worker throughput** improvement (`--enable-prefix-caching` + reduced MAX_TURNS) and a rebuild of the 13-batch pipeline into a multi-region swarm with atomic GCS shard claims (avoiding a historical $1.5K/day cross-region-write incident). A false "crossed 140B" (inflated by error-rollout byte counting) was caught and reverted to ~52%. Datasets `AlienKevin/SWE-ZERO-12M-trajectories` shipped.
- **MISS (negative result) — Marin-32B TerminalCorpus midtrain ([#4760](https://github.com/marin-community/marin/issues/4760)).** SFT `marin-32b-base` on 15% of Nemotron-Terminal-Corpus, eval on Terminal-Bench 2.0 vs Qwen3-32B at comparable budget: **Marin-32B 2/87 (~2.3%) vs Qwen3-32B 18/86 (~20.9%) — roughly 10× worse.** Salvaged a **reusable EOS-config infra fix** (a saved-checkpoint `eos_token_id` mismatch that caused degenerate `zorazora…` output; OLE rate dropped 70%→11% after patching), but even EOS-corrected the gap held. Formally closed as a negative result 2026-06-10.
- **Durable learning — the gap points at base pre-training, not SFT volume.** Trace-diagnostic BPB work found **marin-8b patch_gain = −1.43** (trace context *hurts* patch prediction) vs. **+0.16 for qwen3-8b / +0.18 for llama3.1-8b** — a base-model agentic/coding-prior deficit ([#4963](https://github.com/marin-community/marin/issues/4963), dlwh). (This −1.43 is an **8B** diagnostic that *explains* the 32B agentic gap; it is not a 32B metric.)

---

## Data sources ([#3100](https://github.com/marin-community/marin/issues/3100)) — met

Target: **20T high-quality tokens** for the large MoE runs. By 2026-04-06 the catalog stood at **17.0T tokens across 100 datasets**; ihodes closed it 2026-06-05 as *"Substantially done, we're beyond 19T and have more than what we need for the foreseeable future."*

---

## Scorecard

| Milestone element | Target | Outcome |
|---|---|---|
| **1e23 MoE — launch + pre-register** (#4697, #4281) | Committed: launch + pre-register | **Hit** — launched Apr 11 (v4-1024 EP4 ring), pre-registered vs #4447 |
| **1e23 MoE — completion** (#4697) | *Stretch:* "1e23 completed (~16 days on v4-2048)" | **Slipped past April; achieved in May** — finished on v4-2048 EP8 ragged, sealed 2026-05-20/21 |
| **1e23 loss vs pre-registered bet** | 2.252 (L∞=1.6 isoflop fit, #4447) | **Beat by ~1%** (2.234) vs the conservative registered number; *not* robust vs the 2.224–2.226 free-fit |
| **1e22 isoflop** (#4281, #3800) | Clean best-recipe 1e22 completed | **Hit** (`moe-v7-1e22-d3200`, v4-512) |
| **Isoflop ablations to 1e21** (#4281) | "Good/Great 10T" up to 1e21 | **Not checked** (open) |
| **Agentify experimentation** (#4282, #5184) | ~3 full agent-run decisions, smoke→1e20, via GitHub | **Closeable for April** — 1 clean Gate-2 promotion (AdamH-embed) + many negative results; 3-of-3 not demonstrated |
| **TPU MFU** (#4300) | 25–30% sustained ("made up" band) | **Below target (~16.4% on crashed v4-1024), but owner closed "good enough" 2026-05-31** |
| **H100 MoE MFU** (#4283, #5330) | Fast MoE on H100 | **Progress** — Triton fwd+bwd 1.91× vs XLA on H100×8 |
| **Off Ray** (#4269) | Ray gone | **Hit** (functionally gone, code-complete May 3–4) |
| **Usability & observability** (#4273) | April scope | **Hit** (April scope; closed 2026-06-05) |
| **On-demand / reserved capacity** (#4268, #4379) | Support reserved capacity | **Substantially delivered** (CapacityType enum + queued-resource API); no explicit close-out |
| **Canonical data pipeline** (#4272) | Composable download→tokenize | **Largely done / rolled forward** |
| **Marin-as-a-library** (#4271) | External `import marin` | **Partial** (external demos, not full pip-install) |
| **Canary ≥90%** (#4270) | Both fleets rolling ≥90% | **Missed on GPU** — TPU 9/9 (100%), GPU ~60%, datakit ~71% |
| **Levanter store / K8s logging** (#4474) | Infra improvements | **Hit** ("substantially accomplished," 2026-06-05) |
| **Post-training / synthetic data** (#3192) | Critical path for post-training | **Mixed-to-negative** — 1 win (SWE-bench 0→5.3%), 3 misses (500K SFT stuck, 140B ~52%, Marin-32B ~10× gap); no crisp DoD |
| **Data sources** (#3100) | 20T high-quality tokens | **Hit** (>19T) |

---

## Bottom line

- **Won:** the pre-registered 1e23 scaling bet **beat its forecast** (2.234 vs the registered 2.252 paloma macro) — a well-calibrated result from runs 333× smaller; the off-Ray migration completed; usability/observability and Levanter-store/K8s-logging landed; reserved-capacity support shipped; the data-sources target (>19T) was met; and agentic experimentation produced a real promotable architecture change (AdamH-embed).
- **Slipped / missed:** the marquee run **didn't finish inside April** (crash-bound on v4-1024; completed on v4-2048 in May) — though completion was only a *stretch* goal; **canary ≥90% was not achieved on GPU** (~60% after JAX 0.9.2 fallout); **TPU MFU landed ~16.4% vs a made-up 25–30% band** (owner-signed-off "good enough"); and the **post-training pillar came in mixed-to-negative** (140B generation ~half-done, 500K SFT stuck ~80h, Marin-32B ~10× behind Qwen3-32B).
- **Key learning:** the Marin-32B agentic underperformance is a **base-pretraining prior deficit** (patch_gain −1.43 on Marin-8B vs +0.16 on Qwen3), not just SFT volume — a finding that redirects post-training work upstream into pre-training.
- **Process note:** the milestone was **never formally declared complete and no completion percentage exists** in the corpus. It was wound down in two issue-by-issue sweeps (2026-05-03 and 2026-06-05) and superseded by June work; the two "April 2026" tracking issues (#4256/#4266) were deliberately excluded from the count. So the honest characterization is *"substantially delivered with a real headline win, delivered with a tail into May–June rather than cleanly by April 30,"* not *"100% complete."*

---
<!--provenance-->
> *Gold reference answer. Source: marinmirror frozen eval corpus (`evals/corpus/2026-07-16`) + milestone-scoped weekly summaries via `mws.oa.dev`. All issue/PR numbers, close-out quotes, and load-bearing numbers verified against the frozen corpus with `mum-frozen show`/`search` on 2026-07-16.*
>
> *Corpus caveat: the freeze stores issue/comment text only — not GitHub `state`/`closed_at`/`milestone` metadata — so all open/closed determinations are inferred from in-thread close-out comments and the milestone-scoped weekly summaries, and close dates are approximate. #4268/#4270/#4272 have no close-out comment in corpus; the "Good ≥5% (SWE-bench)" success bar cited elsewhere is not present verbatim in the frozen #4898 text and is omitted here.*
