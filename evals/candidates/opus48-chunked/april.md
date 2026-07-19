# How we did on the April 2026 milestone

**Short version: we hit the thing the milestone was named for, and missed most of the things around it.**

The April 2026 milestone was **"Kick-off pre-trained 100B-A13B 1.2T token MoE (pregistered)"** — it appears verbatim as the milestone header on the weekly summaries for the weeks of [Apr 6](https://mws.oa.dev/summaries/summary-2026-04-06_2026-04-12.html), [Apr 13](https://mws.oa.dev/summaries/summary-2026-04-13_2026-04-19.html), and [Apr 26](https://mws.oa.dev/summaries/summary-2026-04-26_2026-05-02.html), and is named as the target milestone when the infra epics were filed into it ([#4256](https://github.com/marin-community/marin/issues/4256)). It succeeded "Kick-off a 32B-A4B 10T token MoE training run…" (week of [Mar 30](https://mws.oa.dev/summaries/summary-2026-03-30_2026-04-05.html)) and was replaced by "May: Prepping scaling training recipe + data mix + GPU training" (week of [May 4](https://mws.oa.dev/summaries/summary-2026-05-04_2026-05-10.html)).

The milestone was a **kick-off** goal, not a completion goal — and read that way, it landed.

---

## 1. The headline deliverable: hit

The run started **April 11**, inside the month. ClassicLarry on [#4697](https://github.com/marin-community/marin/issues/4697#issuecomment-4239090396) (2026-04-13):

> "I went ahead and started the run on Apr 11th on a v4-1024. Sometime this week we can migrate it to v4-2048 with improved ragged all-to-all setup for better MFU."

And it was genuinely **preregistered** — the prediction was registered off the isoflop fit in [#4447](https://github.com/marin-community/marin/issues/4447) *before* launch, and the epic [#4281](https://github.com/marin-community/marin/issues/4281) records the goal as met:

> "- [x] 1e23 (120B-A12B-ish) compute optimal launched + pre-registered"

**Two honest caveats on the headline:**

- **The shape doesn't match the milestone title.** The milestone says *100B-A13B, 1.2T tokens*; the run actually launched was **d5120, 129B total / ~16B active, ~1.01T tokens** at a 1e23 FLOP budget ([#4697](https://github.com/marin-community/marin/issues/4697)). ihodes flagged the discrepancy on 2026-04-17 — *"I think compute optimal = ~1T / 1e12 tokens here?"* — and ClassicLarry confirmed *"Yes."* ([#4697](https://github.com/marin-community/marin/issues/4697#issuecomment-4271187922)). The "100B-A13B" figure survives only in the MFU target issue [#4300](https://github.com/marin-community/marin/issues/4300), where dlwh admitted the numbers were *"Made these numbers up because i don't have any TPU v4 MOE intuition yet."*
- **The result is a May fact, not an April one.** The run did not finish in April. It completed **May 20**, and it beat its own preregistration — ClassicLarry: *"Final paloma/macro_loss was 2.234, 1% better than the initial prediction of 2.252"* ([#4697](https://github.com/marin-community/marin/issues/4697#issuecomment-4498921338)). That is a genuinely strong scaling-law result (extrapolated from runs 333× smaller), but crediting it to April would be wrong.

April itself ended badly for the run: per the [week of Apr 26 summary](https://mws.oa.dev/summaries/summary-2026-04-26_2026-05-02.html), three attempts *"together burned 319k chip-hours — 94.5% of the week's TPU spend and 96.5% of HW FLOPs — and all three crashed,"* bisected to `wandb.init(resume="allow")` on worker 0 ([#5319](https://github.com/marin-community/marin/issues/5319)).

---

## 2. Epic-by-epic scorecard

| Epic | Goal | End-of-April outcome |
|---|---|---|
| [#4281](https://github.com/marin-community/marin/issues/4281) MoE Scaling to April goal | 1e22 isoflop done; 1e23 launched+prereg; Good/Great 10T ablations | **Mostly hit** — 44/52 sub-issues; first two boxes checked, third not |
| [#4283](https://github.com/marin-community/marin/issues/4283) MoE MFU at scale | 25–30% MFU on TPU v4; H100 parity w/ Megatron | **Missed** — 16.4% TPU; no e2e GPU number. 0/6 sub-issues |
| [#4282](https://github.com/marin-community/marin/issues/4282) Agentify experimentation | Agents run a full architecture idea end-to-end, ~3 decisions | **Partially hit** — closed "for April" on a marginal win |
| [#4268](https://github.com/marin-community/marin/issues/4268) On-demand & reserved capacity | Reserved capacity in Iris | **Largely pre-satisfied**, no explicit close-out |
| [#4269](https://github.com/marin-community/marin/issues/4269) Off Ray completely | No Ray; cluster to 0; no `import ray` | **Hit on code, one live tail** |
| [#4270](https://github.com/marin-community/marin/issues/4270) Canary pass rate to 90%+ | Consistently >90% | **Missed** — volatile all month |
| [#4271](https://github.com/marin-community/marin/issues/4271) Marin-as-a-library | External project can `import marin` | **Substantively hit** |
| [#4272](https://github.com/marin-community/marin/issues/4272) Canonical data pipeline | download→norm→dedup/quality→tokenize | **Deferred** to mid-May |
| [#4273](https://github.com/marin-community/marin/issues/4273) Usability & Observability | Workqueue, observability | **Hit** |
| [#3100](https://github.com/marin-community/marin/issues/3100) Data sources | 20T high-quality tokens | **On track** — 18.1T |
| [#3192](https://github.com/marin-community/marin/issues/3192) Synthetic data | Mid/post-training data | **Missed** |

### What clearly worked

**Getting off Ray ([#4269](https://github.com/marin-community/marin/issues/4269)).** ihodes, 2026-05-03: *"Ray is functionally gone after the late-April delete spree (#5028, #5031, #5076, #5087, #5089, #5131, #5132, #5137, #5138, #5140 and friends)."* yonromai verified: *"Current `main` has no `import ray`, `from ray`, or `@ray.remote` under `lib/marin/`."* The one unmet criterion was "Ray cluster scaled down to 0" — *"`marin-big-run` is still the last Ray cluster because the 1e23 run is still using it"* — i.e. blocked by the milestone's own headline run. The [week of Apr 26 summary](https://mws.oa.dev/summaries/summary-2026-04-26_2026-05-02.html) notes *"The post-Ray-sunset shakeout dominated infra."*

**Observability ([#4273](https://github.com/marin-community/marin/issues/4273)).** Nominally the "lower priority / slack-time" workstream, and it fully delivered. ihodes, 2026-05-03: *"The April scope landed end-to-end — log/stats service split (#5212, #5290, #5370), endpoint proxy + IAP (#5336, #5349), BackgroundTracker (#5332), RPC stats redesign earlier in the month."*

**Marin-as-a-library ([#4271](https://github.com/marin-community/marin/issues/4271)).** rjpower, 2026-04-15: *"We now have 2 demo examples available via https://github.com/marin-community/marin-experiments/ which can be used as examples for writing experiments completely outside of the Marin repo, and using our nightly published artifacts."*

**Token supply ([#3100](https://github.com/marin-community/marin/issues/3100)).** Goal was 20T high-quality tokens. The [week of Apr 26](https://mws.oa.dev/summaries/summary-2026-04-26_2026-05-02.html) token panel shows **18.1T tokens (+69.8B), 33.3% synthetic, 107 datasets** — ~90% of target, ahead of schedule. Closed 2026-06-05: *"Substantially done, we're beyond 19T and have more than what we need for foreseeable future."*

**The 1e22 leg of MoE scaling.** dlwh closed the tracker on 2026-04-13 having identified the completed result thread [#3800](https://github.com/marin-community/marin/issues/3800): `moe-v7-1e22-d3200`, *"about 34.6B total / 4.7B active params, 326B tokens"* on **v4-512**.

### What missed

**MFU ([#4283](https://github.com/marin-community/marin/issues/4283)) — the clearest miss.** Target ([#4300](https://github.com/marin-community/marin/issues/4300)) was *"TPU v4: 25%-30% MFU sustained for 100B-A13B."* Achieved, all TPU v4 with an **active-FLOPs denominator** (structurally lower than dense MFU):

- **14.23% MFU**, 194,443 tok/s — v4-1024, EP4 ring dispatch, 2026-04-13 ([#4697](https://github.com/marin-community/marin/issues/4697#issuecomment-4239090890))
- **16.36% MFU**, 447,320 tok/s — v4-2048, EP8 `ragged_all_to_all`, late April ([#4697](https://github.com/marin-community/marin/issues/4697#issuecomment-4283224985))

That is roughly half to two-thirds of the target band. The epic ended **0/6 sub-issues closed**. #4300 was never closed with a number — dlwh closed it on 2026-05-31 with the single word *"good enough"* ([#4300](https://github.com/marin-community/marin/issues/4300#issuecomment-4586150714)), i.e. the target was relaxed rather than met.

The real April win here was **relative, not absolute**. Per the [week of Apr 13 summary](https://mws.oa.dev/summaries/summary-2026-04-13_2026-04-19.html), the run *"moved mid-flight from v4-1024 with ring dispatch to v4-2048 with ragged expert parallelism at ep=8, running at roughly 20% better MFU on twice the hardware while tracking train loss step-for-step against the ring baseline."* That "20%" is **relative** (14.2% → 16.4%), not 20 points and not 20% absolute. The same summary flags a caveat: *"Perplexity evals come out measurably worse at matched train loss."*

On **GPU**, the goal was throughput parity with Megatron/torchtitan. No end-to-end H100 MFU or tokens/s parity measurement was produced in April at all — only a layer-level microbenchmark (chloechiaw, 2026-04-13, CoreWeave 8×H100, [#4311](https://github.com/marin-community/marin/issues/4311#issuecomment-4235094540)): *"**30B anchor**: near parity. Grug ring EP is 6% slower than Megatron deepep… / **235B anchor**: 3.5x slower, need to take a closer look at why."*

**Canary pass rate ([#4270](https://github.com/marin-community/marin/issues/4270)) — missed.** Goal: *"canary ferry pass rate consistently above 90%."* Actual, from the weekly ferry panels:

| Week | TPU ferry | CoreWeave GPU ferry | Datakit smoke |
|---|---|---|---|
| [Apr 6–12](https://mws.oa.dev/summaries/summary-2026-04-06_2026-04-12.html) | 4✓ / 3✗ | 0✓ / 3✗ | 3✓ / 1✗ |
| [Apr 13–19](https://mws.oa.dev/summaries/summary-2026-04-13_2026-04-19.html) | 2✓ | 5✓ | 5✓ / 4✗ |
| [Apr 26–May 2](https://mws.oa.dev/summaries/summary-2026-04-26_2026-05-02.html) | 9✓ | 6✓ / 4✗ (60%) | 5✓ / 2✗ (71%) |

TPU finished strong (9-for-9), but "consistently above 90%" was not achieved on any lane across the month. The end-of-month GPU regression traced to the JAX 0.9.2 upgrade exposing an NCCL all-to-all crash on CoreWeave H100 ([#5377](https://github.com/marin-community/marin/issues/5377), fixed in [#5379](https://github.com/marin-community/marin/issues/5379)).

**Synthetic data ([#3192](https://github.com/marin-community/marin/issues/3192)) — missed on its headline items.** The one clean win was a real dose-response curve on Marin-8B: **0.0% → 3.3% → 4.0% → 5.3% SWE-bench Verified** at 10K/50K/100K trajectories ([#4898](https://github.com/marin-community/marin/issues/4898)). Against that:

- The 140B-token generation ([#4719](https://github.com/marin-community/marin/issues/4719)) ended at *"~6.50M rollouts and 64.1K PRs at the 100-rollout target, i.e. ~52% rollout coverage and ~51% PR coverage"* — deliberately cut short because *"us-east5 v6e-4 capacity dried up."* Quality was fine; capacity was the binding constraint.
- The **500K SFT point never trained**: *"the run has been listed as down for ~80 hours with `.executor_status=FAILED` at step-38 (init only)."*
- The TerminalCorpus midtrain ([#4760](https://github.com/marin-community/marin/issues/4760)) exposed a real capability gap once an EOS-token bug was fixed: *"Marin-32B SFT step-858 at 2/87 (2.3%) against Qwen3-32B SFT step-1000 at 18/86 (20.9%)… a real ~10× gap"* — described as *"a base-prior weakness that more SFT alone may not close."*

**The third MoE-scaling goal did not land.** [#4281](https://github.com/marin-community/marin/issues/4281)'s *"- [ ] Isoflop ablations on key decisions ('Good/Great 10T') up to 1e21 flops"* remains unchecked, and individual ablation PRs went stale — e.g. [#4068](https://github.com/marin-community/marin/issues/4068) (great 10T first-k-dense isoflop sweep) was *"automatically closed due to inactivity"* on 2026-04-24.

**Data mixture and dedup/quality were explicitly punted.** ihodes opened [#5360](https://github.com/marin-community/marin/issues/5360) to scope quality-and-dedup parameter selection *"against a mid-May launch,"* and Helw150 filed [#5359](https://github.com/marin-community/marin/issues/5359) with sub-issues *"all aimed at locking the mix by mid-June."* So raw token supply was solved in April; the production mixture for the big run was not.

### The ambiguous one: agentify

[#4282](https://github.com/marin-community/marin/issues/4282)'s goal was *"Agents can run full 'try out a new architecture idea' end to end from smoke → tuned isoflop up to 1e20. Prove this out on at least ~3 decisions."* ihodes closed it 2026-05-03: *"I think this is closeable for April,"* citing that *"The agent-driven ablation sweep produced its first cleanly Gate-2-promotable architecture change (AdamH-on-embed, #5184)… i.e. real architectural signal, not just process."*

Worth flagging: the weekly summary bills #5184 as the *"lone Gate-2 winner,"* but the actual close-out is more modest — ClassicLarry, 2026-04-29: *"**Marginal PASS gate 2.** Speedup > 1 at all scales but fading (1.05→1.01). Projections show only −0.003/−0.002 at 1e21/1e23"* ([#5184](https://github.com/marin-community/marin/issues/5184#issuecomment-4340036221)). Real, but small. The claim of "~3 decisions" proven out is not documented anywhere I could find.

---

## 3. Overall read

- **The named milestone was achieved**: the big MoE run was kicked off in April, preregistered, and ultimately validated the scaling extrapolation to within 1%.
- **The supporting infra epics were a mixed bag**: the Ray sunset, observability, and marin-as-a-library all landed; canary reliability and MFU both clearly missed their stated numeric bars.
- **Compute capacity was the recurring cause of misses** — TPU capacity killed the synthetic-data scale-out and the 500K SFT; MFU shortfalls made the headline run far more expensive than planned (319k chip-hours burned in one week on crashed attempts).
- **Bookkeeping overstates the misses.** Several epics showed 0/N sub-issues closed while being substantively delivered (#3100 especially); sub-issue counts are not a reliable completion signal for this milestone.

### Gaps in this answer

There is **no explicit April-milestone retrospective or post-mortem** in the corpus. The close-out was done epic-by-epic in early May, mostly in a single pass by ihodes on 2026-05-03. dlwh flagged *"a separate infra post-mortem to come"* on 2026-05-20, but no such document appears in the frozen corpus. Overall milestone completion percentages were also not found — only per-epic sub-issue counts, which are unreliable as noted. #4268 (on-demand & reserved capacity) has no follow-up discussion at all; it was audited as *"Mostly implemented in Iris"* at kickoff and appears to have needed little April work.

---
<!--provenance-->
> *Data: marinmirror — 86367 chunks, built 0h ago · summaries through 2026-07-06_2026-07-12. No refresh this run (frozen eval corpus).*
>
> *Query: "how did we do on our april 2026 milestone?"*
>
> *Sub-queries: "MoE Scaling up to April goal #4281 — 1e22/1e23 target vs achieved" · "Increase MFU epic #4283 — MFU targeted vs achieved, TPU vs H100" · "Agentify experimentation #4282 — end-to-end agent architecture loop" · "Infra April epics #4268–#4273 — per-epic measurable criteria vs outcome" · "April milestone close-out / retro — was the 100B-A13B run kicked off and preregistered" · "April data + synthetic data workstreams #3100/#3192 — target vs landed"*
