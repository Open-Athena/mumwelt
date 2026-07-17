# How the April 2026 milestone went

The honest verdict is **mixed, with strong infrastructure and agentification progress, a successful launch of the headline MoE experiment, and important synthetic-data validation—but much of the science was still in flight at the April boundary**. Later May closeouts validate several bets, but they should not be backdated into April.

## Modeling: launch achieved; completion and much of the ablation program carried over

April's modeling plan called for a clean MoE isoflop through 1e22 FLOPs, launch and preregistration of a compute-optimal 1e23 MoE, key-decision ablations through 1e21, and—only as a stretch—the 1e23 run's completion [#4266](https://github.com/marin-community/marin/issues/4266), [#4281](https://github.com/marin-community/marin/issues/4281). The first two core goals were met: the 1e22 result existed, and the 1e23 experiment launched on April 11 as a d5120, 129B-total/16B-active MoE over roughly 1T tokens, initially on TPU v4-1024 with ring dispatch and a preregistered Paloma projection around 2.25 [#4281](https://github.com/marin-community/marin/issues/4281), [#4697](https://github.com/marin-community/marin/issues/4697).

The large run was **not complete in April**. It migrated to v4-2048 with EP8 ragged all-to-all and worked through launch/debugging issues; an April 24 alternative forecast of 2.295 was also posted, illustrating that forecasts were still being refined [#4697](https://github.com/marin-community/marin/issues/4697). The eventual 2.234 Paloma macro-loss was reported May 20, about 1% better than the initial 2.252 prediction—but that is a later validation of the April launch, not an April milestone result [#4697](https://github.com/marin-community/marin/issues/4697). The April tracker still showed the “Good/Great 10T” key-decision ablations incomplete [#4281](https://github.com/marin-community/marin/issues/4281).

## Agentification: a credible success

The agentification goal was ambitious: an agent should carry an architecture idea from smoke test through tuned isoflop experiments, coordinated through GitHub, with reproducible notes and babysitting support [#4282](https://github.com/marin-community/marin/issues/4282). The May 3 closeout explicitly judged the April scope closeable: agent-driven sweeps produced a Gate-2-promotable architecture change (AdamH on embeddings), closed a long-running norm-growth investigation, and demonstrated MCP-based run babysitting. That closeout is retrospective evidence about April work, so credit here is appropriate [#4282](https://github.com/marin-community/marin/issues/4282).

## Synthetic agent data: downstream signal established, scale-out unfinished

April moved SWE-ZERO from “the generator runs” to a measurable downstream curve. The retrospective records Marin-8B SWE-bench Verified improving from a 0% baseline to **3.3%, 4.0%, and 5.3%** with 10K, 50K, and 100K trajectory training sets [#3192](https://github.com/marin-community/marin/issues/3192), with the validation design tracked in [#4898](https://github.com/marin-community/marin/issues/4898). That supports the quality of the data direction.

But the 140B-token generation target was far from finished during April: on April 20 it stood at about **2.19M of 12.29M rollouts and 25.0B of 140B tokens (17.9%)** [#4719](https://github.com/marin-community/marin/issues/4719). A supposed “10B” checkpoint was corrected to **536,148 clean trajectories and about 6.1B tokens** after removing error rollouts, a useful warning to prefer corrected counters over labels [#4719](https://github.com/marin-community/marin/issues/4719). Thus April proved both throughput improvements and downstream utility, but generation itself carried into May.

## Infrastructure: substantial end-to-end delivery, with some explicit gaps

The infrastructure program included reserved capacity, eliminating Ray, canary reliability, Marin-as-a-library, a canonical data pipeline, and usability/observability [#4256](https://github.com/marin-community/marin/issues/4256). The clearest successes were:

- Ray was functionally removed by the late-April deletion series, with only a live-infrastructure tail remaining [#4269](https://github.com/marin-community/marin/issues/4269).
- External-library use became concrete through two `marin-experiments` examples using nightly artifacts [#4271](https://github.com/marin-community/marin/issues/4271).
- The observability/usability scope landed end-to-end: split log/stats services, endpoint proxy and IAP, BackgroundTracker, and the RPC stats redesign [#4273](https://github.com/marin-community/marin/issues/4273).

The frozen primary threads do not establish that every infra criterion closed. In particular, the 90%+ canary tracker and canonical download→normalize→dedup/quality→tokenize pipeline remain framed as goals rather than evidenced April closeouts [#4270](https://github.com/marin-community/marin/issues/4270), [#4272](https://github.com/marin-community/marin/issues/4272). The MoE-MFU tracker likewise set H100 and TPU performance objectives without a clean April closeout in the thread [#4283](https://github.com/marin-community/marin/issues/4283).

## Bottom line

April did well on **launching and operationalizing**: the 1e23 MoE was preregistered and running, agent-led experimentation became real, SWE-ZERO acquired a downstream validation curve, Ray removal and observability landed. It did less well on **finishing the scientific program inside the month**: the 1e23 result, the 140B synthetic-data build, and several ablation/reliability/data-pipeline objectives carried over. Calling it either a clean miss or a complete success would flatten that distinction.

### Source/gap notes

Numbers are labeled by their observation date. The May 20 MoE final loss is included only as a later outcome of the April-launched run, not as April achievement. Sparse tracker threads prevent a defensible all-epic completion percentage.

---
<!--provenance-->
> *Data: marinmirror frozen 2026-07-16 — 68,026 chunks · summaries through 2026-07-06_2026-07-12 (no refresh).*
>
> *Query: “how did we do on our april 2026 milestone?”*
>
> *Sub-queries: “April milestone workstream inventory” · “1e23 MoE launch, preregistration, and April boundary” · “MoE ablations and performance” · “agentification closeout” · “SWE-ZERO scale and quality validation” · “Ray/Iris/library/observability infrastructure” · “canary and canonical-data-pipeline gaps” · “skeptic pass on late closeouts and corrected counters”*
