# How did we do on our April 2026 milestone?

**Short version:** The April 2026 milestone — [*"Kick-off pre-trained 100B-A13B 1.2T token MoE (pre-registered)"*](https://github.com/marin-community/marin/issues/4266) — was **substantially delivered on the infra and modeling-process fronts**, and its headline scientific bet **paid off** (the pre-registered run beat its predicted loss). But the marquee run **did not cross the finish line inside April** — it was crash-bound through late April and only completed on **2026-05-20** — and **two quantitative targets were missed** (canary pass rate and TPU MFU). Most epics were closed "for the April milestone" in early May rather than by April 30.

The milestone was split from two parent tracking issues, [Infra April 2026 (#4256)](https://github.com/marin-community/marin/issues/4256) and [Modeling April 2026 (#4266)](https://github.com/marin-community/marin/issues/4266), into a set of epics. Here is how each landed.

## The headline deliverable: the pre-registered 1e23 (~100B-A13B-class) MoE run

The honest verdict is **"scientific win, schedule slip."**

- The run ([#4697](https://github.com/marin-community/marin/issues/4697), `moe_1e23_d5120_bs2048_ep8_ragged_48l`) was **kicked off and pre-registered in April**, against a target of **2.25 paloma macro** from the isoflop fit in [#4447](https://github.com/marin-community/marin/issues/4447). The milestone's stated goal ("1e23 compute-optimal launched + pre-registered") was checked off in [#4281](https://github.com/marin-community/marin/issues/4281).
- **It did not finish in April.** As of the week of Apr 26, three attempts had burned **319k chip-hours — 94.5% of the week's TPU spend and 96.5% of HW FLOPs — and all three crashed** ([summary 2026-04-26](https://mws.oa.dev/summaries/summary-2026-04-26_2026-05-02.html)). The recurring crash was bisected to `wandb.init(resume="allow")` on worker 0 causing a TPU launch-id mismatch one save-step after restart ([#5319](https://github.com/marin-community/marin/issues/5319)); the validated fix was fresh-id-per-attempt with `resume="never"`.
- **It ultimately succeeded — three weeks late.** On **2026-05-20**, ClassicLarry reported the run's **final paloma/macro_loss was 2.234, "1% better than the initial prediction of 2.252"** ([#4697](https://github.com/marin-community/marin/issues/4697)). So the pre-registered bet was **beaten**, validating the scaling-law forecast — but the result landed in May, outside the April window.

## Modeling epics (split from [#4266](https://github.com/marin-community/marin/issues/4266))

- **MoE Scaling ([#4281](https://github.com/marin-community/marin/issues/4281)) — mostly met.** The clean "best recipe" MoE isoflop **up to 1e22 completed** (`moe-v7-1e22-d3200`, paloma macro **2.432**, [#3800](https://github.com/marin-community/marin/issues/3800)) and the 1e23 was launched + pre-registered; both boxes checked. The "isoflop ablations up to 1e21 ('Good/Great 10T')" box stayed partly open. Closed 2026-04-13.
- **Agentify experimentation ([#4282](https://github.com/marin-community/marin/issues/4282)) — met / "morally done."** Closed for April by ihodes (2026-05-03): the agent-driven ablation sweep produced its **first cleanly Gate-2-promotable architecture change — AdamH-on-embed ([#5184](https://github.com/marin-community/marin/issues/5184))**, passing Gate 2 across all four scales and closing the long-running embed-norm-growth investigation [#4569](https://github.com/marin-community/marin/issues/4569) — "real architectural signal, not just process." The MCP babysitter ([#5042](https://github.com/marin-community/marin/issues/5042), [#5071](https://github.com/marin-community/marin/issues/5071)) shipped and a design-doc workflow was established.
- **MoE MFU at scale ([#4283](https://github.com/marin-community/marin/issues/4283)) — mixed; the TPU MFU target was missed.** On H100 the JAX 0.9.2 upgrade unlocked a full **Triton fwd+bwd MoE path 1.91× faster than XLA on H100×8** ([#5330](https://github.com/marin-community/marin/issues/5330)). But the **TPU v4 target of "25–30% MFU sustained for 100B-A13B" ([#4300](https://github.com/marin-community/marin/issues/4300)) was not hit** — the actual 1e23 run sustained only **~16.4% MFU** ([summary 2026-04-26](https://mws.oa.dev/summaries/summary-2026-04-26_2026-05-02.html)). Worth noting dlwh filed that 25–30% figure as numbers he "made up… because i don't have any TPU v4 MOE intuition yet" ([#4300](https://github.com/marin-community/marin/issues/4300)) to keep the run "not unnecessarily wasteful," so this is a soft target rather than a hard commitment.

## Infra epics (split from [#4256](https://github.com/marin-community/marin/issues/4256))

- **Single way of running jobs — off Ray ([#4269](https://github.com/marin-community/marin/issues/4269)) — met.** "Ray is functionally gone after the late-April delete spree" ([#5028](https://github.com/marin-community/marin/issues/5028), [#5031](https://github.com/marin-community/marin/issues/5031), and friends); yonromai confirmed it "code-complete from Marin's side" with one live-infra tail. Closed for April.
- **Improve Usability & Observability ([#4273](https://github.com/marin-community/marin/issues/4273)) — met (April scope).** "The April scope landed end-to-end — log/stats service split ([#5212](https://github.com/marin-community/marin/issues/5212), [#5290](https://github.com/marin-community/marin/issues/5290)), endpoint proxy + IAP, BackgroundTracker" (ihodes, 2026-05-03). Two of the autogenerated completion boxes (30-day historical utilization, agent-e2e) stayed unchecked; the May continuation moved under [#5369](https://github.com/marin-community/marin/issues/5369).
- **Canonical data pipeline ([#4272](https://github.com/marin-community/marin/issues/4272)) — largely done.** A composable `DataPipeline` exists, ≥3 sources were migrated, and a standalone dedup module landed (boxes checked). Remaining boxes — the `add-dataset` skill using the template, ≤20-line config, docs — stayed open at close.
- **Marin-as-a-library ([#4271](https://github.com/marin-community/marin/issues/4271)) — partial.** By 2026-04-15 there were "2 demo examples… good enough for a 'getting started' by volunteers" via the new `marin-experiments` repo (rjpower), but the full clean-venv CI import test / curated public API surface was still forward-looking.
- **On-demand & reserved capacity ([#4268](https://github.com/marin-community/marin/issues/4268)) — mostly in place at kickoff.** The Iris reservation system already existed at the milestone's start; no explicit close-out comment surfaced in the corpus, so I can't confirm formal sign-off.
- **Canary pass rate to 90%+ ([#4270](https://github.com/marin-community/marin/issues/4270)) — missed within the window.** The TPU ferry was perfect (9/9), but the **GPU canary regressed to 60%** after JAX 0.9.2 exposed an NCCL all-to-all crash ([#5377](https://github.com/marin-community/marin/issues/5377), fix [#5379](https://github.com/marin-community/marin/issues/5379)), and the datakit smoke ferry slipped to ~71% ([summary 2026-04-26](https://mws.oa.dev/summaries/summary-2026-04-26_2026-05-02.html)). The alerting completion box also stayed unchecked. This is the clearest quantitative miss.

## Scorecard

| Milestone element | Target | Outcome |
|---|---|---|
| Kick off pre-registered big MoE | Launch the run | **Hit** — launched + pre-registered in April ([#4697](https://github.com/marin-community/marin/issues/4697), [#4281](https://github.com/marin-community/marin/issues/4281)) |
| Pre-registered loss | ~2.25 paloma macro | **Beat** — 2.234, but not final until 2026-05-20 ([#4697](https://github.com/marin-community/marin/issues/4697)) |
| Sustained TPU MFU | 25–30% (self-described "made up") | **Missed** — ~16.4% ([#4300](https://github.com/marin-community/marin/issues/4300)) |
| Off Ray | Ray gone | **Hit** ([#4269](https://github.com/marin-community/marin/issues/4269)) |
| Agentify experimentation | ≥3 agent-run decisions | **Hit** — AdamH-embed promoted ([#4282](https://github.com/marin-community/marin/issues/4282), [#5184](https://github.com/marin-community/marin/issues/5184)) |
| Usability & observability | April scope | **Hit** ([#4273](https://github.com/marin-community/marin/issues/4273)) |
| Canonical data pipeline | Composable download→tokenize | **Mostly hit** ([#4272](https://github.com/marin-community/marin/issues/4272)) |
| Marin-as-a-library | External import | **Partial** ([#4271](https://github.com/marin-community/marin/issues/4271)) |
| Canary ≥90% | Both fleets | **Missed** — GPU ~60% on JAX 0.9.2 ([#4270](https://github.com/marin-community/marin/issues/4270)) |

## Bottom line

- **Won:** off-Ray migration completed; canonical data pipeline stood up; agentic experimentation produced a real promotable architecture change; and the pre-registered 1e23 scaling bet **beat its forecast** (2.234 vs 2.252 paloma macro) — a well-calibrated scientific result.
- **Slipped / missed:** the marquee 100B-A13B-class run **did not finish in April** (crash-bound; completed 2026-05-20); **canary pass-rate 90%+ was not achieved** (GPU at 60%); **TPU MFU 25–30% was not sustained** (~16.4%, on an admittedly made-up target).
- **Process note:** most epics were closed "for the April milestone" in **early May** (≈May 3–4), with continuation work handed to the May infra tune-up ([#5369](https://github.com/marin-community/marin/issues/5369)) — i.e. the milestone was largely a success but delivered with a short tail into early May rather than cleanly by April 30.

---
<!--provenance-->
> *Data: marinmirror (frozen eval corpus) — 68026 chunks, built 1h ago · summaries: 15 weeks through 2026-07-06_2026-07-12. Refresh disabled (frozen-corpus run).*
>
> *Query: "how did we do on our april 2026 milestone?"*
>
> *Sub-queries: "April 2026 milestone overall / success criteria" · "Infra April 2026 (#4256) epic close-outs" · "Modeling April 2026 (#4266) epic close-outs" · "1e23 / 100B-A13B pre-registered run outcome (#4697)" · "canary pass rate 90%+ (#4270)" · "off-Ray single-way-of-running-jobs (#4269)" · "MoE MFU at scale / TPU MFU target (#4283, #4300)"*
