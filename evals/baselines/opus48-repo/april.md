# How did we do on our April 2026 milestone?

**Short answer: We hit it.** The April milestone — [Milestone #7, "April: Kick-off pre-trained 100B-A13B 1.2T token MoE (pregistered)"](https://github.com/marin-community/marin/milestone/7) — closed **on time (due 2026-04-30) with all 25 of its issues completed (25 closed / 0 open)**. The headline deliverable (launch + pre-register a ~100B-A13B–class MoE) was achieved, and the supporting infra/data/agentification epics all landed. There are a couple of scope caveats (see below).

## The headline goal was met

The two "Modeling April 2026" and "Infra April 2026" tracking issues ([#4266](https://github.com/marin-community/marin/issues/4266), [#4256](https://github.com/marin-community/marin/issues/4256)) were split into concrete epics that all landed in this milestone and closed as COMPLETED.

The core modeling epic, [#4281 "MoE Scaling up to April goal"](https://github.com/marin-community/marin/issues/4281), tracked three goals:

- [x] Clean "best recipe" MoE isoflop **up to 1e22 completed**
- [x] **1e23 (120B-A12B-ish) compute-optimal launched + pre-registered** ← the milestone's headline
- [ ] Isoflop ablations on key decisions ("Good/Great 10T") up to 1e21 flops — **not completed**

So the marquee item — a ~120B-A12B (≈100B-A13B) compute-optimal MoE launched and pre-registered — was checked off. The isoflop groundwork behind it was validated up through the 1e22 scale, e.g. the `moe-v7-1e22-d3200` run reported in [#3800](https://github.com/marin-community/marin/issues/3800) (34.6B total / 4.7B active, 326B tokens; `c4_en/bpb` 0.7423, paloma macro loss 2.432).

## Everything else in the milestone closed too

All 25 issues closed COMPLETED. Highlights:

**Infra (the "Infra April 2026" split from [#4256](https://github.com/marin-community/marin/issues/4256)):**
- [#4268](https://github.com/marin-community/marin/issues/4268) On-demand & reserved capacity
- [#4269](https://github.com/marin-community/marin/issues/4269) Single way of running jobs — off Ray completely
- [#4270](https://github.com/marin-community/marin/issues/4270) Canary pass rate to 90%+
- [#4271](https://github.com/marin-community/marin/issues/4271) Marin-as-a-library (Bolinas can import marin)
- [#4272](https://github.com/marin-community/marin/issues/4272) Canonical data pipeline (download → norm → dedup/quality → tokenize)
- [#4273](https://github.com/marin-community/marin/issues/4273) Improve Usability & Observability
- [#4283](https://github.com/marin-community/marin/issues/4283) MoE MFU at scale
- [#4282](https://github.com/marin-community/marin/issues/4282) Agentify experimentation
- [#5065](https://github.com/marin-community/marin/issues/5065) Infra Stability Epic, [#4474](https://github.com/marin-community/marin/issues/4474) Levanter Store / K8s logging, plus Iris reliability fixes ([#4477](https://github.com/marin-community/marin/issues/4477), [#4479](https://github.com/marin-community/marin/issues/4479), [#4480](https://github.com/marin-community/marin/issues/4480), [#5069](https://github.com/marin-community/marin/issues/5069))

**Data / library / evals:**
- [#3100](https://github.com/marin-community/marin/issues/3100) Data sources for pre-training / mid-training
- [#4469](https://github.com/marin-community/marin/issues/4469)–[#4473](https://github.com/marin-community/marin/issues/4473) library-ization (remove experiment imports, package-relative configs, independent versioning, GitHub template repo)
- [#5247](https://github.com/marin-community/marin/issues/5247) Identify useful set of evals
- [#4294](https://github.com/marin-community/marin/issues/4294) Automate Marin updates → Discord

## Caveats / what to keep in mind

1. **One modeling goal slipped inside the epic.** [#4281](https://github.com/marin-community/marin/issues/4281) left the "Good/Great 10T" isoflop ablations (up to 1e21) unchecked, and the epic was closed at **70/84 sub-issues completed** — i.e. closed as "good enough for the April gate," not 100%.

2. **"Kick-off" meant launched + pre-registered, not a finished 1.2T-token production model.** The long production run kept being re-worked afterward: the [May milestone (#8)](https://github.com/marin-community/marin/milestone/8) was "Prepping scaling training recipe + data mix + GPU training," and by [June (#9)](https://github.com/marin-community/marin/milestone/9) the flagship run had pivoted to a differently-shaped **67B-A2B 10T (overtrained)** model rather than the 100B-A13B named in April. So April delivered the kickoff/pre-registration as promised, but the exact 100B-A13B / 1.2T config was not the model that ultimately went to full-scale production.

**Bottom line:** April 2026 was a completed milestone — shipped on time, all 25 issues closed, the headline ~120B-A12B compute-optimal MoE launched and pre-registered, and the infra/data/agentification foundation landed alongside it. The only asterisks are a partially-completed ablation sub-goal and the fact that the specific 100B-A13B shape was later superseded by smaller-active-param configs in the May/June milestones.
