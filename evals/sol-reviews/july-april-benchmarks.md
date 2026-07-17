# Sol vs proposed-gold review: July, April, benchmarks

## Executive verdict

| ID | Better reference | Proposed-gold verdict | Sol verdict | Main reason |
|---|---|---|---|---|
| `july` | Proposed gold, after aggressive trimming | `solid` | `solid_but_incomplete` | Sol has excellent calibration and a clean plan-level answer, but omits the frozen week's achieved state and several load-bearing gates/results. |
| `april` | Proposed gold | `solid` | `flawed_by_omission` | Sol gets the temporal framing right, but misses the milestone's quantitative failures, full post-training scorecard, data-source pillar, and exact registered loss target. |
| `benchmarks` | Proposed gold | `solid` | `seriously_incomplete` | Sol explains the abstraction well but omits the actual current proxy objective, historical eval stack, scaling-proxy evidence, executed target suite, and comparator results. |

The Sol answers consistently outperform the proposed golds on concision and epistemic calibration. The proposed golds consistently outperform Sol on corpus coverage and precise achieved-vs-target reporting. None of the proposed golds has a load-bearing factual error confirmed by the dispute checks below, but all three need substantial compression.

## July

### Verdict and disputes

The Sol answer's core framing is correct: July is a preparation/handoff month; the 67B run spills into August; the 20% B200 MFU number is a tentative gate in #6706, not an achieved result; and the B200 shape is unsettled. This avoids the most dangerous error: turning plan numbers into accomplishments.

The proposed gold is more complete and correctly distinguishes the major measured values. Its GQA-vs-MLA claim is supported by #6522: MLA was calculated at +17.4% FLOPs at 4k and +19.3% at 8k for the hybrid comparison, alongside an explicit Sept-15 implementation-risk concern. The Sol answer should add this decided architectural input.

Claims tied to #7073/#7074/#7024 cannot be expanded as primary issue bodies in the frozen crawl; `mum-frozen show` returns “not found.” They are supported only by the in-corpus July 6–12 weekly summary. The proposed gold discloses this limitation in its provenance; any shortened reference must retain that disclosure. In particular, 18.26% JaxPP MFU, 4/14 shape subissues, and the 0.25–0.9× post-training estimate should be cited as summary-reported facts, not as independently reverified issue bodies.

There is also unresolved planning-source tension around B200 allocation: #6689 establishes only a placeholder-shaped B200 hero run and “last run before 800+ B200,” while the summary supplies approximate two-NVL72/~120B-A8B narration. Do not present either approximate shape or rack count as preregistered fact.

### Facts/citations Sol should add

- Add the achieved July snapshot: the 67B run at roughly step 39k/~2.1T tokens/~18.6% TPU-v4-2048 MFU; the 2T cooldown at 2.2772 Paloma; and the explicit non-apples-to-apples comparison with the 2.269 8T/seqlen-8192 preregistration. Cite #6044 plus the weekly summary.
- Add B200 achieved-vs-gate: 17.8% single-node at d5120 and 18.26% multinode JaxPP, both below the tentative 20% bar. Label both as summary-derived because #7012/#7024 bodies are absent from the frozen GitHub crawl.
- Add the distinct 25% September MFU target from #6710; do not merge it with the tentative 20% August gate or H100 upskilling.
- Add GQA as the working September choice and the MLA FLOP/risk rationale from #6522.
- Add the summary-reported post-training projection: 0.25–0.9× the 10T pretrain in hardware FLOPs, explicitly a projection, plus the fact that final 67B post-training waits until August. Cite the summary and #6705.
- Add that #7073's data/architecture/preregistered-loss lock remained open, while clearly marking the 4/14 count as summary-only.

### What to drop or compress

- The proposed gold's detailed topology incident, YaRN coefficient, Executor diff, named-owner inventory, individual datakit defect wall, and long RL-framework digressions are not necessary to explain the July plan.
- The RL-vs-OPD aside under post-training price is interesting but peripheral and risks turning a plan answer into a literature review.
- Keep the SuperBPE/curation/xorl walk-backs only as a one-sentence “important negative reads,” if at all.

### Gold defects Sol avoids

- The proposed gold is roughly an order of magnitude longer than needed and buries the milestone logic under weekly activity.
- It leans heavily on summary narration for unavailable 7000-series primary issues. The provenance admits this, but the body sometimes visually cites those issue numbers as though their bodies were verified.
- “Three runs in flight” overstates the conceptual plan: the H100 validation is a dry run, not coequal with the two strategic hero tracks. Sol's “two hero-run tracks” is clearer.

### Targeted edits

Replace Sol's second hero-track paragraph with two paragraphs: one for approximate B200 shape/undecided inputs, one for measured MFU vs the 20%/25% gates. After the 67B paragraph, insert a compact achieved-state sentence containing 2.1T, 18.6%, and cooldown 2.2772 with the apples-to-oranges warning. Add one bullet under repeatability for GQA vs MLA.

### Prose scores

| Answer | Clarity | Structure | Concision | Calibration |
|---|---:|---:|---:|---:|
| Sol | 9 | 9 | 9 | 10 |
| Proposed gold | 7 | 8 | 3 | 9 |

## April

### Verdict and disputes

Sol's temporal guard is excellent: launch+preregister was committed, completion was stretch, and the eventual 2.234 result belongs to May. The proposed gold verifies the exact registered comparator: 2.252 from the fixed-L∞=1.6 fit in #4447, not merely “around 2.25.” Sol should use 2.252 when describing the later win.

The proposed gold is also right that #4256/#4266 were scaffolding, deliberately removed from the milestone so they would not “pollute” completion counts. Sol uses them as organizing sources without explaining this administrative fact, which could mislead readers about the formal scorecard.

The biggest Sol omission is the third pillar's full outcome. #3192's retrospective supports the 0→3.3→4.0→5.3 SWE-bench curve, but the proposed gold additionally reports the final scale-out stopping near 6.5M rollouts/~52% and the Marin-32B TerminalCorpus negative result. Sol's April-20 snapshot of 2.19M/25B is accurate for that date, but it is not the best final outcome for answering “how did we do.” Use the later close-out number and label it as the endpoint.

### Facts/citations Sol should add or correct

- Correct “initial prediction around 2.25” to the registered **2.252** (#4447); keep 2.234 as a May close-out (#4697), beating that registered estimate by 0.018/~1%.
- Add the exact milestone title and state that completion was explicitly a stretch goal.
- Add that three v4-1024 attempts had burned 319k chip-hours and all crashed before April ended; this makes the calendar slip concrete. This is summary-supported rather than cleanly available in #4697 alone.
- Add TPU MFU outcome: ~16.4% on a crashed v4-1024 attempt versus a self-described made-up 25–30% band, later accepted as “good enough” in #4300. Do not call it a flat hard miss.
- Add canary score: TPU 9/9, GPU ~60%, datakit ~71%; this is the clearest quantitative infrastructure miss.
- Replace the April-20 SWE-ZERO progress snapshot with the close-out: ~6.5M rollouts/~52% before cutoff. Preserve the corrected-counter warning.
- Add Marin-32B TerminalCorpus: 2/87 (~2.3%) versus Qwen3-32B 18/86 (~20.9%), a negative result (#4760).
- Add data sources #3100: target 20T, achieved beyond 19T and judged sufficient; label this “substantially met,” not mathematically 20T achieved.
- Qualify agentification: one clear Gate-2-promotable AdamH-embed result, while the stated ~3-decision bar is not demonstrated.

### Facts to drop or rephrase

- Sol says the 140B target was “far from finished” based on an April-20 17.9% snapshot. This is temporally true but an incomplete milestone verdict. Rephrase to the final ~52% close-out.
- “Canonical pipeline remains framed as a goal” is too pessimistic relative to the broader closeout evidence; use “largely delivered/rolled forward, without a clean closeout.”

### Gold defects Sol avoids

- The proposed gold is much too long and gives false precision to an inherently informal milestone with no formal completion percentage.
- Its headline “central scientific bet paid off” risks backdating a May result; the body fixes this, but Sol foregrounds the temporal distinction more cleanly.
- The proposed gold's “1 win, 3 misses” taxonomy is useful but somewhat editorial; the sources support the underlying outcomes more strongly than that exact accounting.

### Targeted edits

After Sol's opening, add a two-row “Committed vs stretch” mini-table. Replace the synthetic-data section's progress paragraph with the final ~52% outcome and add the TerminalCorpus negative result. Add a compact infrastructure scorecard containing off-Ray, observability, canaries, MFU, and data sources. End with the exact 2.252→2.234 later validation.

### Prose scores

| Answer | Clarity | Structure | Concision | Calibration |
|---|---:|---:|---:|---:|
| Sol | 9 | 8 | 9 | 10 |
| Proposed gold | 8 | 9 | 5 | 9 |

## Benchmarks

### Verdict and disputes

Sol gives a strong conceptual explanation but does not answer “what are **our** targets and proxies” with enough project-specific inventory. The proposed gold is substantially better as a reference.

The current operational proxy is verified: #6611 defines the OLMoBaseEval Easy **unweighted 51-component Table-9 BPB macro**, and PR #6726 implements it natively with fp32/byte-denominator parity. #6608 records the older Uncheatable-only scaling parent being stopped because Table-9 superseded it operationally, while Uncheatable remained a separate validation track. This entire current-state layer is missing from Sol.

The historical/load-bearing stack is also verified: PR #574 added 14/22 DCLM CORE tasks to `default_eval`; PR #817 switched to the Stanford lm-eval fork with soft metrics; PR #2779 integrated Evalchemy reasoning tasks; PR #2663's ~200-task Mega-Evals effort was auto-closed rather than becoming the default.

Sol's 300M MMLU proxy example (#5247) and negative agentic proxy discussion (#4389) are valid and useful, but they are supporting examples, not substitutes for the actual current proxy and target suite.

### Facts/citations Sol should add

- Add a three-tier map: during-training DCLM CORE/soft metrics (#574/#817); preregistered Paloma scaling ladders (#1337/#4447/#4697/#6044); current data-mixture objective Table-9 (#6611/#6726).
- Add Delphi forecast evidence: 1e21/1e22/1e23 targets 2.75/2.55/2.40; achieved 2.7581 and 2.53 for the first two, with no frozen final 1e23 number. Keep this ladder separate from the MoE ladder.
- Add MoE scaling validation: 2.252 predicted and 2.234 achieved at 1e23 (#4447/#4697).
- Add Table-9's exact definition and scoped supersession of Uncheatable (#6611/#6726/#6608).
- Name the target framework from #5819: AA Intelligence Index v4.0.4 competencies, while stressing that most are aspirational/not executed and exact items are excluded from regular PPL tracking.
- Name the actually executed reasoning/generative subset via Evalchemy (#2779/#3690): AIME, MATH500, GPQA Diamond, LiveCodeBench, HumanEval+/MBPP+.
- Add #4550 as the active downstream-scaling program that superseded/moved beyond #4389's negative soft-agentic-proxy attempt.
- Add the 67B 2.269 stage-1 Paloma target from #6044 as a projection and distinguish it from #6811's placeholder pass@256 acceptance text.
- Add the “beat Qwen” comparator with harness caveats: Marin 32B 61.96 vs Qwen2.5-32B 64.66 and OLMo-3-32B 63.20.
- Mention Mega-Evals only to say it never became the default.

### What to drop or compress

- Sol's generic operational rules can shrink to one closing paragraph once the concrete tier map is added.
- The proposed gold's extensive platform/harness map, full AA component taxonomy, historical Delphi narrative, and comparator digressions can be condensed into a table.
- Avoid presenting AA as the sole canonical release scoreboard; #5819 is explicitly a competency-proxy design exercise, and most AA components were not executed.

### Gold defects Sol avoids

- The proposed gold is encyclopedic and obscures the simple hierarchy: target capabilities → executed held-out evals → development proxies.
- It risks over-centering AA v4.0.4 as “the headline target framework” when the source mainly uses AA to define competencies for proxy coverage.
- It mixes pretraining loss targets, data-mixture objectives, and downstream benchmarks for many pages before giving the reader a compact map. Sol's conceptual distinction should become the proposed gold's opening frame.

### Targeted edits

Insert after Sol's opening a table with columns `tier`, `purpose`, `current instruments`, and `status`. Populate it with DCLM CORE/soft metrics, Paloma preregistration, Table-9 BPB, Evalchemy, and held-out AA-style/agentic targets. Replace the current proxy-portfolio bullets with the verified current-state facts above. Keep the MMLU and agentic-proxy examples as a short “what validation taught us” section.

### Prose scores

| Answer | Clarity | Structure | Concision | Calibration |
|---|---:|---:|---:|---:|
| Sol | 9 | 8 | 9 | 9 |
| Proposed gold | 7 | 7 | 3 | 8 |

## Verification note

All dispute checks used `evals/runners/mum-frozen`. No answer files were edited. Primary bodies were re-expanded where present; unavailable 7000-series July issues were treated as weekly-summary-only evidence rather than silently promoted to primary verification.
