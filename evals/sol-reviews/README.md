# Sol answers vs proposed golds: editorial review

## Overall verdict

Use the **Sol answers as the prose and structure base**, then import a small set of load-bearing facts from the proposed golds. The proposed golds are substantively more complete for all nine questions, decisively so for `h100-67b`, `ablations`, `benchmarks`, and `inference`. Sol is consistently clearer, shorter, and at least as careful about evidence quality.

The Sol files are 602–933 words each; the proposed golds are 2,596–3,756 words. The right revision target is generally **900–1,300 words**, not gold-length replication.

## Prose assessment

Average reviewer scores:

| Version | Clarity | Structure | Concision | Calibration |
|---|---:|---:|---:|---:|
| Sol | 8.7 | 8.1 | 8.7 | 9.0 |
| Proposed gold | 6.8 | 7.6 | 3.4 | 8.3 |

Sol's main prose strengths are early conclusions, useful headings, explicit achieved-vs-target language, and disciplined omission of unsupported detail. Its recurring weakness is **compressed abstraction**: several answers state the right conceptual distinction but omit the concrete number, owner, experiment matrix, or current artifact that makes the answer useful.

The golds' main prose weakness is **inventory-shaped narration**. They often retain run diaries, failure suffixes, stale proposals, and issue-by-issue mechanics after the decision-relevant conclusion is already established. Summary-only claims are usually caveated somewhere, but not always close enough to the claim.

## Revision priority

| Priority | Answer | Targeted change |
|---:|---|---|
| 1 | `benchmarks` | Add the concrete three-tier stack: DCLM CORE/soft metrics, Paloma preregistration/scaling, current Table-9 BPB objective, plus the actually executed Evalchemy subset and comparator results. |
| 2 | `ablations` | Rebuild around a comparison table covering #6757, #6607–#6611, #6969, classifier audits, and contamination/test-integrity; the current answer centers older work. |
| 3 | `h100-67b` | Lead with the missing close-out: ~19.9% on 32 H100 with SGD/Adam; >20% at 128 H100 not cleared; then 21.54% TPU-v4-1024 and ~18.6% sustained TPU-v4-2048. |
| 4 | `inference` | Add the people/subsystem map, Iris serving substrate, full 2.5h/~10M-token/64-H100 RL-step context, Jupiter GH200 track, and guarded earlier baselines. |
| 5 | `gpu` | Add the finished GQA4 sibling, a separate B200 achieved/gate/target paragraph, and the TPU-only parked PP result. Tighten the provenance of 23.8%. |
| 6 | `april` | Add the quantitative scorecard: 319k chip-hours/crashes, ~16.4% MFU, canary rates, final ~52% synthetic-data run, TerminalCorpus negative result, and exact 2.252→2.234 later validation. |
| 7 | `muon` | Add the shipped May Recipe, live hero-run use, isoFLOP evidence, TPU/GPU split, and correct canonical routing wording. |
| 8 | `classifier` | Add `sonnet46-thr05`, sample/cost details, source-only AUC confound, within-source collapse, and the unmerged content-aware successor; retain the conservative sharing conclusion. |
| 9 | `july` | Add the achieved snapshot (2.1T/~18.6%, cooldown 2.2772), B200 17.8%/18.26% versus 20%/25% gates, GQA-vs-MLA rationale, and projected post-training cost. |

## Cross-cutting edit rules

1. Put the decisive numbers and status in the opening paragraph or a compact table.
2. Label every number as measured, projected, targeted, or summary-reported.
3. Keep hardware, model shape, optimizer, and scale attached to every performance metric.
4. Separate completed, running, killed, and proposed experiments in ablation/status answers.
5. Do not cite an umbrella issue for detailed results it does not contain.
6. Preserve Sol's conservative conclusion on classifier-weight redistribution: the corpus establishes no release permission.
7. Keep summary-only 7000-series evidence visibly labeled; do not style unavailable issue bodies as primary verification.
8. Prefer one comparison table over paragraphs of run-by-run narration.

## Detailed reviews

- [GPU, H100-67B, and inference](gpu-h100-inference.md)
- [Muon, ablations, and classifier](muon-ablations-classifier.md)
- [July, April, and benchmarks](july-april-benchmarks.md)

The underlying Sol answer files were not modified during this review.

