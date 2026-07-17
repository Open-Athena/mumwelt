# Independent Sol answer set

These nine answers were researched independently against the frozen 2026-07-16 Marin corpus. They are not regenerated golds and do not evaluate or optimize against the proposed golds, candidates, deterministic scorer, or machine targets.

| Question | Answer | Central fidelity guard |
|---|---|---|
| GPU training | [gpu.md](gpu.md) | Separates the 90B H100 achieved 9.37% MFU from its ~25% target, and treats the July 23.8% figure as a smaller-shape throughput snapshot rather than a completed run. |
| 67B-A2B on H100 | [h100-67b.md](h100-67b.md) | Distinguishes successful bring-up/profile paths from unproven production-scale efficiency. |
| Muon | [muon.md](muon.md) | Separates selective MuonH parameter routing from Adam-handled embeddings, router/gates, vectors/scalars, and LM head; GPU scaling remains unresolved. |
| July 2026 plan | [july.md](july.md) | Treats July as preparation and handoff, with the B200 MFU bar labeled as a target rather than an achievement. |
| April 2026 milestone | [april.md](april.md) | Separates results available at the April boundary from later May close-outs. |
| Data ablations | [ablations.md](ablations.md) | Keeps experiment families, compute budgets, data sizes, and evaluation rulers separate instead of forcing one ranking. |
| Data classifier | [classifier.md](classifier.md) | Distinguishes oracle labeling, fastText/transformer holdout results, and later evidence that the classifier can sort by domain rather than intrinsic quality; sharing rights remain unestablished. |
| Benchmarks and proxies | [benchmarks.md](benchmarks.md) | Separates held-out target benchmarks from development proxies and retains negative proxy-validity evidence. |
| Inference | [inference.md](inference.md) | Separates TPU correctness/bring-up from GPU full-model performance and unresolved production bars. |

## Method

Research used only `evals/runners/mum-frozen`; the corpus was not refreshed. Each track oriented on recent frozen summaries, decomposed its questions into vocabulary-specific searches, expanded primary sources, and applied target-versus-achieved, platform-attribution, temporal, and quote-or-omit checks. Every answer ends with its own source-gap note and provenance trace.

