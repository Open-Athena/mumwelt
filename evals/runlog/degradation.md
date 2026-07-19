# Degradation log — 2026-07-18 A/B run
Questions where an agent reported subagent failures (rate limits). Policy: re-run
BOTH arms of any affected question, so the pair stays matched.

- base/benchmarks : 2 of 6 subagents died (target-benchmarks, loss-proxies); covered in-thread. NEEDS RERUN (both arms)
- chunked/classifier : top-level agent failed outright (rate limit). NEEDS RERUN (both arms)
- base/july : 4 subagents rate-limited but ALL RESUMED via SendMessage; full facet coverage. Recovered — no rerun needed.
- chunked/benchmarks : clean (58 tool uses, no failures). Pair is asymmetric vs base/benchmarks → rerun pair.
- base/april : 3 subagents rate-limited, all RELAUNCHED; full facet coverage. Recovered — no rerun needed.
- base/classifier : clean, 6 subagents all routed via mum-frozen-base. OK.
- chunked/april : 4 subagents rate-limited; 2 resumed, 1 relaunched, 2 facets researched in-thread. PARTIAL (mild).
- base/gpu, base/muon, base/h100-67b, chunked/inference, chunked/benchmarks : clean.

NOTE: rate-limit pressure hit BOTH arms (they ran concurrently), so noise is broadly
balanced rather than systematically favouring one index. But per-question pairing is
uneven, so Q deltas at n=9 are weak evidence. The deterministic retrieval-recall metric
(retrieval_recall.py) is unaffected by any of this and should carry the conclusion.

## k=20 vs k=50 arm (3 questions)
- k50/inference : hit HARD 200/200 session agent-spawn cap (not transient); 4 of 6
  subagents returned, 2 facets covered in-thread. Cannot be fixed by relaunch.
  NOTE: session spawn budget is exhausted, so later agents in this batch are likely
  to be equally or more constrained. Treat this arm's Q values as lower-confidence
  than the deterministic retrieval measurement.
