# Target Benchmarks and Development Proxies

Marin runs a two-tier evaluation philosophy: **target benchmarks** are the
(often expensive, slow, sometimes near-floor-for-small-models) downstream tasks
we ultimately care about, while **development proxies** are the cheap, dense
signals we watch during training and data-mixture search because they can be
computed continuously and correlate — imperfectly — with the targets.

The task menu that both tiers draw from is defined in
[`experiments/evals/task_configs.py`](https://github.com/marin-community/marin/blob/main/experiments/evals/task_configs.py),
wired into groups in
[`experiments/evals/evals.py`](https://github.com/marin-community/marin/blob/main/experiments/evals/evals.py),
and described in
[`docs/explanations/evaluation.md`](https://github.com/marin-community/marin/blob/main/docs/explanations/evaluation.md).

## Target benchmarks

### `CORE_TASKS` — the default multiple-choice suite
The default in-loop and post-hoc MCQ set, a subset from page 43 of the DCLM
paper. It contains: `agieval_lsat_ar`, `arc_easy` (10-shot), `arc_challenge`
(10-shot), `boolq`, `commonsense_qa`, `copa`, `hellaswag` (0- and 10-shot),
`lambada_openai`, `openbookqa`, `piqa`, `wsc273`, and `winogrande`.
`CORE_TASKS_PLUS_MMLU` adds MMLU (0- and 5-shot);
`CORE_TASKS_PLUS_LEADERBOARD` adds `leaderboard_bbh` (3-shot) and
`leaderboard_gpqa`.

### `key_evals` — the headline "do we have a good model" suite
Settings chosen to compare against OLMo 2. `KEY_GENERATION_TASKS`: `ifeval`,
`gsm8k_cot` (8-shot), `drop`, `humaneval` (10-shot), `bbh_cot_fewshot`
(3-shot), `minerva_math` (4-shot). `KEY_MULTIPLE_CHOICE_TASKS`: `mmlu` (0- and
5-shot) and `truthfulqa_mc2` (6-shot). Entrypoint:
[`run_key_evals.py`](https://github.com/marin-community/marin/blob/main/experiments/evals/run_key_evals.py).

### Open-LLM-Leaderboard-style tasks
`OPEN_LM_LEADERBOARD_MCQ` (`leaderboard_bbh`, `leaderboard_mmlu_pro`,
`leaderboard_gpqa`, `leaderboard_musr`) and `OPEN_LM_LEADERBOARD_GEN`
(`leaderboard_ifeval`, `leaderboard_math_hard`). The file also defines large
themed banks — `REASONING_TASKS`, `MATH_TASKS`, `CODE_TASKS`,
`LANGUAGE_TASKS`, `KNOWLEDGE_TASKS`, `MEDICAL_TASKS`, and multilingual banks
(`MMMLU_*`, `MGSM_*`, `XSTORYCLOZE_*`, `BELEBELE`, `INCLUDE_44`).

### Agentic / Harbor benchmarks (the hard targets)
Run through Marin's Harbor integration in containerized environments: **AIME,
Terminal-Bench, SWE-bench Verified**, and other registry datasets
([`docs/harbor-integration.md`](https://github.com/marin-community/marin/blob/main/docs/harbor-integration.md)).
These are the "expensive, slow, and often near-floor for smaller models"
targets that motivate the proxy research below (#4389).

## Development proxies

### In-loop training metrics (per-eval, logged to W&B)
Beyond raw task accuracy, the in-loop Levanter evaluator tracks four
multiple-choice metrics used as smoother, lower-variance signals
([evaluation.md](https://github.com/marin-community/marin/blob/main/docs/explanations/evaluation.md)):

1. **Bits per byte (`bpb`)** = `-log_prob / byte_length * ln(2)` — the primary
   loss-style proxy watched during pretraining.
2. **Log probability (`logprob`)** of the correct answer.
3. **Choice log probability (`choice_logprob`)**.
4. **Length-normalized choice probability (`choice_prob_norm`)**.

`convert_to_task_metrics` in `task_configs.py` turns any task list into
`lm_eval/<task>/<metric>` keys so these proxies can feed scaling-laws analysis.

### Perplexity-gap dashboards
A running "perplexity gap" diagnostic evaluates bits-per-byte/perplexity across
many data-domain slices (structured text, gittables, and — being added —
time-series, table, and geospatial slices) to compare models cheaply across
domains: #5534, #5059 (`exp_model_perplexity_gap_all_available_diag.py`).

### Proxies for expensive / agentic targets (active research)
Because agentic targets are too costly for data-mixture swarms, Marin is
searching for cheap proxies that track them:

- **#4389 — "Identify a soft proxy for agentic benchmarks to support
  data-mixture studies."** Tests *positive-trace loss* (conditional loss on
  successful trajectories) and *success–failure gap* (loss(failed) −
  loss(successful)) as proxies for SWE-bench Verified / Terminal-Bench.
  Current read: **mixed/negative** — on a ~15-model MATH study and top
  Qwen3-8B SWE-bench finetunes, log-likelihood proxies were noisy and
  non-monotonic (rank correlation often < 0.3–0.8), and the gap did worse than
  plain loss. Also notes OpenThoughts-TBLite as a cheaper-hard-benchmark
  alternative.
- **#4550 — "Reliable scaling for downstream evals/post-training"** (the
  umbrella issue). Goal: predict expensive-run downstream performance from
  cheaper runs (smaller model and/or fewer tokens). Candidate proxies:
  weighted log-likelihood on strong-model trajectories, pass@large-k, held-out
  generalization loss, and min SFT validation loss. Notes that **plain
  perplexity is a weak/misleading proxy at fixed size**, motivating a shift to
  predicting along **scaling ladders** (Marin IsoFLOPs / Delphi-Nemotron) with
  *selection regret* rather than raw R² as the success metric.
- Sub-threads: **pass@k** as a proxy (#4549 "Predicting pass@k"), **isoflops/
  Delphi ladders** across mid/post-training mixes (#4551), and **predicting
  outcomes from intermediate checkpoints / fewer tokens** (#4548).

## Summary

| Tier | What | Examples |
|---|---|---|
| Target benchmarks | Downstream accuracy we optimize for | `CORE_TASKS` (DCLM subset), MMLU, `key_evals` (OLMo2-comparable: GSM8K, HumanEval, IFEval, DROP, MATH, BBH, TruthfulQA), leaderboard (BBH/GPQA/MMLU-Pro/MuSR), agentic Harbor tasks (AIME, Terminal-Bench, SWE-bench Verified) |
| Development proxies | Cheap dense signals watched during training | In-loop `bpb`/`logprob`/`choice_logprob`/`choice_prob_norm`, perplexity-gap slices (#5534, #5059), and researched proxies for hard targets: positive-trace loss & success–failure gap (#4389), pass@k (#4549), held-out/SFT val loss and scaling-ladder prediction (#4550, #4551, #4548) |
