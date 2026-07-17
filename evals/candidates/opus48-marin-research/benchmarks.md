# Target benchmarks and development proxies

Marin's eval strategy is explicitly **two-tiered**: a set of external, generative
**target benchmarks** we ultimately want to move (led by the Artificial Analysis
Intelligence Index), and a set of cheap, benchmark-leak-safe **development
proxies** — mostly perplexity-based — that we can read *during* pretraining, long
before the target benchmarks are meaningful. The whole point of the proxy tier is
to catch data/tokenizer/format gaps early and to *predict* the targets without
training the model (or the team) against held-out benchmark items.

## Target benchmarks (what we're ultimately judged on)

**The headline target is the Artificial Analysis (AA) Intelligence Index** —
Marin's proxy suite is explicitly designed against "competencies surfaced by
Artificial Analysis-style model comparisons," pinned to AA's Intelligence Index
methodology (v4.0.4 at the time) ([#5819](https://github.com/marin-community/marin/issues/5819)).
The AA text-only components enumerated as our target competencies are:

- **GDPval-AA** (professional knowledge work / document deliverables)
- **tau2-Bench Telecom** (agentic tool/state tracking)
- **Terminal-Bench Hard** (terminal / sysadmin task execution)
- **SciCode** (scientific Python coding)
- **AA-LCR** (long-context reasoning over ~100k-token doc sets)
- **AA-Omniscience** (factual accuracy + hallucination/abstention)
- **IFBench** (precise instruction following / formatting)
- **HLE text-only** (hard academic reasoning)
- **GPQA Diamond** (graduate-level science MCQ)
- **CritPt** (scientific/technical reasoning)

with AA "extra" text evals tracked alongside the main index: **MMLU-Pro, MMLU,
AIME 2025, LiveCodeBench, and Global-MMLU-Lite** (multilingual tracked
separately) ([#5819](https://github.com/marin-community/marin/issues/5819)).

**These targets are actually run as generative evals** (pass@1, not perplexity)
through the **evalchemy** harness. The evalchemy task set named in the same
thread is **MATH500, AIME24/25/26, GPQA-Diamond, HLE, and LiveCodeBench v5/v6**,
with MMLU-Pro mirrored as eval JSON
([#5819](https://github.com/marin-community/marin/issues/5819)). The AA suite is
also load-bearing enough that the decontamination pipeline decontaminates against
it directly — the v0 decon run covered **8 Artificial Analysis evals plus 849
lm-eval-harness leaves** ([#5519](https://github.com/marin-community/marin/issues/5519)).

**Comparator target:** on the pretraining side the concrete "beat this model" bar
is **Qwen3-32B** — the proxy suite is literally the "Marin 32B vs Qwen3 32B
model-perplexity gap suite," and the circuit-coverage dashboard was opened
because Marin 32B was **trailing Qwen3 32B** on short deterministic continuations
([#5836](https://github.com/marin-community/marin/issues/5836),
[#6070](https://github.com/marin-community/marin/issues/6070)).

**Pretraining loss target (a different kind of "target"):** for the in-flight
hero runs the headline number is **Paloma macro-loss**, tracked against
*preregistered* targets — e.g. the 67B-A2B run's preregistered **2.269 stage-1
target** at the 8T/80%-completion mark, against which the 2T intermediate
cooldown read **2.277** (favorable but not apples-to-apples)
([#6811](https://github.com/marin-community/marin/issues/6811),
[#6044](https://github.com/marin-community/marin/issues/6044)). This is a
loss/quality target, not a downstream benchmark.

## Development proxies (what we watch during a run)

The proxy tier is the **perplexity-gap suite** — a competency-organized portfolio
of raw-text perplexity slices, consolidated into a single durable entrypoint
`experiments/evals/model_perplexity_gap_suite.py`
([#5836](https://github.com/marin-community/marin/issues/5836)). Its workflow is
documented at `docs/references/perplexity-gap-analysis.md`
([#5803](https://github.com/marin-community/marin/issues/5803),
[#5318](https://github.com/marin-community/marin/issues/5318)).

**Core design rule — avoid "benchmark-maxxing."** The proxies deliberately do
**not** contain the actual target-benchmark items. Exact AA / HLE / GPQA / AIME /
MATH-500 / MMLU-Pro / IFBench / LiveCodeBench / SciCode / HumanEval / MBPP items
are barred from regular tracking even where a train split exists; instead each
target competency is approximated with a *distributional* surrogate
([#5819](https://github.com/marin-community/marin/issues/5819)).

Existing proxy coverage (the `perplexity_gap_registry` bundles) includes:

- **Paloma** slices (wikitext-103, m2d2 wikipedia/s2orc, etc.) — general
  English / knowledge
- **Uncheatable Eval** (arxiv, wikipedia, bbc_news, github python/cpp) — recent
  text + code surface
- **FineWeb2 multilingual** (top-50 languages + Indic) — multilingual competence
- code / long-tail slices (**Stack v2**, SVG-Stack, VerilogEval), **formal
  methods** (Z3, CoqGym, RTL), **bio/chem** notation, and **structured tables**
  (ToTTo, WikiTableQuestions, GitTables)
- an **LM-eval bridge** staging train/dev splits as raw text — with
  `mmlu_auxiliary_train` and `gsm8k_train` allowed only as small **canaries**
  ([#5819](https://github.com/marin-community/marin/issues/5819),
  [#5277](https://github.com/marin-community/marin/issues/5277)).

The reframed proxy taxonomy is organized by **competency** (MCQA, sci/technical
reasoning, math/numeric, agentic coding/debugging, instruction following,
long-context reading, professional-document, factuality/abstention, multilingual,
structured data), each tagged with a **`ppl_fidelity`** rating because raw
perplexity is a faithful proxy for some competencies (factual recall,
sci-prose familiarity) but weak for others (abstention, agentic state tracking,
long-context *retrieval*), which get non-PPL companion metrics instead
([#5819](https://github.com/marin-community/marin/issues/5819)). The competency
gaps were spun into subissues **#5823–#5829** (generated-math, instruction-
following, long-context, factuality/abstention, professional-document, MCQA
train/dev, sci/technical reasoning) plus a FEVER factuality slice
([#5819](https://github.com/marin-community/marin/issues/5819),
[#5863](https://github.com/marin-community/marin/issues/5863)).

**The proxies also include narrower diagnostic signals:** a **circuit-coverage**
probe suite (arithmetic, string/token manipulation, exact-character tracking,
literal formatting — where Marin 32B trailed Qwen3 32B)
([#6070](https://github.com/marin-community/marin/issues/6070),
[#6238](https://github.com/marin-community/marin/issues/6238)), and
**agentic-coding BPB** proxies — success-only agentic bits-per-byte was found to
be a usable, high-SNR signal at 300M
([#5401](https://github.com/marin-community/marin/issues/5401),
[#4550](https://github.com/marin-community/marin/issues/4550)).

## The link between the two tiers (and how good the proxies actually are)

The stated success criterion for a proxy is that it **predicts downstream
benchmark movement**. `#6712` is the umbrella for data-selection diagnostics,
whose explicit metric-to-improve is the **rank correlation between a PPL proxy
and downstream evals** ([#6712](https://github.com/marin-community/marin/issues/6712)).
An early correlation read inside `#5005` ran the priority raw-PPL bundle over a
historical 300M/6B mixture panel (40 raw-PPL datasets, 242 signal rows) against a
**P < 0.05** SNR target and reported low SNR on many slices — i.e. the proxies
are still being validated, and several are not yet predictive
([#5005](https://github.com/marin-community/marin/issues/5005)). Programming-
ecosystem coverage (`#5254`) was closed as done pending an IID validation set
from Marin's own code
([#5254](https://github.com/marin-community/marin/issues/5254)).

## Caveats / gaps

- **Not everything gets a pretraining proxy.** Truthfulness, hallucination
  robustness, and abstention are **explicitly deferred** to post-training /
  non-PPL evaluation — they are AA *targets* with no faithful PPL surrogate
  ([#5819](https://github.com/marin-community/marin/issues/5819)).
- **The AA index itself is a moving target** (pinned to methodology v4.0.4 as of
  the May work); the corpus doesn't show a later re-pin, so treat the exact
  component list as of mid-2026.
- The proxy suite is **still being calibrated** against downstream evals
  (`#6712`, `#5005`); low measured SNR on many slices means the proxy→target
  link is a work in progress, not a settled result.

---
<!--provenance-->
> *Data: marinmirror (frozen 2026-07-16 eval corpus) — 68026 chunks, built 1h ago · summaries through 2026-07-06_2026-07-12 (refresh disabled for this run).*
>
> *Query: "What are our target benchmarks and our development proxies?"*
>
> *Sub-queries: "target benchmarks" · "development proxies" · "Artificial Analysis Intelligence Index target" · "eval suite / proxy metrics for development" · "held-out evaluation benchmarks goals" · "evalchemy generative eval suite (MATH500/AIME/GPQA/LiveCodeBench)" · "perplexity-gap suite bundles" · "core eval dashboard tracked during training" · "smooth scaling proxy before e19/e20 flops" · "Paloma perplexity as development metric"*
