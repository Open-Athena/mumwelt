# Target benchmarks and development proxies

**Short answer.** Marin has no single named "target benchmark suite." What exists is a three-tier portfolio: (1) a large **perplexity/BPB basket** — Paloma macro loss, Uncheatable Eval, OLMoBaseEval Easy — which is what we actually *steer on* day to day; (2) **standard generative/capability benchmarks** — DCLM CORE, MMLU, GPQA, HumanEval, AIME24/25, MATH500, GSM8K — run through Evalchemy; and (3) **agentic benchmarks** — Terminal-Bench 2.0 and SWE-bench Verified — run through Harbor. Tier 1 is the development proxy; tiers 2–3 are the targets. The rule meant to connect them is written down but not yet operationalized, and several of our proxies have measurably failed to predict the targets.

The closest thing to an authoritative statement of the portfolio is Percy Liang in `#evals` (2026-07-13): these new evals "will be added to our basket of our **200+ perplexity evals and standard generative evals** that we will monitor at the end of training (which includes pretraining + midtraining + SFT) before RL, in addition to using hint-mediated forecasting of the harder generative evals" ([msg](https://discord.com/channels/1354881461060243556/1356487738840318002/1526324939743957022)). A community-facing wishlist of evals the core team wants was opened as [#7090](https://github.com/marin-community/marin/issues/7090) — the pointer is in-corpus, the issue body is past the freeze, so I can't summarize its contents.

---

## 1. The development proxies (what we steer on)

### Paloma macro loss — the primary in-training metric

The 67B-A2B 10T hero run tracks exactly **three** "Final eval metrics" in its job summary: `eval/paloma/c4_en/bpb`, `eval/paloma/macro_loss`, and `eval/uncheatable_eval/macro_loss` ([#6044](https://github.com/marin-community/marin/issues/6044#issuecomment-4820179240)). That list *is* the practical development-proxy set for a hero run.

Paloma macro loss is an average over **16 validation slices with equal weight per slice** — "so a single PTB doc gets the same weight in the macro as 49,828 twitterAAE docs" ([#6277](https://github.com/marin-community/marin/issues/6277#issuecomment-4644979372)). Slices include `c4_en`, `c4_100_domains`, `falcon-refinedweb`, `mc4`, `redpajama`, `dolma-v1_5`, `dolma_100_programing_languages`, `wikitext_103`, `m2d2_s2orc`, `m2d2_wikipedia`, `ptb`, and several toxicity/dialect sets. A structural caveat that matters for reading any cross-config comparison: the seqlen signal "lives almost entirely in 3 of 16 slices" (the concatenated mega-doc slices); the other 13 are "effectively insensitive to seq_len" (same comment).

**Paloma is what we preregister against.** The practice is well established and the track record is good:

| Run | Preregistered target | Achieved | Source |
|---|---|---|---|
| Delphi dense 1e22 | 2.55 | **2.53** | [#1337](https://github.com/marin-community/marin/issues/1337#issuecomment-4016705345) |
| 1e23 MoE (d5120) | 2.252 (L∞=1.6) | **2.234** | [#4697](https://github.com/marin-community/marin/issues/4697#issuecomment-4498921338) |
| 67B-A2B stage 1 (8T mark) | **2.269** | not yet reached | [#6044](https://github.com/marin-community/marin/issues/6044#issuecomment-4820008980) |

The 2.269 is a **target, not a result**: "Preregistered loss target for stage 1 of the run (first 8T tokens): 2.269 paloma macro loss when evaled at seqlen 8192 on 1024 sequences per eval" ([#6044](https://github.com/marin-community/marin/issues/6044#issuecomment-4820008980)). It comes from a 3-point fit with a *chosen* asymptote (E=1.4 → 2.269; E=1.3 → 2.258; E=1.6 → 2.295), and Larry flagged the thinness himself: "We are extrapolating from 3 points — our prior 1e23 run extrapolated at compute optimal from over 20 points."

The July intermediate cooldown finished at **2.2772 Paloma macro loss, 0.8242 bpb**, a 0.109 (−4.6%) drop over ~211B cooldown tokens ([weekly summary 2026-07-06_2026-07-12](https://mws.oa.dev/summaries/summary-2026-07-06_2026-07-12.html), on [#6811](https://github.com/marin-community/marin/issues/6811)). **This is not an apples-to-apples hit on the 2.269 target**: it is measured at ~2.1T tokens rather than 8T, after a full LR decay, on a different data-mix phase, and at 8× the context (65,536 vs the preregistered seqlen 8,192) — and per the slice analysis above, seqlen alone moves the macro. It is a favorable early signal, not retirement of the preregistration.

### The BPB suites: Uncheatable Eval and OLMoBaseEval Easy

Two named BPB suites are used as explicit optimization targets for data-mixture work:

- **Uncheatable Eval** — perplexity on "a diverse, high-quality, and fresh dataset (e.g., arXiv, GitHub, news)" for contamination resistance ([#1600](https://github.com/marin-community/marin/issues/1600)); mixtures optimized against it in [#6602](https://github.com/marin-community/marin/issues/6602) / [#6608](https://github.com/marin-community/marin/issues/6608).
- **OLMoBaseEval Easy ("Olmix Table 9")** — a **51-component** BPB suite, now with a Marin-native evaluator so it runs in Levanter on TPU rather than through the external Stanford-SC path ([PR #6726](https://github.com/marin-community/marin/pull/6726)); mixtures optimized against it in [#6603](https://github.com/marin-community/marin/issues/6603), coverage gaps tracked in [#6717](https://github.com/marin-community/marin/issues/6717).

### Proxy *models*: the scaling ladders and the mixture swarm

Two decades below the hero run is the standing convention. willheld's description of the process: "We train an isoflop at small scales… We then train a scaling ladder using the same scaling recipe, **up to e(X-2) where eX is the hero run scale**. We use that to forecast the performance of the hero run" ([#scaling-laws, 2026-07-09](https://discord.com/channels/1354881461060243556/1356490712199462912/1524915671555899403)). Concretely the Delphi ladder spans **3e18 → 1e21 FLOPs** on a `v5p-8 → v5p-64` TPU ladder ([#6607](https://github.com/marin-community/marin/issues/6607)).

For data mixtures the proxy is far smaller: a swarm of **238 runs at 60M params / 1.2B tokens (Chinchilla)** ([#2345](https://github.com/marin-community/marin/issues/2345#issuecomment-4102921841)), chosen because Olmo 3's swarm config is "roughly 1/233 the size trained on 1/2000 of the data" of the 7B it informs ([#2345](https://github.com/marin-community/marin/issues/2345)).

For tokenizer decisions the proxy metric is **feBPB** — FLOP-equivalent BPB — which deliberately prices serving cost at the *deployment* model, not the proxy: "Cost is priced at the deployment target — a ~250B-total / ~20B-active MoE — not the small proxy models we train to measure BPB… **Pricing at the wrong scale over-penalizes large vocab and flips verdicts**" ([#6796](https://github.com/marin-community/marin/issues/6796#issuecomment-4878397410)).

---

## 2. The target benchmarks (what we're ultimately judged on)

**Base models.** The historical default is the DCLM CORE MCQ set plus MMLU and GPQA-Diamond ([#594](https://github.com/marin-community/marin/issues/594), [#1215](https://github.com/marin-community/marin/issues/1215)) — willheld: "It's everything in CORE_EVALS on main + MMLU + GPQA Diamond" ([msg](https://discord.com/channels/1354881461060243556/1370264236361646131)).

**Post-training / reasoning, via Evalchemy.** The merged integration runs "AIME24/25, AMC23, HMMT, MATH500, HumanEval+, MBPP+, LiveCodeBench, GPQA Diamond" as Marin eval steps on TPU via vLLM ([PR #2779](https://github.com/marin-community/marin/pull/2779)). Current convention: "marin-community/evalchemy + lm-eval v0.4.12, boxed grader; 128 samples/problem, temp 0.7, top_p 1.0, no chat template, 32k ctx" ([#6279](https://github.com/marin-community/marin/issues/6279#issuecomment-4850098698)).

**Agentic, via Harbor.** Terminal-Bench 2.0 and SWE-bench Verified. Benjamin Feuer's framing of the stack: a `marin-community/evalchemy` fork "combined with `marin-community/harbor` should cover the evals we need for post-training" ([#evals, 2026-06-11](https://discord.com/channels/1354881461060243556/1514733965070827641/1514739979132342492)).

The July eval epic is scoped purely to making those two harnesses usable, not to defining the suite — [#6863](https://github.com/marin-community/marin/issues/6863) is one line: "DoD: Both Evalchemy and Harbor can be easily triggered from Marin on TPUs," with children [#6865](https://github.com/marin-community/marin/issues/6865) (Harbor; "DoD: Harbor evals can be invoked, **Qwen3-32B parity on TBench2**") and [#6866](https://github.com/marin-community/marin/issues/6866) (Evalchemy).

**Selected achieved numbers**, tagged:

- **Marin 8B Base**: "leads fully open-source models of this scale on 12/16 LM benchmarks and outperforms… Llama 3.1 8B Base" ([msg](https://discord.com/channels/1354881461060243556/1356487738840318002/1371666714923630692)).
- **Delphi 1e22, midtrain+SFT+RL**: MATH500 1.6 → **53.4**, AIME24 0 → **6.6**, GSM8K 11.0 → **68.6** ([#reinforcement-learning close-out](https://discord.com/channels/1354881461060243556/1521285492602048542/1521288546697085150), tracked in [#6279](https://github.com/marin-community/marin/issues/6279)). Caveat from the same thread: these are maxima across *different* RL legs, and "they all also regressed on at least one of the benchmarks."
- **Terminal-Bench 2.0, Marin-32B** (15% TerminalCorpus SFT, v5p-64): **2/87 (~2.3%) vs Qwen3-32B**, logged explicitly as "a clear (negative) result" ([#4760](https://github.com/marin-community/marin/issues/4760#issuecomment-4667077804)).
- **The #6865 Qwen3-32B/TBench2 parity target is not met.** The nearest attempt closed as "reproduction not achieved. Best Terminal-Bench 2.0 result was 17.2% (15/87)… vs the 27.4% NemotronTerminal-32B target," with "agent timeouts (63/87 at 900s) dominat[ing] failures, not model errors" ([#4307](https://github.com/marin-community/marin/issues/4307#issuecomment-4667060933)). The real July TPU parity win is a *different* model and benchmark: GLM-4.7-swesmith on SWE-bench-Verified-100, **0.240 on Iris v6e vs 0.237 on H100** ([#6958](https://github.com/marin-community/marin/issues/6958#issuecomment-4886598435)).

---

## 3. The rule that is supposed to connect them

The Eval Manager epic ([#6499](https://github.com/marin-community/marin/issues/6499), penfever, 2026-06-18) defines the owner of "the eval suite and the **per-stage go/no-go gate metric**," and states the governing rule verbatim:

> **Nothing advances on perplexity alone past Stage 3; Stages 4–10 gate on capability / agentic / safety benchmarks.**

That is precisely the target-vs-proxy boundary. **But it is a stated design principle, not an operating gate.** In the same issue the defining work is still an unchecked box: "☐ Per-stage gate definitions — *to be scoped* (write the go/no-go metric + threshold for each active stage; pin the Stage-3 perplexity cutoff → Stages 4–10 capability/agentic/safety benchmarks)." No per-stage metric table or thresholds exist anywhere in the corpus, and the Stage 1–10 ladder itself is referenced by only two other June epics ([#6500](https://github.com/marin-community/marin/issues/6500), [#6626](https://github.com/marin-community/marin/issues/6626)) and defined in neither. The rationale for independence is worth quoting: "The gate is meant to be independent: evals are only credible if the measurer isn't the party being measured."

Note also that the July hero-run cooldown issue states its bar as neither Paloma nor a named benchmark: "Criteria is **pass@256 on X Y Z evals**" — with the eval names left as literal placeholders ([#6811](https://github.com/marin-community/marin/issues/6811)).

---

## 4. Where the proxies fail to predict the targets

This is the live problem, and it is owned: [#6712](https://github.com/marin-community/marin/issues/6712) ("Data-selection diagnostics") has a one-row Key-facts table — "**Metric to improve | Rank correlation between PPL proxy and downstream eval**." As of the freeze that issue has no comments and no achieved rank-correlation number; it is a target statement.

The negative results are substantial and should temper any claim that our proxies track our benchmarks:

- **Uncheatable-eval loss moved *opposite* to DCLM benchmark scores.** Under a heading literally reading "(Negative Result) DCLM Benchmark Scores": "despite getting lower uncheatable eval val loss we are getting lower benchmark scores at all scales" ([#2351](https://github.com/marin-community/marin/issues/2351#issuecomment-4879165171)). Eval misconfiguration was "Disproven" (replicated on both the Marin CORE sweep and the OLMO Easy Eval Suite); the surviving hypothesis is "Perhaps uncheatable eval val loss is not the right proxy for the downstream benchmarks?" The team already knew this directionally: "The loss advantage on uncheatable evals does not translate as strongly to better benchmark / task performance as we hope (we probably benchmaxxing it too hard)" ([#2345](https://github.com/marin-community/marin/issues/2345#issuecomment-4339429296)).
- **MMLU has no signal at proxy scale.** dlwh, bluntly: "the models are too small to get signal with mmlu" ([#2345 thread, 2026-03-22](https://github.com/marin-community/marin/issues/2345)). Quantified in the same issue: R² < 0.1 for MMLU vs R² > 0.7 for C4EN BPB, with seed-only variance nearly as large as whole-swarm variance; SNR at 60M/1.2B runs 22.2 for `uncheatable_eval::bbc_news` bpb down to **1.10 for `mmlu_5shot/bpb`** ([comment](https://github.com/marin-community/marin/issues/2345#issuecomment-4210192639)). Scaling the proxy to 300M/3B "did not reduce the variance" ([comment](https://github.com/marin-community/marin/issues/2345#issuecomment-4115490875)).
- **Agentic soft proxies failed.** Loss on successful trajectories vs SWE-bench Verified accuracy: "the highest was ~0.3" rank correlation, and "the best model actually got *higher* loss… than the worst model" ([#4389](https://github.com/marin-community/marin/issues/4389#issuecomment-4209804794)). Issue closed 2026-06-10; work moved to the still-open [#4550](https://github.com/marin-community/marin/issues/4550).
- **Tokenizer proxy verdicts reversed at scale.** Proxy ladders read −4.7% to −6.8% feBPB for SuperBPE; at the converged 10B-total/500M-active soak the margin was **−0.9%**, and the 128k SuperBPE arms *underperformed* the 128k Llama-3 baseline (+0.7…+1.0%) ([#6796](https://github.com/marin-community/marin/issues/6796#issuecomment-4884599203)). The n-gram embedding anti-scaled outright: benefit "shrank from −0.39% at hidden-1024 to +0.03% (neutral) at hidden-2048 — opposite the paper's scale-up claim" ([comment](https://github.com/marin-community/marin/issues/6796#issuecomment-4880641873)).
- **Contamination undercuts a target benchmark directly.** 13-word n-gram containment against Nemotron-CC-Math found **28.20% of MATH500 test** and **56.67% of AIME24** overlapping, while GSM8K is largely clean at ~2% ([#6742](https://github.com/marin-community/marin/issues/6742#issuecomment-4884573218)) — despite Nemotron-CC-Math's stated dedup against MATH500 and GSM8K.

**Where a PPL proxy did work**, for balance: base-model BPB on real math traces predicts post-RL math accuracy at r = −0.89 (cot) / −0.92 (tir), and −0.97 within the 7–8B cohort ([#6096](https://github.com/marin-community/marin/issues/6096)) — but with the author's own deflater: "it predicts capability **level**, not RL-friendliness," since predicting the RL *gain* fails at R² = 0.33 ([close-out](https://github.com/marin-community/marin/issues/6096#issuecomment-4836443867)).

The unresolved crux, from Rohith: "1) for any task, in principle does BPB on some 'right' validation corpus predict performance; 2) if yes, how do we identify this validation set. I'd guess the answer to (1) is yes. but the answer to (2) is still idk" ([#evals, 2026-04-27](https://discord.com/channels/1354881461060243556/1356487738840318002/1498462494778327070)). Nothing in the corpus answers (2).

---

## 5. What's changing

willheld has **proposed** — not adopted — redefining "compute optimal" away from pretraining loss: fit the ladder to "the **pass@256** of the pretrained model on skills that are relevant for agents rather than purely with pretraining loss," on the simplifying assumption that pass@256 converts to pass@1 under enough RL ([#scaling-laws, 2026-07-09](https://discord.com/channels/1354881461060243556/1356490712199462912/1524915671555899403)). He names the holes himself ("this naively has some problems with emergence"). Supporting work exists in [#4549](https://github.com/marin-community/marin/issues/4549) (predicting pass@k without sampling k times). No experiment and no adoption in the corpus — but note it is the same metric the July cooldown issue already reaches for.

## Gaps and caveats

- **No canonical suite document exists.** dlwh opened an `#evals` thread titled "Marin target metrics" on 2026-06-12 ([msg](https://discord.com/channels/1354881461060243556/1356487738840318002/1514825572671291427)) prompted by a community ask for "a summary of metrics marin is optimizing"; the thread has no consolidated answer in-corpus. This answer is assembled from primary sources, not from an existing spec.
- **Freeze boundary.** [#7090](https://github.com/marin-community/marin/issues/7090) (the eval wishlist) and [#7044](https://github.com/marin-community/marin/issues/7044) (CI for the harbor/evalchemy forks) are referenced from Discord and the weekly summary but their bodies are outside the 2026-07-16 freeze — I make no claims about their contents.
- **Eval output has no shared schema.** The three evaluators "write divergent, mostly nested/timestamped layouts with no shared schema" ([#6781](https://github.com/marin-community/marin/issues/6781)); a unified `summary.json` is proposed, not landed.
- **One in-corpus disagreement worth flagging:** an eval-harness config bug means published Delphi HumanEval numbers may be 0-shot rather than 10-shot — "I think we may have undersold delphi's humaneval capability in the blogpost" ([#evals, 2026-07-16](https://discord.com/channels/1354881461060243556/1356487738840318002/1527176663333343233)), tracked in #7229 (outside the freeze).

---
<!--provenance-->
> *Data: marinmirror — 68026 chunks, built 48h ago · summaries through 2026-07-06_2026-07-12 (frozen eval corpus, 2026-07-16; no refresh this run).*
>
> *Query: "What are our target benchmarks and our development proxies?"*
>
> *Sub-queries: "which downstream benchmarks does Marin target for hero runs (MMLU/GPQA/HumanEval/AIME24/MATH500/TBench2/UncheatableEval/OLMoBaseEval)" · "Eval Manager role, eval suite, and per-stage go/no-go gate metric (#6499, #6863)" · "Paloma macro loss and bits-per-byte as the in-training development metric; preregistered scaling-law loss targets (#6811, #6046, #6044)" · "small-scale development proxies: Delphi ladder, mixture swarm #2345, feBPB tokenizer proxy (#6796, #6916)" · "how well do proxies predict target benchmarks — #6712 rank correlation, DCLM negative result, agentic BPB proxies, pass@256" · "eval harnesses for target benchmarks: Evalchemy and Harbor on TPUs, post-training results, contamination checks"*
