# Review: Sol answers vs proposed golds

## `gpu`

### 1. Substantive verdict

**Proposed gold is materially more complete; Sol is clearer and better calibrated in its core claims.** Sol gets the most important target/result distinction right: the 90B/5.3B-active campaign improved 2.8% → 9.37% on 256 H100 and explicitly closed at the practical pure-FSDP ceiling rather than the ~25% target ([#6304](https://github.com/marin-community/marin/issues/6304)). It also correctly refuses to turn the weekly summary's ~23.8% into a durable completed MLA run. However, it misses three question-defining branches that the gold recovers: the finished **GQA4** sibling, the B200/GB200 state and contested hero-run shape, and the TPU-only/parked pipeline-parallel comparison.

### 2. Facts/citations Sol should add, drop, or correct

- **Add the finished sibling that resolves the apparent contradiction.** The only finished 500B H100 run is `grug-d2048-L24-gqa4-500B-r8-nosim-v2` (10.77B, GQA4, 500B tokens, final train loss 1.629, eval BPB 0.7273). Sol currently says replacements were running but never explains that a different GQA4 architecture did finish. This distinction is more informative than the generic “snapshot” wording.
- **Tighten the 23.8% sentence.** Replace “~23.8% is a measured throughput snapshot” with “~23.8% is **weekly-summary-asserted**; no frozen W&B card logs MFU for this d2048-500B family.” The proposed gold is right that “measured” overstates provenance.
- **Add B200 as a separate hardware section.** State that 17.8% on 8×B200 is summary-asserted and below #6706's ≥20% gate; 25% is a later target. Then add #6689's conflicting “288 B200 (4×NVL72)” stub versus the summary's “~120B-A8B on 2×NVL72”; model size, data, FLOP budget, and preregistered loss were unresolved. Cite [#6689](https://github.com/marin-community/marin/issues/6689), [#6706](https://github.com/marin-community/marin/issues/6706), and [#6710](https://github.com/marin-community/marin/issues/6710).
- **Add the freshest Blackwell run state only if kept compact.** The `gb200-d1280-sonic-cute` sweep showed reference/cuDNN failures and only smaller FA4/CuTe variants finishing; the d5120/L48 bring-up was stable through 128 GPUs while 256/512/640-GPU attempts crashed. Label these W&B run states, with no MFU logged.
- **Add pipeline parallelism with platform guard.** #6532 demonstrated a 1.22× advantage over FSDP at v6e-32 but 0.78× on one TPU host, then was parked and PR #6534 closed unmerged. It validates the structural direction but is not an H100 result ([#6532](https://github.com/marin-community/marin/issues/6532)).
- **Add the identifiable hero-run attribution.** The 67B-A2B 10T run was on TPU: 21.5% on the v4-1024 attempt was a high-per-chip point/78-day projection, while the launched v4-2048 run later sustained ~18.6%. Sol should avoid implying GPU progress covers the active hero pretrain ([#6044](https://github.com/marin-community/marin/issues/6044)).
- **Drop or qualify “Iris job federation all exist.”** The cited #6292 proves cluster bring-up, Kueue, and multi-node training, but not the later whole-job Iris federation claim; Sol's own research note says #7064 is absent. Either cite the weekly summary explicitly or remove “federation.”

### 3. Gold defects Sol rightly avoids

- The gold is overlong for the broad question and devotes too much space to individual crash-loop rows, GB200 suffixes, and W&B forensics.
- Several gold details are only summary-attested or refer to issue numbers not retrievable from the frozen primary corpus (#7012, #7024, #6979, #6998, #7079, etc.). The gold generally flags this, but its prose sometimes still presents those numbers prominently enough to look primary.
- “Almost certainly” linking the finished GQA4 run to the summary's 23.8% is an inference, not a corpus fact. Sol rightly avoids making that identification.
- The gold's v4-1024 wording is temporally narrow: #6044 shows the first attempt there, but the later production run did move to v4-2048. A final answer should state both rather than declare one slice uniquely correct.

### 4. Prose scores (1–10)

| Dimension | Sol | Gold |
|---|---:|---:|
| Clarity | 9 | 7 |
| Structure | 8 | 7 |
| Concision | 9 | 4 |
| Calibration | 8 | 8 |

### 5. Targeted edits

- Replace the first sentence under “current validation run” with: **“The weekly summary's ~23.8% is not a frozen run-card metric: the MLA attempts crash-looped on July 13 and were still recovering July 14, while a separate 10.77B GQA4 sibling did finish 500B tokens on July 9.”**
- Insert a short “Blackwell” subsection after reliability: one paragraph for the 17.8%/20%/25% distinction, one sentence for #6689's contested run definition.
- Replace the final paragraph's generic “Blackwell kernel…” sentence with a concrete three-line status: H100 pure-FSDP ceiling; B200 below gate; multi-rack stable only through 128 GPUs at freeze.

## `h100-67b`

### 1. Substantive verdict

**Proposed gold is decisively better on substance.** Sol correctly explains the memory wall, Muon tax, startup failures, and the distinction between component wins and a production training result. But it omits the actual June H100 headline—**~19.9% MFU on four nodes / 32 H100 with SGD/Adam**—and therefore its “final MFUs” section is incomplete. Frozen Discord directly verifies that number and the associated decision: “Muon needs to go for now” ([June 26 GPU thread](https://discord.com/channels/1354881461060243556/1399998407657001062/1520200189690773564)). It also omits the eventual TPU production result and most of the July revival.

### 2. Facts/citations Sol should add, drop, or correct

- **Add the load-bearing H100 close-out:** ~19.9% MFU on 4 nodes = 32 H100 with SGD/Adam; below the >20% at 128-GPU bar and on a substitute optimizer. This is the closest production-shape H100 number and must lead the “final MFUs” table ([#6693](https://github.com/marin-community/marin/issues/6693), [Discord](https://discord.com/channels/1354881461060243556/1399998407657001062/1520200189690773564)).
- **Correct “There is no corpus-supported final ≥20% MFU” by adding what did exist.** The sentence is literally true but rhetorically evasive because it suppresses 19.9%. Say: “The best production-shape H100 close-out was ~19.9% at 32 H100 with SGD/Adam; the >20%/128-GPU bar was not cleared.”
- **Add the optimizer communication magnitude:** the expert-weight reshuffle was about 121.9 GiB bf16 and did not shrink with node count. That explains why Muon, not just Newton–Schulz compute, was structurally bad on FSDP.
- **Add ring vs DeepEP.** Ring EP remained the fastest trainable backend; DeepEP's ~19.7% component path did not beat ring and never reached a real train step ([#4312](https://github.com/marin-community/marin/issues/4312)).
- **Add the resolution on TPU.** The hero run moved off H100. v4-1024 EP=1 reached 21.54% but projected ~78 days and crashed; the chosen v4-2048 production run sustained ~18.6% on `resume15k_v2`. Sol's current use of the 90B H100 comparison as its third “final MFU” is less relevant than these actual 67B numbers ([#6044](https://github.com/marin-community/marin/issues/6044)).
- **Add July working-copy results, carefully labeled.** The reported 26.7% at 8 H100 → 26.5% at 64 H100 applies to a reduced 64-expert/~18B working copy, not the 67B/256-expert model. JaxPP's 18.26% on 32 H100 is likewise a later PP point, not the hero run's final MFU. Both are weekly-summary-attested through out-of-corpus #6979/#7024 and must be labeled accordingly.
- **Add #6532's June PP failure and later distinction.** The hand-rolled PP work was parked/closed unmerged; July JaxPP was a different vehicle. Sol's generic “pipeline-parallel” mention does not capture this chronology.
- **Qualify the single-node A/B context.** 4.75% MuonH versus 16.19% SGD is excellent diagnostic evidence, but it is not the campaign's “final” result. Move it under blockers and reserve the final table for 19.9% H100, 21.54% TPU-v4-1024 attempt, and ~18.6% sustained TPU-v4-2048.

### 3. Gold defects Sol rightly avoids

- The gold mixes primary-source claims with weekly-summary-only claims. #6979, #6998, #7012, and #7024 cannot be shown directly in the frozen corpus; their roofline, weak-scaling, JaxPP, and B200 numbers must remain labeled summary-attested.
- The Blackwell planning coda is beyond the narrow question and makes the answer feel like a GPU roadmap rather than a 67B bring-up retrospective.
- “Final MFU” is slightly misleading for an in-progress 10T TPU run; ~18.6% is the latest sustained production-run value, not a completed-run final.
- Some component numbers (DeepEP ~19.7%, one-node fused-kernel ~20%) risk being mistaken for train-step or production-shape results despite caveats. Sol's stricter component-vs-end-to-end separation is better.

### 4. Prose scores (1–10)

| Dimension | Sol | Gold |
|---|---:|---:|
| Clarity | 8 | 7 |
| Structure | 8 | 8 |
| Concision | 8 | 5 |
| Calibration | 8 | 8 |

### 5. Targeted edits

- Replace the opening paragraph with a four-number synopsis: **19.9% on 32 H100 with SGD/Adam; >20% at 128 H100 not cleared; 21.54% on the slower v4-1024 TPU attempt; ~18.6% sustained on the chosen v4-2048 TPU run.**
- Rename “Final MFUs” to “Measured close-out points” and use a four-column table: hardware/config, optimizer, achieved MFU, interpretation.
- Delete the 90B/5.3B-active comparison from the final table; keep it as a one-sentence corroborating comparison under “Why FSDP stalled.”
- Add one compact chronology: OOM/startup → reduced-batch profiling → 19.9% SGD/Adam → Muon shelved → run moved to TPU → July reduced-shape/JaxPP experiments.

## `inference`

### 1. Substantive verdict

**Proposed gold is much more complete; Sol is a good executive summary but does not answer “who” or “current performance” deeply enough.** Sol correctly distinguishes TPU correctness/eval readiness from GPU RL throughput, correctly reports the ~10× Levanter speedup / within ~12% of vLLM TPU as a scoped comparison, and correctly refuses to invent a GPU token-rate baseline. It misses the Iris inference service, the role/component map, the achieved CoreWeave end-to-end RL step number, Jupiter/GH200 work, and prior quantified baselines.

### 2. Facts/citations Sol should add, drop, or correct

- **Add the serving substrate first:** Romain Yon owns the Iris inference service / GrugMoE vLLM enablement; #5285 landed the handoff contract, #5887 built brokered serving, #5400's formal design went stale. Without this, Sol's “people” answer starts too low in the stack.
- **Add the actual people map:** Romain Yon (Iris/vLLM/export), Benjamin Feuer/penfever (MarinSkyRL and role epic), David Hall (Levanter/JAX reference), Russell Power (`marin-serve`/broker work), AlienKevin (prior async-RL and SWE-ZERO throughput), Isaac Hodes (tracking and format/harness scopes).
- **Add #6500/#6626/#6627.** #6500 is a role charter, not evidence that one specialist role is filled; the bidirectional Format Library and Agent Harness are scoped but not shown complete ([#6500](https://github.com/marin-community/marin/issues/6500), [#6626](https://github.com/marin-community/marin/issues/6626), [#6627](https://github.com/marin-community/marin/issues/6627)).
- **Replace “~10M generated tokens per RL step” alone with the complete achieved context:** about **2.5 hours per end-to-end RL step**, ~10M generated tokens/step, 64×H100 / 8 nodes on CoreWeave, 131k context. Explicitly say this is a wall-clock and count, not isolated inference throughput; the generation/training split was not measured.
- **Add the separate Jupiter/GH200 track.** #6335's Qwen3-Coder-30B-A3B proof of life ran ~12 hours on 6 nodes / 24 GH200 at JSC Jupiter. Do not conflate its GH200 pinning and 80B bring-up failures with CoreWeave H100 ([#6335](https://github.com/marin-community/marin/issues/6335)).
- **Add GPU correctness specificity:** GrugMoE EP8 correctness was validated on 8×H100 CoreWeave; RL-grade performance was explicitly deferred to #6870. Sol currently says only “full-size support active,” which understates what landed.
- **Add existing baselines with guards:** #2237 achieved 4.1 → 15.4 req/s (3.7×) on the **trainer**, not rollout inference; #4719 achieved 3.4× per-worker rollout throughput from prefix caching on vLLM/TPU. These show #6709 is missing an H100 serving baseline, not that Marin has no inference measurement history ([#2237](https://github.com/marin-community/marin/issues/2237), [#4719](https://github.com/marin-community/marin/issues/4719)).
- **Add fork divergence concretely:** serving uses `marin-community/vllm` plus `tpu-inference`; RL used `mlfoundations/vllm` branch `penfever/working`, creating a coordination blocker.
- **Add the 4096-token wedge only with caveat:** the reproducible case was a position-limit NaN → HTTP 400, not a broker failure; the original field stall remained unreproduced. This is summary-only in the freeze.
- **Correct “Romain drove much of…” into firmer sourced ownership language.** Issue authorship plus #5285/#5400/#5887/#6867 supports “Romain owns/leads Iris inference and GrugMoE vLLM,” while the current hedge makes the people section feel under-researched.

### 3. Gold defects Sol rightly avoids

- The gold is too expansive: format libraries, agent harnesses, stale RFCs, individual serve boot bugs, fork-lock timeouts, RPA v3, and 131k training details swamp the direct answer.
- Several freshest bug details (#6983, #7085, #7117, #7097, #7111/#7133) are only weekly-summary narrated because their bodies are unavailable. Sol rightly avoids presenting them as primary-source-verified.
- The gold's “owns” language sometimes derives from visible activity rather than formal assignment. Its own caveat that #6500 is a role definition, not a filled role, should govern the entire people map.
- The 2.5-hour RL step is useful but not an inference benchmark. The gold labels this correctly; its prominence still risks readers converting it into an implicit generation rate.

### 4. Prose scores (1–10)

| Dimension | Sol | Gold |
|---|---:|---:|
| Clarity | 9 | 7 |
| Structure | 8 | 8 |
| Concision | 9 | 4 |
| Calibration | 9 | 8 |

### 5. Targeted edits

- Replace the opening with a three-track map: **Iris/vLLM serving (Romain), Levanter/JAX reference (David), MarinSkyRL RL inference (Benjamin),** then mention supporting owners.
- Add a compact performance table with rows for TPU vLLM correctness, Levanter within ~12%, CoreWeave 2.5 h / 10M tokens / 64 H100, #2237 trainer baseline, and #4719 rollout baseline. Include a “what this is not” column.
- Split open issues into four buckets only: GPU baseline/performance, full-size correctness, fork consolidation, and serving substrate/design debt. Move individual bugs into one parenthetical sentence.
- Replace “there is no verified final GPU throughput result” with: **“There is no isolated H100 rollout tok/s baseline or agreed target; the current achieved number is only an end-to-end ~2.5 h RL step producing ~10M tokens on 64 H100.”**

## Overall recommendation

Use the Sol prose as the editorial base, but import the proposed gold's missing load-bearing facts. The largest substantive repairs are: add GQA4/B200/PP to `gpu`; add 19.9% at 32 H100 and the 18.6% v4-2048 resolution to `h100-67b`; and add Iris ownership, 2.5 h/10M/64-H100 context, Jupiter GH200, and prior baselines to `inference`. Preserve Sol's stronger brevity and its discipline around summary-only evidence.
