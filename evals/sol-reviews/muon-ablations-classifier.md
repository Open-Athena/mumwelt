# Sol vs proposed-gold review: muon, ablations, classifier

## Executive verdict

| Question | Substantive verdict | Better base prose | Main action |
|---|---|---|---|
| `muon` | Proposed gold is much more complete, but internally inconsistent on parameter routing and badly overlong. Sol is calibrated but omits the shipped May Recipe, LR/isoFLOP evidence, live hero-run proof, and the TPU/GPU decision. | Sol | Expand Sol with a compact, verified “shipped recipe / evidence / platform split” section; do not import the gold's run diary. |
| `ablations` | Proposed gold contains the real 2026 mixture program and test-integrity work; Sol mostly summarizes older/background experiments and therefore misses the center of gravity. Gold is comprehensive but too encyclopedic and occasionally blurs primary-thread evidence with summary-only claims. | Gold for substance; Sol for readability | Rebuild around four current comparison tables, then retain only two historical lessons. |
| `classifier` | Proposed gold is substantially more current and complete (v0 → fast-transformer → diagnosed failure → RFC fix). Sol gets the core training/eval and the uncertainty around sharing right, but stops before #6860/#7040. Gold's “default-open” conclusion is an inference, not a corpus decision. | Sol | Add the gen-3 diagnosis/RFC and richer v0 training details; keep Sol's conservative sharing conclusion. |

---

## 1. `muon`

### Substantive verdict

Use Sol as the editorial base, but its present answer is incomplete. It correctly explains selective MuonH, distinguishes measured results from the projected crossover, and refuses to call the H100 grouped path complete. The proposed gold is the stronger research record because it adds the actual shipped May Recipe ([PR #6153](https://github.com/marin-community/marin/pull/6153)), the hero-run use of MuonH ([#6044](https://github.com/marin-community/marin/issues/6044)), the LR/isoFLOP work ([#5951](https://github.com/marin-community/marin/issues/5951), [#6074](https://github.com/marin-community/marin/issues/6074)), the TPU/GPU split ([#6493](https://github.com/marin-community/marin/issues/6493)), and the 2T cooldown/open optimizer-swap question ([#6811](https://github.com/marin-community/marin/issues/6811)). Those are load-bearing to “current approach.”

### Exact additions to Sol

1. After the short answer, add one paragraph:

   > “This selective MuonH routing shipped as part of the May Recipe in PR #6153 and is the optimizer used by the live 67B-A2B, 10T-token run on TPU v4-2048 (#6044).”

2. Add a compact “why it survived scale-up” paragraph with three carefully labeled facts:

   - #5596's measured advantage shrank with scale and its fit produced a **mixed verdict / projected crossover near 1.65e21 FLOPs**.
   - #5619's no-warmup rerun weakened that crossover concern.
   - #6074's ~2.12× is a **whole-May-Recipe scaling-law projection**, dependent on dropping the distrusted 1e18 point and pinning the exponent; it is not a measured Muon-only win.

3. Add the measured hardware split, without importing the gold's entire experiment log:

   - One 8×H100 cost-isolation A/B in #6493: MuonH ~4.75% MFU / 21,962 tok/s versus SGD ~16.19% / 74,848 tok/s. Explicitly call SGD a cost control, not a candidate optimizer.
   - The grouped/stacked path improved an isolated harness but did not establish an end-to-end production-training win.
   - On TPU, Muon overhead was approximately 1–2%, which is the key reason it remained viable there.

4. Add one current status sentence: the v4-2048 main run reached ~18.6% MFU, and #6811 closed the 2T cooldown at 2.2772 Paloma versus the preregistered 2.269 target, explicitly **not apples-to-apples**; swapping MuonH to Adam for cooldown/midtraining remained open.

### Corrections / drops

- **Correct Sol's first paragraph.** It currently says “attention gates” stay on Adam based on #5596. The corrected #5596 mask indeed keeps attention gates/vector-scalar leaves on Adam for that ablation, but the final May Recipe routing requires a source at the shipped config level. Do not present the ablation mask as a complete canonical production routing map without PR #6153/#6044.
- **Correct the proposed gold's contradiction.** It first says “only lm_head/output_proj stays on AdamH,” then in the same opening says embeddings, routers, biases, and vector/scalar leaves stay on Adam/AdamH. Replace “only” with “among matrix-shaped weight groups, lm_head/output_proj remains on AdamH; non-matrix/special groups retain Adam-family routing.”
- **Drop the proposed gold's W&B-head diary** (step ~54,922, topology incident, checkpoint publication details) from the core answer. A one-sentence live-status note is enough.
- **Drop or footnote the exact LR constants** from #5951. The gold itself documents a units-bug/self-correction cascade, so `0.09700`, `-0.395`, and R²=0.996 distract more than they clarify unless the final committed constants are audited directly.
- **Soften “GPU/Blackwell track has dropped it.”** State it as the recorded June planning decision (“Muon shelved for the Blackwell readiness track at that snapshot”), not a timeless architectural ban. #6493 proves cost pressure; it does not by itself prove every later GPU recipe excludes Muon.

### Gold defects Sol avoids

- Internal “only lm_head” versus “embeddings/routers/vectors also stay Adam” contradiction.
- Treating a large amount of run status as part of the optimizer-history answer.
- Excessive precision around self-corrected LR fits.
- Risk of turning a dated platform-track decision into a permanent statement.

### Prose scores

| Version | Clarity | Structure | Concision | Calibration |
|---|---:|---:|---:|---:|
| Sol | 8 | 8 | 8 | 9 |
| Proposed gold | 6 | 7 | 2 | 8 |

### Targeted prose edit

Replace Sol's four-step history with three headings: **Shipped recipe**, **Evidence that selected it**, **Platform/status caveat**. This preserves its strong calibration while making room for #6153/#6074/#6044/#6811 in roughly 900–1,100 words total.

---

## 2. `ablations`

### Substantive verdict

The proposed gold is decisively better on substance. Sol covers useful history (#72, #102, #164, #820, #2351) and the classifier comparison, but it misses the main current ablation program: curated/proportional/nemotron (#6757), OLMix/DSP and the four-rung Delphi ladder (#6607/#6608/#6609/#6611), embedding-surrogate validation (#6969), contamination (#6742), and the newer classifier audit (#6860). The question asks “so far,” so omitting those makes Sol materially incomplete.

The gold's problem is presentation. It reads like an experiment registry rather than an answer to “which mixes/classifiers, at what sizes, on which tests?” Its strongest content should be converted into comparison tables.

### Exact additions to Sol

1. Make **#6757 the first current result**, not the older 1B baseline program:

   - Mixes: curated, proportional, nemotron.
   - Scales: d512/d768/d1024/d1280 at 3.82e17/2.81e18/1.16e19/3.46e19 FLOPs.
   - Tests: macro Uncheatable BPB plus symmetric downstream suite.
   - Verdict: curated vs nemotron 1.78× [1.45, 2.24], but curated vs proportional 1.14× [0.93, 1.41], statistically indistinguishable. The tie is the actual decision-relevant conclusion.

2. Add a compact OLMix/DSP table:

   - Common basis: 39 Dolma3/Dolmino buckets.
   - Proxy swarms at 60M/1.2B and 300M/6B tokens; scale validation at 3e18, 2e19, 3e20, and 1e21 FLOPs.
   - Targets: Uncheatable BPB and 51-component Table-9 macro BPB.
   - Key verified result: effective-exposure DSP OOF Spearman 0.918 versus paper-faithful single-tied OLMix 0.380 on the Table-9 target (#6611).
   - Clearly label killed/superseded panels in #6608 and live/incomplete panels in #6609.

3. Add #6969 as a **summary-sourced, achieved validation**: a never-swept `dolma_starcoder` mixture realized 0.9410 Uncheatable BPB versus 0.9554 sweep-best, 0.9495 OLMix-reuse, and 0.9759 proportional. Keep the “~8σ” and winner's-curse caveat only if directly retained from the frozen summary.

4. Expand the classifier section with the source-only confound: source identity alone predicts the old oracle at AUC 0.852, already above fastText's 0.846 (#6739/#6849), and #6860 finds 12 of 98 sources with within-source score SD <0.1. This is more decision-relevant than listing older proposed classifiers.

5. Add the test-integrity gate from #6742: 28.2% of MATH-500 and 17/30 AIME24 examples matched Nemotron-CC-Math; cleaning validation reduced the reported Delphi endpoint error from +18.56% to +2.83%. This directly answers why the test set matters.

### Corrections / drops

- **Remove Sol's current-campaign paragraph sourced only to #6713.** #6713 is an umbrella with little result detail; the claimed DCLM/Nemotron/FineWeb/Resiliparse/FineWeb-Edu snapshot should cite the frozen weekly summary or the individual W&B runs. As written, the citation does not directly support the numbers.
- **Do not say “label-quality ceiling” without qualification.** #6739 later shows the apparent ~0.87 ceiling is heavily source-confounded (source-only AUC 0.852). Call it an oracle-fidelity plateau, not a ceiling on intrinsic-quality modeling.
- **Drop #2404 entirely from the proposed gold.** Even as a historical note, the question targets current comparisons and the question metadata explicitly flags it as superseded. The gold gains nothing by citing it.
- **Drop the speculative/new-feature inventory** (web-graph centrality, medical proposal) because these are proposals, not completed ablations.
- **Reduce detailed job-management prose** (Iris parents, teardown SIGABRT, deprecated dataset scripts) to one methodological footnote where it changes interpretation.
- **Separate summary-only evidence.** The proposed gold says some primary threads “post-date the freeze,” which is confusing wording for a 2026-07-16 snapshot. Say instead: “the frozen index contains the weekly-summary result but not the primary issue body; therefore this claim is summary-sourced.”

### Gold defects Sol avoids

- Mixing completed ablations, live panels, killed panels, and proposals in one continuous narrative.
- Citing superseded #2404.
- Treating summary-derived details with near-primary precision.
- Allowing proxy-fit mechanics and job telemetry to swamp the requested comparison matrix.
- The gold's own “label-quality ceiling” wording is too strong after its later source-confound section.

### Prose scores

| Version | Clarity | Structure | Concision | Calibration |
|---|---:|---:|---:|---:|
| Sol | 8 | 8 | 8 | 9 |
| Proposed gold | 6 | 7 | 2 | 8 |

### Targeted prose edit

Lead with one four-column table: **experiment / variants / training scale / evaluation + verdict**. Use five rows only: #6757, #6607–#6611, #6969, #5810/#6739/#6849/#6860, and #2351/#6742. Follow with two paragraphs on historical cooldown/classifier lessons. This would cut the gold by roughly two-thirds without losing a decision-relevant result.

---

## 3. `classifier`

### Substantive verdict

Sol is a strong concise answer to the literal question: it identifies the deployed fastText model, explains Sonnet-oracle distillation, gives the held-out AUC/ρ results, highlights the downstream-validity problem, and correctly says public weight sharing is not established. The proposed gold adds important missing current state: exact v0 training details, source-confound quantification, #6860's within-source collapse, and the unmerged #7040/rav-quality-coherence response. Those additions should be imported selectively.

### Exact additions to Sol

1. Name the deployed artifact immediately: **`sonnet46-thr05`**, fastText v0.
2. Add the v0 dataset/cost facts from #5810:

   - 7,000 stratified samples across 104 then-active sources.
   - 1,387 API/parse failures → 5,613 usable labels.
   - $32.73 spend under a $50 cap.
   - 961 fresh-oracle holdout, seed 43.

3. Clarify architecture: fastText is a bag-of-hashed-word/ngram representation feeding a linear binary `P(high)` classifier; the current Sol describes the labeling but not the model mechanics.
4. Add the decisive confound: source identity alone reaches AUC 0.852 and explains η²=0.41 of oracle-score variance, so deployed fastText at 0.846 does not beat a source lookup. The fast-transformer's 0.875 clears that baseline by only 0.023 AUC ([#6739](https://github.com/marin-community/marin/issues/6739), [#6849](https://github.com/marin-community/marin/issues/6849)).
5. Add #6860: 12 of 98 sources have score SD <0.1, with some sources effectively constant. This makes the “cannot rank within source” failure concrete.
6. Add the current fix as status, not deployment: branch `rav/quality-coherence` and RFC PR #7040 use a content-type-aware, source-blind target; the reported Spearman 0.44→0.69 is against a **new coherent-quality target**, not comparable to the old oracle ρ. PR #7040 was referenced in frozen Discord/digest records but its body is absent from the frozen GitHub corpus, so keep that provenance caveat.
7. State explicitly that #6741's fast-transformer was a recommendation/experiment, not wired into production; deployed remained v0.

### Corrections / drops

- **Keep Sol's sharing conclusion; reject the gold's “default-open” conclusion.** Public report hosting, Marin's general open-weight practice, and AllenAI's analogous release are useful context, but none answers whether this Claude-distilled artifact may be redistributed. The corpus-supported conclusion is “no explicit decision; confirm artifact license and Claude-output provenance.”
- Delete the proposed gold's extended tangential discussion of a general Claude-distillation debate. It does not concern this classifier and the gold itself admits that.
- Do not describe the 104-source count as current. It is the May-run count; the corpus later records 98/114 depending on pipeline layer.
- Reword “label-quality ceiling” as **oracle-fidelity plateau under a source-confounded target**.
- Keep the internal label-count inconsistency (5,613 usable versus a 5,052 train-class sum) as a one-line audit note at most; it should not interrupt the main training explanation.

### Gold defects Sol avoids

- Inferring release policy from broad organizational behavior.
- Spending multiple paragraphs rebutting an off-topic distillation discussion.
- Overloading the answer with source-count drift and pipeline implementation detail before answering “can weights be shared?”
- Calling the 0.87 plateau a label-quality ceiling without foregrounding that the label target is source-confounded.

### Prose scores

| Version | Clarity | Structure | Concision | Calibration |
|---|---:|---:|---:|---:|
| Sol | 9 | 8 | 9 | 9 |
| Proposed gold | 6 | 7 | 3 | 8 |

### Targeted prose edit

Use a four-part answer of roughly 800 words: **deployed model**, **training data/cost**, **evaluation and confound**, **sharing/status**. Put the fast-transformer and #7040 fix in a two-row “successors” table. End the sharing section with a direct sentence: “The corpus records no release decision; technical portability is not permission to redistribute.”

---

## Recommended revision order

1. `ablations`: most substantively incomplete; add current mixture optimization and test-integrity results first.
2. `muon`: add shipped/current evidence and fix routing wording.
3. `classifier`: add gen-3 audit/RFC while preserving the concise, conservative sharing answer.

All dispute checks above used only `evals/runners/mum-frozen`; answer files were not modified.
