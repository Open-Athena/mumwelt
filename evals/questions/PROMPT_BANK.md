# Eval prompt bank

Example prompts for new evals. The four seeds (`gpu`, `july`, `april`, `muon`) already have
question records under `evals/questions/`. Prompts below are candidates for new eval questions —
each becomes a `questions/<id>.json` (schema in `DESIGN.md` §2) once a corpus-grounded gold is
curated against the active freeze.

## Added 2026-07-16 (batch 1)

1. **ablations** — Can you share more details about the data ablations you've run so far? What
   data mixes / data classifiers were compared, on what training-data sizes and test data?
2. **classifier** — What is the data-classifier model and how was it trained and evaluated?
   (Assume https://github.com/marin-community/marin/issues/5810.) Can the model weights be shared?
3. **benchmarks** — What are the target benchmarks and the development proxies?
4. **h100-67b** — What happened when we tried to bring the 67B A2B up on H100s? What did we try?
   What worked, what didn't, and what were the final MFUs?
5. **inference** — Who is doing inference work and where is it at? What's the current performance
   and are there any open issues?
