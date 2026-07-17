# Handoff: applying GEPA to the marin-research skill

**Purpose:** brief a fresh agent (no prior conversation context) to (a) understand the eval
harness that already exists in `evals/`, and (b) execute the plan to improve the
`marin-research` skill with a GEPA-style reflective prompt optimizer.

Status as of 2026-07-16 · branch `evals-harness` · **nothing committed** (all of `evals/` is
untracked working tree). Author of prior work left off *before* starting GEPA — see
"Where this leaves off" at the bottom.

---

## 1. What already exists (the eval harness)

A regression/comparison harness for the Marin retrieval skills, built to answer DESIGN.md's
central bet: *can a fast skill match the slow `marin-research` on the same frozen corpus?*

- **Design spec:** `evals/DESIGN.md` — read §1 (principles), §2 (what an eval is), §4
  (scoring / the Q number), §5 (harness modes), §9 (auto-optimization — this is the GEPA hook).
- **Frozen corpus:** `evals/corpus/2026-07-14/` — a content-addressed snapshot: `corpus-index.db`
  (66,207 chunks), `corpus-manifest.json`, `summaries/` (15 weekly-summary HTML pages), and
  `FREEZE.json` (sha256s + counts). The DB and `summaries/` are gitignored (large, local);
  the manifest + FREEZE.json are tracked. **The frozen corpus = indexed DB chunks UNION the
  issue #s referenced by the frozen summaries** (this matters for the hallucination gate).
- **4 seed questions:** `evals/questions/{gpu,july,april,muon}.json` — each has must/nice/
  forbidden/recency citations, `facts` (atomic regexes, numbered Fact 1…n), machine-checkable
  `trap_checks`, and `gold_prose`. Golds for april/muon are hand-written; gpu/july were
  *composed from loop2 facts* (weaker — see stale-gold note below).
- **Deterministic scorer:** `evals/runners/score.py` — the heart. `score(question, answer_text,
  freeze_set, mode)` returns Q (0–100) + hard gates + every facet + rich per-fact/per-trap/
  per-citation diagnostics. **No LLM.** Reused by the viewer and (to be) by GEPA.
- **Candidate answers already collected & scored:**
  - `evals/baselines/opus48-repo/*.md` — Opus 4.8 pointed at *live* github (no marin skills). mode=`live`.
  - `evals/candidates/opus48-marin-research/*.md` — Opus 4.8 running the **marin-research skill on
    the frozen corpus**. mode=`frozen`. **This is the candidate GEPA will improve.**
- **Viewer:** generated to `/tmp/marin-evals.html` by `evals/runners/build_viewer.py`. Scored,
  self-contained, candidate tabs (Gold / baseline / marin-research) each driving a left scoring
  panel + right answer. `open /tmp/marin-evals.html` to view.

### The Q formula (the GEPA reward)
```
Q = 100·(0.35·cite_recall_MH + 0.35·fact_cov_MH + 0.30·trap_pass) − 10·(forbidden cited)
clamped 0–100; any hard-gate failure (hallucinated cite in frozen mode / wrong abstention) ⇒ Q=0
```
Nice-to-have citations, recency, and provenance are reported but NOT in Q.

---

## 2. Repo map (everything under `evals/`)
```
evals/
  DESIGN.md                       # the spec (read §4, §5, §9)
  gepa-handoff.md                 # this file
  questions/<id>.json             # the eval set (4 seeds)
  baselines/opus48-repo/<id>.md   # live-github baseline answers
  candidates/opus48-marin-research/<id>.md   # marin-research-on-frozen answers
  corpus/2026-07-14/              # frozen corpus (db + summaries gitignored; FREEZE.json tracked)
  runners/
    score.py                      # deterministic scorer + freeze_issue_set()  [the GEPA metric]
    build_questions.py            # regenerates questions/*.json
    build_viewer.py               # regenerates /tmp/marin-evals.html
    mum-frozen                    # `mum` pinned to the frozen corpus; refresh hard-blocked
```

---

## 3. How to run things

```bash
# score every candidate against the golds (prints the scoreboard)
python3 evals/runners/score.py

# rebuild the questions / the viewer after any edit
python3 evals/runners/build_questions.py
python3 evals/runners/build_viewer.py && open /tmp/marin-evals.html

# run mum against the FROZEN corpus (this is how the marin-research candidates were produced)
evals/runners/mum-frozen status
evals/runners/mum-frozen search-multi "GPU MFU ceiling" "Muon cost" --json
# NEVER use plain `mum` for eval runs (it hits the live mirror and may refresh). refresh is
# hard-blocked in the wrapper.
```

The marin-research candidates were produced by spawning one Opus 4.8 subagent per question,
each given **only the question** (no golds/facts/expected citations), told to follow
`mumwelt/skills/marin-research/SKILL.md` but substitute `evals/runners/mum-frozen` for every
`mum` call, single-process (no nested subagents), no refresh. That clean-room protocol is what
keeps the eval honest — preserve it.

---

## 4. Current scoreboard (Q, 0–100)

| question | Gold | Opus 4.8 baseline (live GH) | Opus 4.8 + marin-research (frozen) |
|---|---|---|---|
| july  | 100 | 78 | **94**  (cite 1.00, fact 0.83, trap 1.00) |
| muon  | 100 | 24 | **57** |
| april | 100 | 71 | **57**  (trap 0.33 — weak) |
| gpu   | 100 | 34 | **41**  (cite 0.00 — STALE GOLD, see §5) |

marin-research beats the naive baseline on july & muon, ~ties on gpu, loses on april.

---

## 5. ⚠️ CRITICAL: the golds are stale relative to the freeze

The golds were curated from a **Jun-28** loop2 pass; the freeze is **Jul-14**. `marin-research`
correctly follows the skill's "prefer the latest / close-out state" and reports the *newer*
picture from the frozen corpus, so it gets penalized by stale golds:

- **gpu** is the clearest case: gold says "9.37% MFU ceiling → flagship moved off GPU to TPU"
  (Jun-28). marin-research reports "**~23.8% MFU achieved, ramp-and-de-risk, Hopper back on the
  menu**" (Jul-14) and cites none of the gold's `#6304/#6493/#6292/#6044` → 0/4 citation recall.
  **That 0.00 is a gold defect, not a skill defect.** The gold may be *wrong* vs its own freeze.

**Do NOT run GEPA before re-vetting the golds against the 2026-07-14 freeze** (DESIGN §8 step 4).
Optimizing toward stale golds trains the skill to reproduce outdated answers. This is the hard
prerequisite. Start with gpu (most stale), then re-check july/april/muon issue numbers &
close-out states against `evals/runners/mum-frozen show <url>`.

---

## 6. The GEPA plan

### Why GEPA fits well here
GEPA (reflective / Pareto prompt optimizer, e.g. `dspy.GEPA`) is most effective when each
rollout returns **textual diagnostic feedback** to reflect on — not just a scalar. `score.py`
already emits exactly that: which facts missed (+regex+note), which must-have citations were
dropped, which traps failed and *why*, forbidden hits, provenance. And the artifact GEPA
evolves is a **text prompt** — which is what `mumwelt/skills/marin-research/SKILL.md` is.

### What to optimize
`SKILL.md` has two separable textual modules:
1. **Synthesis / fidelity-guard prompt** (§3–4: target-vs-achieved, platform attribution,
   close-out-over-proposal, quote-or-omit). **Optimize this first** — via `--mode replay`
   (fixed retrieval, only the synthesis prompt varies) it gives deterministic, cheap, fast
   rollouts. Near-term wins likely live here (april trap 0.33, muon history-citation recall).
2. **Orient / decompose prompt** (§1–2: harvest vocabulary from summaries, build sub-queries).
   Needs `--mode full` (real retrieval) — expensive; do later.

### Prerequisites (blockers)
1. **Re-vet golds vs the freeze** (§5). Hard blocker.
2. **Expand the eval set.** 4 questions overfit instantly. Need ~20–30 across DESIGN's classes
   (standard + unanswerable + recency-trap + platform-trap) with a **train/val split** —
   optimize on train, report on held-out val only.
3. **Harden Q against reward-hacking** (DESIGN §9). A gameable metric WILL be exploited: dumping
   every corpus `#issue` to max citation recall, or emitting literal fact-regex tokens. Guards:
   cap answer length / penalize over-citation, keep forbidden penalties, **hold out a subset of
   facts/citations** from the optimization signal to detect memorization, and NEVER let the
   LLM-judge into the loss (deterministic Q + hard gates only).

### How to wire it (recommended: thin custom loop, not raw dspy)
The skill is executed by a *tool-using agent*, not a clean DSPy LM graph, so wrapping it in
`dspy.GEPA`'s module abstraction is awkward. Instead run GEPA's shape as a small loop that
reuses our pieces unchanged:
```
candidate SKILL.md ─┐
                    ├─ run marin-research (replay mode) per eval instance → answer
                    ├─ score.py → Q + textual feedback
reflect(worst instances + their feedback) → propose edited SKILL.md
keep a Pareto frontier over eval INSTANCES (not just mean Q) → next generation
```
= reflective mutation + Pareto selection, with the SKILL.md text as the single evolvable
parameter, `score.py` as the metric+feedback, and an LLM reflection step reading the
lowest-Q transcripts to propose edits.

### Suggested order
1. Re-vet golds vs the freeze (blocker).
2. Grow eval set to ~20–30 with train/val split + new classes.
3. Add anti-gaming guards to `score.py`.
4. Build the replay-mode reflective loop; optimize the **synthesis prompt** first; report Q lift
   on held-out val vs the current `SKILL.md`. Only then attempt the retrieval/decompose module.

---

## 7. Where this leaves off / open decision for the user

The user asked "is there a way to apply GEPA to our skill to improve it?" — answer above is
yes, with §5 as the blocker. They were offered two immediate next steps and have NOT yet chosen:
- **(A)** re-vet the 4 golds against the 2026-07-14 freeze (the true prerequisite, start w/ gpu), or
- **(B)** stand up the step-4 reflective-loop skeleton now against the current 4 evals as a
  proof-of-plumbing (explicitly overfit-prone, not a real optimization).

Confirm which before proceeding. If (A): use `evals/runners/mum-frozen show/search` to check each
gold's must-have facts & citations still reflect the freeze's close-out state, and update
`build_questions.py` + regenerate. If (B): build the loop per §6 but label results as plumbing-only.
