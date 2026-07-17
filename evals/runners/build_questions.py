#!/usr/bin/env python3
"""Port the four loop2 seed questions into evals/questions/<id>.json (design §2)."""
import json, pathlib

ROOT = pathlib.Path("/Users/isaach/workspace/mumwelt")
QDIR = ROOT / "evals" / "questions"
QDIR.mkdir(parents=True, exist_ok=True)
GOLDS = pathlib.Path.home() / "workspace/marinmirror/scratch/loop2/golds"

CORPUS = "2026-07-16"

gpu_prose = """## How training on GPUs is going

GPU enablement is live but throughput is the blocker. CoreWeave H100 (`cw-us-east-02a`) is
up and running a ~90B-total/5.3B-active MoE on 256 H100s ([#6292](https://github.com/marin-community/marin/issues/6292)/[#6293](https://github.com/marin-community/marin/issues/6293)).

**MFU is the headline problem.** [#6304](https://github.com/marin-community/marin/issues/6304)
drove pure-FSDP MFU from 2.8% to **9.37%**, but ~9.4% is a **pure-FSDP ceiling** — well short of
the ~**25% target**. Memory/sharding is fragile: HBM-capped (batch ≥ 1536 OOMs), ~74 GiB pre-step
OOMs, and sharding crashes ([#6321](https://github.com/marin-community/marin/issues/6321)/[#6367](https://github.com/marin-community/marin/issues/6367)/[#6296](https://github.com/marin-community/marin/issues/6296)).

**Muon is expensive on GPU.** MuonH runs ~**4.75% MFU vs ~16% for SGD** — a heavy FSDP all-gather
tax ([#6493](https://github.com/marin-community/marin/issues/6493)/[#6534](https://github.com/marin-community/marin/issues/6534)) — which drove the pipeline-parallelism exploration.
PP hides the Muon tax and wins at multi-host TPU scale, but single-node FSDP wins on NVLink, so PP
was **not adopted on GPU** ([#6532](https://github.com/marin-community/marin/issues/6532)/[#6431](https://github.com/marin-community/marin/issues/6431)).
On kernels, the GPU MoE stack isn't cleanly ready; a Triton `ragged_dot` kernel replaced the TPU-only
Pallas GMM ([#2828](https://github.com/marin-community/marin/issues/2828)/[#4297](https://github.com/marin-community/marin/issues/4297)).

**Bottom line:** GPU throughput isn't competitive, so the flagship **67B-A2B 10T hero run was switched
OFF GPUs onto TPUs (v4)** ([#6044](https://github.com/marin-community/marin/issues/6044)).
"""

july_prose = """## Our July 2026 plan

The **"July hero run"** milestone was filed 2026-06-26 by ihodes (hero-run template, ~[#6700](https://github.com/marin-community/marin/issues/6700)–[#6716](https://github.com/marin-community/marin/issues/6716)).

**The run.** Centerpiece is a **67B-total / ~2B-active MoE on ~10T tokens** (d=2560), running on
**TPU v4-2048** — moved off GPU ([#6044](https://github.com/marin-community/marin/issues/6044)/[#6704](https://github.com/marin-community/marin/issues/6704)).

**Architecture & long context.** Extend context to **262k via two-stage YaRN** ([#6701](https://github.com/marin-community/marin/issues/6701)/[#6047](https://github.com/marin-community/marin/issues/6047)).

**Data.** Pretraining mix + CoreWeave pipeline ([#6713](https://github.com/marin-community/marin/issues/6713)/[#6700](https://github.com/marin-community/marin/issues/6700)), SFT ([#6714](https://github.com/marin-community/marin/issues/6714)), and RL data ([#6707](https://github.com/marin-community/marin/issues/6707)).

**Preregistered loss.** Stage-1 **target ~2.269 paloma macro** over the first 8T tokens ([#6702](https://github.com/marin-community/marin/issues/6702)).

**Perf gate.** Get **B200 MFU above ~20% before the run** (commitment) ([#6706](https://github.com/marin-community/marin/issues/6706)/[#6710](https://github.com/marin-community/marin/issues/6710)).

**Sequencing.** The final 10T post-training **slips to early August** ([#6705](https://github.com/marin-community/marin/issues/6705)); the next hero run (nB-AmB XT) targets **EOY at 256–500B** ([#6689](https://github.com/marin-community/marin/issues/6689)).
"""

Q = {
  "gpu": {
    "id": "gpu",
    "question": "explain how training on GPUs is going",
    "corpus": CORPUS,
    "answerable": True,
    "skill_hint": "research",
    "source": "loop2/config.json:questions.gpu (facts F1-F7)",
    "dims": [
      "cluster bringup / enablement",
      "throughput & MFU (achieved vs target)",
      "memory / OOM / sharding fragility",
      "optimizer (Muon) cost",
      "parallelism strategy (FSDP vs pipeline)",
      "kernels / MoE software-stack readiness",
      "which hardware the flagship 67B run actually uses",
    ],
    "citations": {
      "must_have": ["#6304", "#6493", "#6292", "#6044"],
      "nice_to_have": ["#6293", "#6321", "#6367", "#6296", "#6532", "#6534", "#6431", "#2828", "#4297"],
      "forbidden": ["#5167"],
      "recency_critical": ["#6044"],
    },
    "facts": {
      "must_have": [
        {"id": "F1", "match": r"CoreWeave|256\s*H100", "note": "CoreWeave H100 bringup, MoE on 256 H100"},
        {"id": "F2", "match": r"9\.37%|9\.4%", "note": "pure-FSDP MFU ceiling"},
        {"id": "F2b", "match": r"25%", "note": "the ~25% is a TARGET, not achieved"},
        {"id": "F4", "match": r"4\.75%|MuonH", "note": "Muon MFU tax vs ~16% SGD"},
        {"id": "F7", "match": r"TPU|v4", "note": "67B flagship moved OFF GPU onto TPU"},
      ],
      "nice_to_have": [
        {"id": "F3", "match": r"OOM|HBM|shard", "note": "memory/sharding fragility"},
        {"id": "F6", "match": r"Triton|ragged_dot", "note": "GPU MoE kernel stack"},
      ],
    },
    "traps": {
      "target_vs_achieved": "~25% MFU is a TARGET; 9.37% is the achieved pure-FSDP close-out",
      "platform": "report the H100/GPU result, do not report a TPU MFU as the GPU number",
      "superseded": ["#5167 optimizer study is not the GPU-MFU answer"],
    },
    "trap_checks": {
      "target_vs_achieved": {"require": [r"9\.37%|9\.4%", r"25%"]},
      "platform": {"require": [r"TPU|v4"]},
      "superseded": {"forbid_cite": ["5167"]},
    },
    "gold_prose": gpu_prose,
  },

  "july": {
    "id": "july",
    "question": "explain our july 2026 plan",
    "corpus": CORPUS,
    "answerable": True,
    "skill_hint": "research",
    "source": "loop2/config.json:questions.july (facts G1-G7)",
    "dims": [
      "what the milestone is (name / when / who filed it)",
      "the hero-run model: scale & which hardware it runs on",
      "architecture & long-context plan (target length and method)",
      "data workstreams (pretrain / SFT / RL)",
      "loss preregistration target (the number)",
      "MFU / perf gate before the run",
      "schedule / sequencing & the NEXT hero run",
    ],
    "citations": {
      "must_have": ["#6704", "#6044", "#6701", "#6702", "#6706", "#6705", "#6689"],
      "nice_to_have": ["#6700", "#6047", "#6713", "#6714", "#6707", "#6710", "#6716"],
      "forbidden": [],
      "recency_critical": ["#6705"],
    },
    "facts": {
      "must_have": [
        {"id": "G2a", "match": r"67B|67\s*B", "note": "67B-total MoE"},
        {"id": "G2b", "match": r"TPU|v4", "note": "runs on TPU v4-2048, moved off GPU"},
        {"id": "G3", "match": r"262k|262K|YaRN", "note": "262k context via two-stage YaRN"},
        {"id": "G5", "match": r"2\.269", "note": "preregistered stage-1 loss target"},
        {"id": "G6", "match": r"B200", "note": "B200 MFU gate before the run"},
        {"id": "G7", "match": r"August|Aug", "note": "final 10T slips to early August"},
      ],
      "nice_to_have": [
        {"id": "G1", "match": r"2026-06-26|hero run", "note": "milestone name / filed date"},
        {"id": "G4", "match": r"SFT|RL", "note": "data workstreams"},
      ],
    },
    "traps": {
      "platform": "hero run is on TPU v4-2048 (moved OFF GPU); do not report GPU as the run hardware",
      "temporal": "report the close-out plan including the August slip, not a mid-thread proposal",
      "target_vs_achieved": "~2.269 loss and B200>20% are TARGETS/commitments, not achieved results",
    },
    "trap_checks": {
      "platform": {"require": [r"TPU|v4"]},
      "temporal": {"require": [r"Aug"]},
      "target_vs_achieved": {"require": [r"2\.269"]},
    },
    "gold_prose": july_prose,
  },

  "april": {
    "id": "april",
    "question": "how did we do on our april 2026 milestone?",
    "corpus": CORPUS,
    "answerable": True,
    "skill_hint": "research",
    "source": "loop2/config.json:questions.april (facts X1-X7) + golds/april.md",
    "dims": [
      "what the April milestone was (scope/goals)",
      "infra goals & their status",
      "modeling goals & their status",
      "the flagship pretrain (100B-A13B 1.2T) & its status",
      "what was achieved / shipped",
      "what was missed / slipped",
      "overall how it went",
    ],
    "citations": {
      "must_have": ["#4266", "#4256", "#4281", "#3800", "#4297", "#4269", "#4270", "#4271"],
      "nice_to_have": ["#4268", "#4272", "#4273", "#4282", "#4283", "#5212", "#5290", "#5370"],
      "forbidden": [],
      "recency_critical": ["#4270", "#4271"],
    },
    "facts": {
      "must_have": [
        {"id": "X1a", "match": r"100B-A13B|100B", "note": "the 100B-A13B 1.2T MoE milestone"},
        {"id": "X1b", "match": r"1\.2T", "note": "1.2T tokens"},
        {"id": "X3", "match": r"2\.432", "note": "1e22 run paloma macro_loss"},
        {"id": "X4", "match": r"8\.8%|6\.04%", "note": "Triton kernel end-to-end MFU gain (achieved)"},
        {"id": "X5", "match": r"[Oo]ff-?[Rr]ay|import ray", "note": "off-Ray achieved"},
        {"id": "X7a", "match": r"4270|canary", "note": "canary-90% target unresolved (missed)"},
        {"id": "X7b", "match": r"4271|as-a-library|library", "note": "marin-as-a-library unresolved (missed)"},
      ],
      "nice_to_have": [
        {"id": "X2", "match": r"1e23", "note": "1e23 launched + preregistered"},
        {"id": "X6", "match": r"log-service|log service", "note": "observability shipped"},
      ],
    },
    "traps": {
      "target_vs_achieved": "MFU #4283 v4 25-30% and 'match Megatron' were TARGETS; the +8.8% (5.55->6.04%) kernel gain is what was ACHIEVED",
      "platform": "the 1e22 run #3800 is TPU v4-512; the kernel MFU gain is on GPU H100x8 - attribute each correctly",
      "completeness": "must surface the MISSED items (#4270 canary, #4271 library), not only the wins",
    },
    "trap_checks": {
      "target_vs_achieved": {"require": [r"8\.8%|6\.04%"]},
      "platform": {"require": [r"TPU|v4", r"H100"]},
      "completeness": {"require": [r"canary", r"librar"]},
    },
    "gold_prose": (GOLDS / "april.md").read_text().strip(),
  },

  "muon": {
    "id": "muon",
    "question": "what is our current muon approach, and how did we end up here?",
    "corpus": CORPUS,
    "answerable": True,
    "skill_hint": "research",
    "source": "loop2/config.json:questions.muon (facts X1-X7) + golds/muon.md",
    "dims": [
      "the current chosen optimizer (MuonH config: which leaves, AdamH where, when decided)",
      "early verification: Muon's critical batch size vs AdamW (sqrt-LR scaling)",
      "negative results tried (e.g. MuonRemez partial orthogonalization)",
      "data-dependent findings (nano-walks: Muon vs AdamH crossover)",
      "architecture refinements (magnitude-direction decoupling gate results)",
      "the GPU bottleneck (single-node FSDP / Newton-Schulz cost)",
      "the pivot and what's next (pipeline parallelism, delayed-gradient staleness)",
    ],
    "citations": {
      "must_have": ["#5167", "#5134", "#1565", "#2284", "#5585", "#6388", "#6493", "#6431"],
      "nice_to_have": ["#1293", "#4281", "#6492", "#6423"],
      "forbidden": [],
      "recency_critical": ["#6431", "#6493"],
    },
    "facts": {
      "must_have": [
        {"id": "X1a", "match": r"MuonH", "note": "chosen optimizer is MuonH (matrix/expert leaves)"},
        {"id": "X1b", "match": r"AdamH", "note": "AdamH on embeddings/lm_head"},
        {"id": "X1c", "match": r"2026-06-06", "note": "decided 2026-06-06"},
        {"id": "X2", "match": r"critical batch|sqrt-LR", "note": "larger critical batch size vs AdamW"},
        {"id": "X3", "match": r"MuonRemez", "note": "MuonRemez negative result"},
        {"id": "X6", "match": r"Newton-Schulz|1\.49s", "note": "NS step-tail cost on GPU"},
        {"id": "X7", "match": r"pipeline parallel", "note": "reluctant pivot to pipeline parallelism"},
      ],
      "nice_to_have": [
        {"id": "X4", "match": r"nemotron|nano walk", "note": "nano-walk data-swap crossover"},
        {"id": "X5", "match": r"magnitude-direction|d512", "note": "magnitude-direction decoupling gate"},
      ],
    },
    "traps": {
      "temporal": "the current decision is MuonH (optimizer studies closed 2026-06-06); do not report an earlier candidate as current",
      "platform": "the Newton-Schulz step-tail bottleneck is GPU-specific; the how-we-got-here history is mostly TPU",
      "superseded": ["MuonRemez (#2284) was tried and rejected - not the current approach"],
    },
    "trap_checks": {
      "temporal": {"require": [r"MuonH"], "require_cite": ["5167"]},
      "platform": {"require": [r"Newton-Schulz"]},
      "superseded": {"require": [r"MuonRemez"]},
    },
    "gold_prose": (GOLDS / "muon.md").read_text().strip(),
  },
}

for qid, obj in Q.items():
    # renumber facts sequentially 1..n (must-have first, then nice-to-have)
    n = 1
    facts = obj.get("facts", {})
    for bucket in ("must_have", "nice_to_have"):
        for f in facts.get(bucket, []):
            f["id"] = str(n)
            n += 1
    p = QDIR / f"{qid}.json"
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {p}  ({p.stat().st_size} bytes)")
