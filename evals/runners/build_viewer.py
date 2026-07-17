#!/usr/bin/env python3
"""Render evals/questions/*.json + candidate answers into a scored /tmp HTML viewer.

Candidate tabs at the top of each question drive a
left SCORING panel (Q + per-category pass/fail, from runners/score.py) and a
right ANSWER panel. Self-contained: opens straight from file://.
"""
import json, pathlib, re, html, sys

ROOT = pathlib.Path("/Users/isaach/workspace/mumwelt")
QDIR = ROOT / "evals" / "questions"
BDIR = ROOT / "evals" / "baselines" / "opus48-repo"
MRDIR = ROOT / "evals" / "candidates" / "opus48-marin-research"
SOLDIR = ROOT / "evals" / "sol-answers"
sys.path.insert(0, str(ROOT / "evals" / "runners"))
import score as scorer  # noqa: E402

BASELINE_DATE = "2026-07-16"
OUT = pathlib.Path("/tmp/marin-evals.html")
GH = "https://github.com/marin-community/marin/issues/"
ORDER = ["gpu", "h100-67b", "muon", "july", "april",
         "ablations", "classifier", "benchmarks", "inference"]
FREEZE = scorer.freeze_issue_set(ROOT / "evals/corpus/2026-07-16/corpus-index.db",
                                 ROOT / "evals/corpus/2026-07-16/summaries")

# candidate sources: (key, label, mode, dir-or-None-for-gold, meta-line)
CANDIDATE_SOURCES = [
    ("gold", "Gold answer", "reference", None,
     "Human-curated reference · corpus-grounded"),
    ("baseline", "Opus 4.8 baseline", "live", BDIR,
     f"Opus 4.8 · live github.com/marin-community/marin · no marin skills · {BASELINE_DATE}"),
    ("marin-research", "Opus 4.8 + marin-research", "frozen", MRDIR,
     f"Opus 4.8 · marin-research skill · frozen 2026-07-16 corpus · {BASELINE_DATE}"),
    ("sol", "Sol independent", "frozen", SOLDIR,
     f"Sol · max-depth subagent research · frozen 2026-07-16 corpus · {BASELINE_DATE}"),
]

TIPS = {
    "q": ("Deterministic quality score, 0–100.  Q = 100·(0.35·cite-recall + "
          "0.35·fact-coverage + 0.30·trap-pass), minus 10 for each forbidden citation, "
          "clamped to 0–100. A hard-gate failure forces Q=0. Nice-to-have citations and "
          "recency are reported but NOT part of Q."),
    "gates": ("Pass/fail checks that override the score. A hallucinated citation (frozen "
              "mode) or a wrong abstention forces Q=0 regardless of the components."),
    "must": "Issues the answer is expected to cite. Recall over this set = 35% of Q.",
    "nice": ("Bonus citations — a richness / tiebreak signal only. Deliberately NOT part of Q, "
             "so a candidate can't trade one must-have for two nice-to-haves."),
    "recency": ("The subset of must-have citations carrying the latest / close-out state — the "
                "'retrieved date' metric. Reported alongside Q, not separately weighted."),
    "forbidden": ("Issues that are wrong, superseded, or off-platform for this question. Citing "
                  "one means the wrong source was pulled — each costs −10 points. It is the "
                  "machine-checkable half of a 'superseded' trap."),
    "prov": ("Whether each cited issue exists inside the frozen 2026-07-16 corpus. 'Post-freeze' "
             "cites are newer than the freeze: tolerated in live mode, counted as hallucinations "
             "in frozen mode."),
    "facts": ("Atomic facts the answer should state (Fact 1…n per question), each verified by a "
              "regex over the answer text. Must-have coverage = 35% of Q. Each fact is scored "
              "independently, so a fact is split whenever one claim has two checkable parts."),
    "traps": ("Known ways to answer this question subtly wrong (target-vs-achieved, platform, "
              "temporal, superseded). Each has a deterministic require/forbid check; the fraction "
              "passed = 30% of Q."),
    "mode": ("reference = the gold itself (gates off).  live = candidate hit the live repo "
             "(post-freeze cites tolerated).  frozen = candidate must work off the frozen corpus "
             "(post-freeze cites count as hallucinations)."),
}


def info(key, right=False):
    return (f'<span class="info{" r" if right else ""}" tabindex="0" role="button" '
            f'aria-label="explain" data-tip="{html.escape(TIPS[key])}">i</span>')


# ---------- markdown ----------
def md_to_html(md):
    out = []
    for block in re.split(r"\n\s*\n", md.strip()):
        block = block.strip()
        if not block:
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", block)
        if re.match(r"^-{3,}$", block) or block == "***":
            out.append("<hr>")
        elif m:
            lvl = min(len(m.group(1)) + 2, 6)
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
        elif all(l.lstrip().startswith(">") for l in block.splitlines()):
            inner = " ".join(l.lstrip()[1:].strip() for l in block.splitlines())
            out.append(f'<blockquote class="prov">{inline(inner)}</blockquote>')
        elif block.startswith("- ") or block.startswith("* "):
            items = "".join(f"<li>{inline(l[2:])}</li>"
                            for l in block.splitlines() if l.strip()[:2] in ("- ", "* "))
            out.append(f"<ul>{items}</ul>")
        else:
            out.append(f"<p>{inline(block)}</p>")
    return "\n".join(out)


def inline(s):
    s = html.escape(s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)",
               r'<a href="\2" target="_blank" rel="noopener">\1</a>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*(?!\*)([^*]+)\*(?!\*)", r"<em>\1</em>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"(?<![\w/#])#(\d{3,5})\b",
               rf'<a href="{GH}\1" target="_blank" rel="noopener">#\1</a>', s)
    return s


def chip(n, cls):
    n = n.lstrip("#")
    return f'<a class="chip {cls}" href="{GH}{n}" target="_blank" rel="noopener">#{n}</a>'


def band(q):
    return "good" if q >= 80 else ("warn" if q >= 40 else "bad")


def load_candidate(path):
    if not path.exists():
        return None
    raw = path.read_text()
    note = ""
    m = re.search(r"<!--\s*NOTE:\s*(.*?)-->", raw, re.S)
    if m:
        note = m.group(1).strip()
    body = re.sub(r"<!--.*?-->", "", raw, flags=re.S).strip()
    return {"body": body, "note": note}


# ---------- left scoring panel ----------
def render_scores(q, s):
    comp = s["Q_components"]
    cit = s["citations"]

    def cchips(key, hitcls="hit", misscls="miss"):
        d = cit[key]
        parts = [chip(n, hitcls) for n in d.get("hit", [])]
        parts += [chip(n, misscls) for n in d.get("miss", [])]
        return "".join(parts) or '<span class="empty">—</span>'

    # gates
    grows = []
    for name, g in s["gates"].items():
        if name == "hallucinated_citation":
            if s["mode"] == "frozen":
                ok = not g["fail"]
                txt = "all cites in freeze" if ok else f'{len(g["detail"])} not in freeze'
            else:
                ok = True
                n = len(cit["provenance"]["out_of_freeze"])
                txt = "n/a (reference)" if s["mode"] == "reference" else \
                      (f'{n} cite(s) post-freeze — tolerated (live)' if n else "all cites in freeze")
        else:
            ok = not g["fail"]
            txt = g["detail"]
        grows.append(
            f'<div class="grow"><span class="dot {"ok" if ok else "no"}"></span>'
            f'<span class="gname">{html.escape(name.replace("_"," "))}</span>'
            f'<span class="gdet">{html.escape(txt)}</span></div>')

    # facts
    def factrows(rows):
        if not rows:
            return '<span class="empty">—</span>'
        return "".join(
            f'<div class="fact"><span class="dot {"ok" if r["pass"] else "no"}"></span>'
            f'<span class="fid">Fact&nbsp;{html.escape(r["id"])}</span>'
            f'<code class="rx">{html.escape(r["match"])}</code>'
            f'<span class="note">{html.escape(r["note"])}</span></div>' for r in rows)

    # traps
    trap_parts = []
    for t in s["traps"]:
        why = (f'<span class="twhy">✗ {html.escape(t["why"])}</span>'
               if not t["pass"] and t["why"] else "")
        trap_parts.append(
            f'<div class="trap"><span class="dot {"ok" if t["pass"] else "no"}"></span>'
            f'<div class="tbody"><span class="tname">{html.escape(t["name"].replace("_"," "))}</span>'
            f'<span class="tdesc">{html.escape(t["desc"])}</span>{why}</div></div>')
    traprows = "".join(trap_parts)

    prov = cit["provenance"]
    prov_extra = "".join(chip(n, "extra") for n in prov["out_of_freeze"])
    fb = cit["forbidden_cited"]
    fb_html = ("".join(chip(n, "fbhit") for n in fb) if fb
               else '<span class="ok-inline">none ✓</span>')

    qv = s["Q"]
    formula = (f'100·(0.35×{comp["cite_recall_MH"]:.2f} + '
               f'0.35×{comp["fact_cov_MH"]:.2f} + '
               f'0.30×{comp["trap_pass"]:.2f})'
               + (f' − {comp["forbidden_penalty"]}' if comp["forbidden_penalty"] else '')
               + (' → gate&nbsp;fail ⇒ 0' if s["gate_failed"] else ''))

    return f"""
    <div class="panel qhero {band(qv)}">
      <div class="qbig"><span class="qnum">{qv:.0f}</span><span class="qmax">/100</span>
        <span class="qlabel">deterministic&nbsp;Q {info('q', right=True)}</span></div>
      <div class="qformula">{formula}</div>
      <div class="qbars">
        <div class="qbar"><span>cite&nbsp;recall</span><i style="--v:{comp['cite_recall_MH']*100:.0f}%"></i><b>{comp['cite_recall_MH']:.2f}</b></div>
        <div class="qbar"><span>fact&nbsp;coverage</span><i style="--v:{comp['fact_cov_MH']*100:.0f}%"></i><b>{comp['fact_cov_MH']:.2f}</b></div>
        <div class="qbar"><span>trap&nbsp;pass</span><i style="--v:{comp['trap_pass']*100:.0f}%"></i><b>{comp['trap_pass']:.2f}</b></div>
      </div>
    </div>

    <div class="panel">
      <h3>Hard gates {info('gates')}</h3>{"".join(grows)}
    </div>

    <div class="panel">
      <h3>Citations</h3>
      <div class="crow"><span class="lbl mh">must-have {info('must')} <b>{len(cit['must_have']['hit'])}/{len(cit['must_have']['hit']+cit['must_have']['miss'])}</b></span><div class="chips">{cchips('must_have')}</div></div>
      <div class="crow"><span class="lbl rc">recency-critical {info('recency')} <b>{len(cit['recency_critical']['hit'])}/{len(cit['recency_critical']['hit']+cit['recency_critical']['miss'])}</b></span><div class="chips">{cchips('recency_critical')}</div></div>
      <div class="crow"><span class="lbl nh">nice-to-have {info('nice')} <b>{len(cit['nice_to_have']['hit'])}</b></span><div class="chips">{"".join(chip(n,'hit') for n in cit['nice_to_have']['hit']) or '<span class="empty">—</span>'}</div></div>
      <div class="crow"><span class="lbl fb">forbidden {info('forbidden')}</span><div class="chips">{fb_html}</div></div>
      <div class="crow prov"><span class="lbl">provenance {info('prov')}</span><span class="provtxt">in freeze <b>{len(prov['in_freeze'])}</b> · post-freeze (live) <b>{len(prov['out_of_freeze'])}</b></span></div>
      {f'<div class="crow"><span class="lbl">post-freeze cites</span><div class="chips">{prov_extra}</div></div>' if prov['out_of_freeze'] else ''}
    </div>

    <div class="panel">
      <h3>Facts {info('facts')} <span class="sub">(regex over answer · {s['facts']['cov_MH']*100:.0f}% must-have)</span></h3>
      <div class="glabel mh">must-have</div>{factrows(s['facts']['must_have'])}
      <div class="glabel nh">nice-to-have</div>{factrows(s['facts']['nice_to_have'])}
    </div>

    <div class="panel">
      <h3>Traps {info('traps')}</h3>{traprows or '<span class="empty">—</span>'}
    </div>

    <div class="panel">
      <h3>Dimensions <span class="sub">(coverage guide · not scored)</span></h3>
      <ul class="dims">{"".join(f"<li>{html.escape(d)}</li>" for d in q.get('dims', []))}</ul>
    </div>"""


def render_answer(q, cand, s, meta_line, prose_md, note=None):
    note_html = f'<div class="bnote">{inline(note)}</div>' if note else ""
    return f"""
    <div class="panel answer">
      <div class="bmeta">{html.escape(meta_line)}</div>
      <div class="prose">{md_to_html(prose_md)}</div>
      {note_html}
    </div>"""


def render_candidate(q, cand, s, left_scores, right_answer):
    return f"""
    <div class="candidate{' active' if cand=='gold' else ''}" data-cand="{cand}">
      <div class="grid">
        <div class="col">{left_scores}</div>
        <div class="col">{right_answer}</div>
      </div>
    </div>"""


def render_question(q):
    cands = []
    for key, label, mode, d, meta in CANDIDATE_SOURCES:
        if d is None:                      # gold
            prose, note = q["gold_prose"], None
        else:
            c = load_candidate(d / f"{q['id']}.md")
            if not c:
                continue
            prose, note = c["body"], c["note"]
        s = scorer.score(q, prose, FREEZE, mode=mode)
        cands.append((key, label, s, meta, prose, note))

    tabs = "".join(
        f'<button class="ctab{" active" if cand=="gold" else ""}" data-cand="{cand}">'
        f'{html.escape(label)}<span class="qpill {band(s["Q"])}">Q {s["Q"]:.0f}</span></button>'
        for cand, label, s, *_ in cands)

    blocks = "".join(
        render_candidate(q, cand, s, render_scores(q, s),
                         render_answer(q, cand, s, meta, prose, note))
        for cand, label, s, meta, prose, note in cands)

    ans = q["answerable"]
    return f"""
<section class="q" id="q-{q['id']}" data-id="{q['id']}">
  <header class="qhead">
    <h2>{html.escape(q['question'])}</h2>
    <div class="meta">
      <span class="tag id">{html.escape(q['id'])}</span>
      <span class="tag">route: {html.escape(q.get('skill_hint','?'))}</span>
      <span class="tag">corpus {html.escape(q['corpus'])}</span>
      <span class="tag {'yes' if ans else 'no'}">{'answerable' if ans else 'UNANSWERABLE'}</span>
    </div>
  </header>
  <div class="ctabs">{tabs}<span class="ctabinfo">candidate {info('mode')}</span></div>
  {blocks}
</section>"""


questions = [json.loads((QDIR / f"{qid}.json").read_text()) for qid in ORDER]
tabbar = "".join(
    f'<button class="tab{" active" if i==0 else ""}" data-target="{q["id"]}">{html.escape(q["id"])}</button>'
    for i, q in enumerate(questions))
sections = "\n".join(render_question(q) for q in questions)


# ---------- top at-a-glance chart: 3 runs × every eval ----------
def _q3(q):
    r = {}
    for key, label, mode, d, meta in CANDIDATE_SOURCES:
        if d is None:
            prose = q["gold_prose"]
        else:
            c = load_candidate(d / f"{q['id']}.md")
            prose = c["body"] if c else None
        r[key] = scorer.score(q, prose, FREEZE, mode=mode)["Q"] if prose else None
    return r


_chart = [(q["id"], _q3(q)) for q in questions]


def _bar(cls, v):
    if v is None:
        return f'<i class="bar {cls} na" title="n/a"></i>'
    return f'<i class="bar {cls}" style="--h:{v:.0f}%" title="{v:.0f}"><b>{v:.0f}</b></i>'


_grid = "".join(f'<div class="gridline" style="bottom:{p}%"><span>{p}</span></div>'
                for p in (0, 25, 50, 75, 100))
# chart shows the three researched runs (gold omitted — always 100)
_groups = "".join(
    f'<button class="grp" data-target="{qid}" title="{html.escape(qid)}">'
    f'<span class="cols">{_bar("mr", s["marin-research"])}{_bar("sol", s["sol"])}'
    f'{_bar("base", s["baseline"])}</span>'
    f'<span class="xlabel">{html.escape(qid)}</span></button>' for qid, s in _chart)


def _mean(key):
    xs = [s[key] for _, s in _chart if s.get(key) is not None]
    return sum(xs) / len(xs) if xs else 0


chart_html = (
    '<div class="chart panel"><div class="chart-head"><span class="ttl">Q by eval — at a glance</span>'
    '<span class="lg"><i class="s-mr"></i>skill + frozen</span>'
    '<span class="lg"><i class="s-sol"></i>Sol (independent)</span>'
    '<span class="lg"><i class="s-base"></i>no-skill + live</span>'
    f'<span class="chmean">mean · skill {_mean("marin-research"):.0f} · '
    f'Sol {_mean("sol"):.0f} · baseline {_mean("baseline"):.0f}</span>'
    f'</div><div class="chart-scroll"><div class="chart-plot">{_grid}'
    f'<div class="bars">{_groups}</div></div></div></div>')

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1a1d21;--muted:#6b7280;--line:#e5e7eb;
--accent:#2563eb;--mh:#dc2626;--nh:#0891b2;--rc:#7c3aed;--fb:#b45309;
--yes:#059669;--no:#dc2626;--code:#f1f3f5;}
@media(prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#171a20;--ink:#e6e8eb;--muted:#9aa2ad;
--line:#282d36;--accent:#60a5fa;--mh:#f87171;--nh:#38bdf8;--rc:#c084fc;--fb:#fbbf24;
--yes:#34d399;--no:#f87171;--code:#0c0e12;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1240px;margin:0 auto;padding:26px 22px 90px}
.top{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:4px}
.top h1{font-size:20px;margin:0;letter-spacing:-.01em}
.top .note{color:var(--muted);font-size:13px}
.top .freeze{margin-left:auto;font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.tabs{display:flex;gap:6px;position:sticky;top:0;z-index:6;padding:12px 0;
background:linear-gradient(var(--bg),var(--bg) 72%,transparent)}
.tab{border:1px solid var(--line);background:var(--card);color:var(--ink);padding:7px 16px;
border-radius:999px;cursor:pointer;font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.03em}
.tab:hover{border-color:var(--accent)}
.tab.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.q{display:none}.q.show{display:block}
.qhead h2{margin:8px 0 10px;font-size:22px;letter-spacing:-.01em}
.meta{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
.tag{font-size:11.5px;font-weight:700;padding:3px 9px;border-radius:6px;background:var(--code);
color:var(--muted);text-transform:uppercase;letter-spacing:.03em}
.tag.id{background:var(--accent);color:#fff}
.tag.yes{color:#fff;background:var(--yes)}.tag.no{color:#fff;background:var(--no)}
.ctabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px;border-bottom:1px solid var(--line);padding-bottom:0}
.ctab{border:1px solid var(--line);border-bottom:none;background:var(--card);color:var(--muted);
padding:9px 16px;border-radius:10px 10px 0 0;cursor:pointer;font-size:13.5px;font-weight:700;
display:inline-flex;align-items:center;gap:9px;position:relative;top:1px}
.ctab:hover{color:var(--ink)}
.ctab.active{color:var(--ink);border-color:var(--line);background:var(--bg);border-bottom:2px solid var(--bg)}
.qpill{font-size:11px;font-weight:800;padding:2px 8px;border-radius:999px;color:#fff;font-variant-numeric:tabular-nums;text-shadow:0 1px 1px rgba(0,0,0,.30)}
.qpill.good{background:#15803d}.qpill.warn{background:#b45309}.qpill.bad{background:#b91c1c}
/* top at-a-glance chart */
.chart{padding:15px 20px 12px;margin-bottom:16px}
.chart-head{display:flex;gap:16px;align-items:center;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-bottom:16px}
.chart-head .ttl{font-weight:700;text-transform:uppercase;letter-spacing:.05em;font-size:11px;color:var(--ink)}
.chart-head .lg{display:inline-flex;align-items:center;gap:6px}
.chart-head .lg i{width:11px;height:11px;border-radius:3px;display:inline-block}
.chart-head .chmean{margin-left:auto;font-variant-numeric:tabular-nums;font-weight:600}
.s-gold{background:#10b981}.s-mr{background:#3b82f6}.s-sol{background:#8b5cf6}.s-base{background:#f59e0b}
.chart-scroll{overflow-x:auto}
.chart-plot{position:relative;height:184px;min-width:660px;border-bottom:2px solid var(--line);margin:0 0 24px 26px}
.gridline{position:absolute;left:0;right:0;height:0;border-top:1px dashed var(--line)}
.gridline span{position:absolute;left:-8px;top:-7px;transform:translateX(-100%);font-size:9px;color:var(--muted);font-variant-numeric:tabular-nums}
.bars{position:absolute;inset:0;display:flex;gap:8px;align-items:flex-end}
.grp{position:relative;flex:1 1 0;min-width:46px;display:flex;align-items:flex-end;justify-content:center;height:100%;background:none;border:0;cursor:pointer;padding:0}
.grp .cols{display:flex;gap:4px;align-items:flex-end;height:100%;width:100%;justify-content:center}
.bar{width:20px;height:var(--h);min-height:3px;border-radius:3px 3px 0 0;position:relative}
.bar.gold{background:#10b981}.bar.mr{background:#3b82f6}.bar.sol{background:#8b5cf6}.bar.base{background:#f59e0b}
.bar.na{background:repeating-linear-gradient(45deg,var(--line),var(--line) 3px,transparent 3px,transparent 6px);height:100%!important;opacity:.5}
.bar b{position:absolute;top:-15px;left:50%;transform:translateX(-50%);font-size:9.5px;font-weight:800;color:var(--ink);font-variant-numeric:tabular-nums}
.grp:hover .bar{filter:brightness(1.14)}
.grp .xlabel{position:absolute;bottom:-21px;left:0;right:0;text-align:center;font-size:10px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.01em;white-space:nowrap}
.grp:hover .xlabel{color:var(--accent)}
.grp.sel .xlabel{color:var(--accent)}
.candidate{display:none}.candidate.active{display:block}
.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px}
@media(max-width:940px){.grid{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:15px 17px;margin-bottom:15px}
.panel h3{margin:0 0 11px;font-size:12.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.panel h3 .sub{text-transform:none;letter-spacing:0;font-weight:400;font-size:11.5px}
/* Q hero */
.qhero{border-width:1px}
.qhero.good{border-color:color-mix(in srgb,var(--yes) 45%,var(--line))}
.qhero.warn{border-color:color-mix(in srgb,var(--fb) 45%,var(--line))}
.qhero.bad{border-color:color-mix(in srgb,var(--no) 45%,var(--line))}
.qbig{display:flex;align-items:baseline;gap:7px}
.qnum{font-size:46px;font-weight:800;letter-spacing:-.03em;line-height:1}
.qhero.good .qnum{color:var(--yes)}.qhero.warn .qnum{color:var(--fb)}.qhero.bad .qnum{color:var(--no)}
.qmax{font-size:17px;color:var(--muted);font-weight:600}
.qlabel{margin-left:auto;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:700}
.qformula{font:12px ui-monospace,Menlo,monospace;color:var(--muted);margin:9px 0 12px}
.qbars{display:flex;flex-direction:column;gap:6px}
.qbar{display:grid;grid-template-columns:96px 1fr 34px;align-items:center;gap:9px;font-size:12px;color:var(--muted)}
.qbar i{height:7px;border-radius:4px;background:var(--line);position:relative;display:block}
.qbar i::after{content:"";position:absolute;inset:0;width:var(--v);border-radius:4px;background:var(--accent)}
.qbar b{color:var(--ink);font-variant-numeric:tabular-nums;text-align:right}
/* gates */
.grow{display:flex;align-items:center;gap:9px;padding:4px 0}
.gname{font-size:13px;font-weight:600;text-transform:capitalize}
.gdet{margin-left:auto;font-size:12px;color:var(--muted)}
.dot{width:9px;height:9px;border-radius:50%;flex:0 0 auto}
.dot.ok{background:var(--yes)}.dot.no{background:var(--no)}
/* citations */
.crow{display:flex;gap:10px;align-items:flex-start;margin-bottom:9px}
.lbl{flex:0 0 132px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.03em;color:var(--muted);padding-top:4px}
.lbl.mh{color:var(--mh)}.lbl.nh{color:var(--nh)}.lbl.rc{color:var(--rc)}.lbl.fb{color:var(--fb)}
.lbl b{font-size:12px}
.chips{display:flex;gap:6px;flex-wrap:wrap}
.chip{font-size:12.5px;font-weight:600;padding:3px 9px;border-radius:6px;text-decoration:none;
border:1px solid transparent;font-variant-numeric:tabular-nums}
.chip:hover{filter:brightness(1.08)}
.chip.hit{background:color-mix(in srgb,var(--yes) 16%,transparent);color:var(--yes);border-color:color-mix(in srgb,var(--yes) 38%,transparent)}
.chip.miss{background:color-mix(in srgb,var(--no) 12%,transparent);color:var(--no);border-color:color-mix(in srgb,var(--no) 34%,transparent);text-decoration:line-through}
.chip.fbhit{background:color-mix(in srgb,var(--no) 14%,transparent);color:var(--no);border-color:color-mix(in srgb,var(--no) 36%,transparent);font-weight:800}
.chip.extra{background:var(--code);color:var(--muted);border-color:var(--line)}
.empty{color:var(--muted);opacity:.55}
.ok-inline{color:var(--yes);font-size:12.5px;font-weight:600}
.prov .provtxt{font-size:12.5px;color:var(--muted)}.provtxt b{color:var(--ink)}
/* facts */
.glabel{font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;margin:8px 0 4px}
.glabel.mh{color:var(--mh)}.glabel.nh{color:var(--nh)}
.fact{display:flex;gap:9px;align-items:baseline;padding:4px 0;border-bottom:1px dashed var(--line)}
.fact:last-child{border-bottom:none}
.fact .dot{position:relative;top:1px}
.fid{flex:0 0 52px;font-size:11px;font-weight:700;color:var(--muted)}
.rx{background:var(--code);color:var(--accent);padding:1px 6px;border-radius:5px;
font:12px ui-monospace,Menlo,monospace;white-space:nowrap;max-width:190px;overflow:hidden;text-overflow:ellipsis}
.note{color:var(--muted);font-size:12.5px}
/* traps */
.trap{display:flex;gap:9px;padding:6px 0;border-bottom:1px dashed var(--line)}
.trap:last-child{border-bottom:none}.trap .dot{position:relative;top:5px}
.tbody{display:flex;flex-direction:column;gap:2px}
.tname{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.03em;color:var(--fb)}
.tdesc{font-size:12.5px}
.twhy{font-size:11.5px;color:var(--no);font-family:ui-monospace,Menlo,monospace}
.dims{margin:0;padding-left:18px;font-size:13px;color:var(--muted)}.dims li{margin:3px 0}
/* answer */
.answer{position:sticky;top:60px}
.bmeta{font-size:11.5px;color:var(--muted);font-family:ui-monospace,Menlo,monospace;
margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid var(--line)}
.prose{font-size:14px;line-height:1.62}
.prose h2,.prose h3,.prose h4{margin:15px 0 7px;font-size:15px}
.prose h2:first-child,.prose h3:first-child{margin-top:0}
.prose p{margin:9px 0}.prose ul{margin:9px 0;padding-left:20px}
.prose a{color:var(--accent);text-decoration:none}.prose a:hover{text-decoration:underline}
.prose code{background:var(--code);padding:1px 5px;border-radius:4px;font:12.5px ui-monospace,Menlo,monospace}
.prose hr{border:none;border-top:1px solid var(--line);margin:14px 0}
.prose blockquote.prov{margin:12px 0 0;padding:9px 12px;border-left:3px solid var(--line);
background:var(--code);border-radius:0 7px 7px 0;font-size:11.5px;color:var(--muted);line-height:1.5}
.prose blockquote.prov em{font-style:italic}
.bnote{margin-top:14px;padding:9px 11px;border-left:3px solid var(--fb);background:var(--code);
border-radius:0 7px 7px 0;font-size:12.5px;color:var(--muted)}
a{color:var(--accent)}
/* info tooltips */
.info{display:inline-flex;align-items:center;justify-content:center;width:14px;height:14px;
border-radius:50%;border:1px solid var(--muted);color:var(--muted);font:800 9px/1 Georgia,serif;
font-style:italic;cursor:help;position:relative;vertical-align:middle;user-select:none}
.info:hover,.info:focus{background:var(--accent);border-color:var(--accent);color:#fff;outline:none}
.info::after{content:attr(data-tip);position:absolute;left:50%;top:21px;transform:translateX(-50%);
width:250px;background:var(--ink);color:var(--bg);padding:9px 11px;border-radius:9px;
font:400 11.5px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0;
text-transform:none;box-shadow:0 8px 28px rgba(0,0,0,.32);opacity:0;visibility:hidden;
transition:opacity .12s;z-index:50;pointer-events:none;white-space:normal;font-weight:400}
.info.r::after{left:auto;right:0;transform:none}
.info:hover::after,.info:focus::after{opacity:1;visibility:visible}
.qlabel .info{border-color:var(--muted)}
.lbl .info,.glabel .info{color:var(--muted);border-color:var(--muted)}
.ctabinfo{margin-left:auto;align-self:center;font-size:11px;color:var(--muted);
text-transform:uppercase;letter-spacing:.04em;display:inline-flex;gap:5px;align-items:center}
/* legend */
.legend{background:var(--card);border:1px solid var(--line);border-radius:12px;margin-bottom:18px;padding:2px 4px}
.legend>summary{cursor:pointer;padding:11px 14px;font-size:13px;font-weight:700;color:var(--ink);list-style:none}
.legend>summary::-webkit-details-marker{display:none}
.legend[open]>summary{border-bottom:1px solid var(--line);margin-bottom:2px}
.legrid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;padding:14px}
@media(max-width:940px){.legrid{grid-template-columns:1fr}}
.lgcard{font-size:12.5px;color:var(--muted);line-height:1.5}
.lgcard.span{grid-column:1/-1}
.lgcard b{color:var(--ink)}
.lgcard>b:first-child{display:block;margin-bottom:4px;font-size:13px}
.lgcard code{display:block;background:var(--code);color:var(--accent);padding:7px 9px;border-radius:7px;
margin:5px 0;font:12px/1.5 ui-monospace,Menlo,monospace;white-space:pre-wrap}
"""

JS = """
const tabs=[...document.querySelectorAll('.tab')],secs=[...document.querySelectorAll('.q')],grps=[...document.querySelectorAll('.grp')];
function show(id){tabs.forEach(t=>t.classList.toggle('active',t.dataset.target===id));
secs.forEach(s=>s.classList.toggle('show',s.dataset.id===id));
grps.forEach(g=>g.classList.toggle('sel',g.dataset.target===id));history.replaceState(null,'','#'+id);}
tabs.forEach(t=>t.addEventListener('click',()=>show(t.dataset.target)));
grps.forEach(g=>g.addEventListener('click',()=>show(g.dataset.target)));
const init=(location.hash||'').replace('#','');
show(secs.some(s=>s.dataset.id===init)?init:secs[0].dataset.id);
document.querySelectorAll('.q').forEach(sec=>{
  const ctabs=[...sec.querySelectorAll('.ctab')],cands=[...sec.querySelectorAll('.candidate')];
  ctabs.forEach(ct=>ct.addEventListener('click',()=>{
    ctabs.forEach(x=>x.classList.toggle('active',x===ct));
    cands.forEach(c=>c.classList.toggle('active',c.dataset.cand===ct.dataset.cand));
  }));
});
"""

DOC = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Marin evals — scored</title><style>{CSS}</style></head>
<body><div class="wrap">
  <div class="top"><h1>Marin skill evals</h1>
    <span class="note">seed golds · gold vs 3 candidates</span>
    <span class="freeze">freeze 2026-07-16 · 68,026 chunks + 15 wk summaries · gh 20,609 / wandb 19,446 / discord 27,971</span></div>
  {chart_html}
  <div class="tabs">{tabbar}</div>
  <details class="legend">
    <summary>How scoring works &nbsp;▾</summary>
    <div class="legrid">
      <div class="lgcard span"><b>Q — the one quality number (0–100)</b>
        <code>Q = 100·(0.35·cite-recall + 0.35·fact-coverage + 0.30·trap-pass) − 10·(forbidden cited)</code>
        clamped to 0–100. Any <b>hard-gate</b> failure forces Q=0. Nice-to-have citations, recency,
        and provenance are reported but <b>not</b> in Q.</div>
      <div class="lgcard"><b>must-have citation</b>Issues the answer must cite. Recall over this set is <b>35%</b> of Q.</div>
      <div class="lgcard"><b>nice-to-have citation</b>Bonus / tiebreak only — <b>not</b> in Q, so a must-have can't be traded for two nice-to-haves.</div>
      <div class="lgcard"><b>recency-critical</b>The must-have subset carrying the latest / close-out state — the "retrieved date" metric. Reported, not weighted.</div>
      <div class="lgcard"><b>forbidden citation</b>Wrong / superseded / off-platform issues (e.g. gpu #5167). Citing one = <b>−10</b> points; the machine-checkable half of a "superseded" trap.</div>
      <div class="lgcard"><b>facts (Fact 1…n)</b>Atomic facts checked by regex over the answer; must-have coverage is <b>35%</b> of Q. Numbered per question; each fact scores independently, so one claim with two checkable parts becomes two facts.</div>
      <div class="lgcard"><b>traps</b>Known ways to be subtly wrong (target-vs-achieved, platform, temporal, superseded). Each is a deterministic require/forbid check; fraction passed = <b>30%</b> of Q.</div>
      <div class="lgcard"><b>hard gates</b>Overriding pass/fail: a hallucinated citation (frozen mode) or a wrong abstention forces Q=0.</div>
      <div class="lgcard"><b>provenance &amp; modes</b><b>reference</b> = the gold (gates off) · <b>live</b> = hit the live repo (post-freeze cites tolerated) · <b>frozen</b> = must use the frozen corpus (post-freeze cites are hallucinations).</div>
    </div>
  </details>
  {sections}
</div><script>{JS}</script></body></html>"""

OUT.write_text(DOC)
print(f"wrote {OUT}  ({OUT.stat().st_size} bytes)")
