"""Manage the bundled agent skills: list, print, or install them.

Skills live as ``mumwelt/skills/<name>/SKILL.md`` (Claude-compatible YAML frontmatter +
model-agnostic body). ``install`` copies them into a host's skill directory (Claude's
``~/.claude/skills`` by default); ``print`` dumps the markdown for pasting into any other
agent's system prompt / tool docs.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent / "skills"
DEFAULT_DEST = Path.home() / ".claude" / "skills"


def _skill_dirs() -> list[Path]:
    return sorted(p for p in SKILLS_DIR.iterdir() if (p / "SKILL.md").exists()) \
        if SKILLS_DIR.exists() else []


def _frontmatter_desc(md: Path) -> str:
    desc = ""
    for line in md.read_text(encoding="utf-8").splitlines():
        if line.startswith("description:"):
            return line.split(":", 1)[1].strip()
    return desc


def dispatch(a) -> None:
    sub = getattr(a, "sub", "list") or "list"
    if sub == "list":
        for d in _skill_dirs():
            print(f"  {d.name}\n      {_frontmatter_desc(d / 'SKILL.md')[:100]}")
        if not _skill_dirs():
            print("  (no skills bundled)")
    elif sub == "print":
        for d in _skill_dirs():
            if a.dest and a.dest != d.name:
                continue
            print(f"\n===== {d.name} =====\n")
            print((d / "SKILL.md").read_text(encoding="utf-8"))
    elif sub == "install":
        dest = Path(a.dest).expanduser() if a.dest else DEFAULT_DEST
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for d in _skill_dirs():
            shutil.copytree(d, dest / d.name, dirs_exist_ok=True)
            n += 1
        print(f"installed {n} skill(s) → {dest}", file=sys.stderr)
