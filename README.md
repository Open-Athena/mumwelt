# mumwelt
#### Or, Marin's umwelt
*Written with human hands*

> There are not only the manifolds in space and time in which things can be spread out. There is also the manifold of environments, in which things repeat themselves in always new forms.
- Jakob von Uexküll, *A Foray into the World of Animals and Humans*

mumwelt provides the environment for an agent to effectively commune with the Marin project.

It is
- A CLI (`mum`), which queries and explores (via link expansion) the corpus of data around the Marin project (hosted on marinmirror.xyz.exe)
- This corpus includes GitHub issues, PRs, comments, code (and recent branches), Weights & Biases data, summaries of the week's learnings, and text from the public Discord server
- A set of skills to answer broad questions about the how and why of Marin, and specific questions about the code (in `main` and in branches), and publish formatted HTML gists to share

With it, you, a human, can better understand how Marin works, why it is the way it is, and more quickly contribute to the program if you so desire. 

It is the **client** to [`marinmirror`](https://marinmirror.exe.xyz) (the hosted, indexed, embedded, pre-processed corpus described above). 

## Install

Realistically, you can just point your agent at this repo and ask it to set things up.

Note that you will need access granted to your GitHub account via OAuth to [`marinmirror`](https://marinmirror.exe.xyz). You can apply for access here (which is gated on Marin Discord access), and Open Athena will manually review and approve your application.


# LLM-written details

## Install details

```bash
pip install -e .
```

Installing `mumwelt` will also install embedding models to preprocess your queries, enabling semantic search across the corpus.

Token: `mum` authenticates to marinmirror with a GitHub bearer token, resolved from
`MARINMIRROR_TOKEN` → `gh auth token` → `~/.config/marin/token`. The weekly summaries are
public (no token).

Your first `mum refresh` lands in one of three places, and it tells you which:

1. **No token found** — nothing was sent. Run `gh auth login` (`mum` picks the token up
   from `gh auth token`), or set `MARINMIRROR_TOKEN`, or write `~/.config/marin/token`.
2. **Token found but not authorized** — the server refused it, and `mum` prints the
   server's own reason. Authorizing is a one-time browser step: open
   [marinmirror](https://marinmirror.exe.xyz/), sign in, submit an access request, and
   re-run `mum refresh` once it's approved. `mum refresh --open` opens that page for you.
3. **Authorized** — the corpus downloads.

In every case the public weekly summaries still refresh, so an unauthorized install is
partially useful rather than dead.

## The `mum` CLI

```
mum status                         freshness of the local corpus + summaries vs upstream
mum refresh [--force]              pull the latest corpus (marinmirror) + weekly summaries (mws.oa.dev)
mum search "<q>" [flags]           hybrid keyword+semantic search; cited hits
mum search-multi "q1" "q2" …       run several queries concurrently, merged + deduped
mum show <url|ref> [--window N]    expand a hit to context (discord window / github thread)
mum run <project>/<run>            a W&B run's metadata + final summary numbers
mum summaries [list|show|links|refresh] [<period|latest>]
mum publish [file] --title "…"     render Markdown → LaTeX-styled HTML in a secret gist + share link
mum skills [list|print|install [dest]]
```

`mum search` / `search-multi` flags: `-k N` (default 50; `search-multi` also takes
`--total`, default 60, capping the merged set), `--source`, `--kind`, `--since`, `--until`,
`--fts-only` (skip the model load), `--json`.

Sources: `github`, `discord`, `wandb`, `narrative`, and `code` — the marin repo's Python
symbols (`--source code`), one chunk per function/class/method plus one per module, covering
`main` and every branch touched in the last 30 days (`--kind branch-symbol`, tagged with the
branch and its last committer). Identifier search splits camelCase in both directions, so
`ExecutorStep`, `executor_step`, and "executor step" all reach each other.

`mum publish` flags: `--title`, `--author`, `--description`, `--filename`, `--public`
(default: secret/unlisted), `--no-date`, `--open`, `--json`. Renders via `pandoc` if
present, else a built-in converter — no extra Python deps. Requires the `gh` CLI.

Local cache lives under `~/.cache/marin/` (`corpus-index.db` + `summaries/`).

## Skills

Three skills, shipped in Claude's `SKILL.md` format and as a portable [`AGENTS.md`](AGENTS.md):

- **`mumwelt-code`** — a single "where is X / how is Y implemented" lookup: search the
  embeddings-backed code lane first, and escalate to `mumwelt` only if a clear, citable
  answer doesn't emerge.
- **`mumwelt`** — decompose a broad question → fan out parallel searches across
  subagents (or `mum search-multi`) → verify → synthesize a cited answer.
- **`mumwelt-publish`** — turn a finished, cited writeup into a LaTeX-styled HTML page in a
  secret gist, and hand back an `htmlpreview.github.io` link for easy sharing.

```bash
mum skills install            # → ~/.claude/skills/  (Claude Code / claude.ai)
mum skills print              # dump the markdown for any other agent's prompt
```

The agent (any LLM) reasons and synthesizes; `mum` does retrieval. Search fuses FTS5
keyword search with cosine over the corpus's vector spaces — prose in 384-d
`BAAI/bge-small-en-v1.5`, code in 768-d `jinaai/jina-embeddings-v2-base-code` — each scored
separately (the widths differ) and combined by reciprocal-rank fusion, so identifiers and
natural-language intent both hit and code is matched by a code-trained encoder rather than
an English one.
