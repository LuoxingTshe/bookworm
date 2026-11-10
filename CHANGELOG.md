# Bookworm 蠹 — 2.0.0 (2026-11-09)

- Rename the package and command to Bookworm; remove the SCIBooster import/CLI aliases and legacy repository files.
- Add unified autonomous `run`, JSON stdin/file `task`, configuration `doctor`, and versioned task schemas.
- Resolve explicit workspaces and source selection; preserve legacy environment names.
- Enforce input, year, paper, tool-step and observed-token budgets; reset usage for continuation.
- Persist atomic checkpoints, typed partial results and automatic BibTeX/RIS/CSV exports.
- Keep all earlier run directories, including failed and interrupted runs; preserve original continuation inputs.
- Bound HTTP operations, close clients, isolate usage counters, and avoid leaking upstream credential URLs.
- Restore offline regression tests, add protocol/recovery tests and CI wheel installation checks.
- Continue the existing workspace cleanup: omit obsolete renderer, bundled unrelated skills and historical experiment fixtures from the distribution.

## Migration from SCIBooster

Use `bookworm` / `python -m bookworm` / `from bookworm import service` for new work.
Existing corpus files remain readable; the `scibooster` command and import package are removed. Reinstall with `pip install -e .`.
`build` now defaults to quick/auto. Partial CLI runs exit 3, interrupted runs exit 130.
`--keep-history` is retained for compatibility; automatic deletion of old runs is removed.
Obsidian now generates Bookworm tags while preserving legacy user-note delimiters.

---

# Release notes

## Unreleased

- 2026-10-30: trimmed to the skill-agent essentials. `SKILL.md` at the root makes the directory a skill; the Canvas layout files moved into the package `layout/` directory (package data); removed the dev renderer and `serve` command, tests, experiment scripts, CI, vendored skills and `.claude/agents` (all still in Git history); dropped the `fastapi` / `uvicorn` dependencies and added `click` explicitly.
- Sub-agent interface: a `service` layer returns typed results for every task; the CLI gains global `--json` / `--quiet` (`SCIB_JSON`, `SCIB_QUIET`) with one JSON object on stdout, stable error codes and exit codes, and a `schema` command. `build` / `agent` treat SIGTERM as an interruption. `AGENTS.md` gives tool-neutral instructions for main agents (Codex, WorkBuddy, Claude Code, …); a Claude Code sub-agent definition was registered (since removed). Human CLI output is unchanged.

## beta1.0 — 2026-10-08

First official Beta release. Git tag: `beta1.0`; Python package version: `1.0.0b1`.

- Retrieve literature through WoS Starter and OpenAlex with DeepSeek intent parsing and relevance screening, citation expansion, co-citation gap filling and an optional tool-calling Agent.
- Export a citation-linked corpus, Obsidian notes and Canvas, BibTeX, RIS and CSV; evaluate recall against fixed literature packages.
- Merge duplicate literature versions conservatively, prefer published records, preserve original versions and citation provenance, and report uncertain matches for review. Existing corpora can be deduplicated offline into a new directory.
- Store runtime caches and results separately under `artifacts/`. Test cases retain literature inputs, gold sets, baselines and test pipelines.
- After successful retrieval and output generation, retain the latest complete result and clear cached responses. Failed, interrupted and incomplete runs preserve earlier results; `--keep-history` and `--keep-cache` support repeat experiments.
- Continue Agent sessions in a new directory and snapshot source corpora for variance experiments so offline analysis survives result cleanup.
- Display the release with `scibooster --version`.

Validation: 96 offline Python tests and 8 citation-layout tests pass. No paid API calls are needed for these checks. Actual retrieval requires configured API credentials; see the README for setup and current retrieval limitations.
