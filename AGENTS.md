# Bookworm 蠹

Bookworm is an autonomous literature research tool and a sub-agent for shell-capable agents.

## Delegated retrieval

Read [SKILL.md](SKILL.md). Prefer `bookworm --quiet task request.json` or `bookworm --json --quiet run "..."`.
Use `--home /absolute/workspace` when the caller's working directory differs from the configured workspace.
Run `doctor` before the first retrieval; never print `.env` or API keys. Use the default quick budget unless the user requests a wider search.
Report `complete`, artifact paths, corpus size, core papers, all warnings and usage. A partial result is not a completed search. Never delete research artifacts as part of a retrieval task.

## Code changes

Keep CLI commands thin; place orchestration and typed results in `bookworm/service.py` and register result models in `RESULT_MODELS`.
Validate task input with `ResearchRequest`; keep new corpus/result fields backwards-compatible with defaults.
Keep LLM prompts in `bookworm/llm/prompts.py`. Implement retrieval tools by reusing `pipeline/` and `sources/`.
Preserve one-line JSON stdout on success, partial completion and invalid input. Keep progress on stderr and credentials out of responses/traces.
Use atomic writes for recoverable state, never overwrite a continuation input, and never prune old runs implicitly.
Before publishing run Python tests, Ruff, Node layout tests and a wheel build/install smoke test. Offline tests must not call live APIs.
