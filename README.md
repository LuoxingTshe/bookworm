# Bookworm 蠹

**Autonomous literature retrieval, usable on its own or as a sub-agent.**

Bookworm turns a research question into a citation-linked corpus. It searches Web of Science / OpenAlex, scores relevance with DeepSeek, follows references and citing works, fills co-citation gaps, and exports references. The same service powers the terminal, a JSON task protocol, and the Python API.

Formerly **SCIBooster**. Release **2.0.0** keeps existing corpus files, legacy environment names and Obsidian notes readable. The distribution is named `bookworm-research`; the command and Python package are `bookworm`.

## Install

Python 3.10+ on macOS or Linux. Node.js is optional for the bundled fCoSE graph layout; Python provides a fallback.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
cp .env.example .env
# Edit .env: set DEEPSEEK_API_KEY; optionally WOS_API_KEY and OPENALEX_API_KEY.
.venv/bin/bookworm doctor
```

`doctor` checks configuration locally, without exposing keys or contacting providers. It does **not** verify network connectivity, account permissions, or remaining API credit. OpenAlex access and limits depend on your account and the provider's current policy.

Use `bookworm --home /path/to/workspace ...` or `BOOKWORM_HOME` to load that workspace's `.env` and resolve configured data directories. Otherwise the current directory is the workspace. Explicit input/output paths are relative to the caller's working directory. Environment variables override `.env`.

## Run autonomously

```bash
bookworm run "基于点云的三维景观可见性分析" --years 2018-2026
bookworm run "graph neural networks for drug discovery" -s W2741809807 --max-papers 40 --obsidian
```

By default, DeepSeek selects tools to search, inspect, screen, add papers and finish. `--source auto` enables WoS when its key is configured, otherwise OpenAlex. The agent can also use OpenAlex while WoS is enabled. `--source openalex` disables WoS tools explicitly.

For a fixed sequence of retrieval stages, use the pipeline:

```bash
bookworm run "三维景观可见性分析" --mode pipeline --source openalex --tier quick
bookworm build "三维景观可见性分析" --tier standard --hops 1 --no-obsidian
```

| Preset | Max papers | Pipeline query cap | Results/query | Snowball hops |
| --- | ---: | ---: | ---: | ---: |
| quick (default) | 60 | 2 | 30 | 0 |
| standard | 200 | 3 | 50 | 1 |
| deep | 400 | 4 | 100 | 3 |

The agent uses the tier's paper cap; its exploration is governed by `--max-steps` (default 12) and `--max-tokens` (default 100,000). Pipeline OpenAlex generates up to two complementary queries; the query cap cannot create additional queries. `build --help` exposes lower-level pipeline controls.

Token budgets are **observed limits**, not guaranteed billing caps: new LLM calls stop after recorded usage reaches the limit, but in-flight screening batches may exceed it. Request timeouts are per HTTP operation, not a whole-task deadline. WoS and OpenAlex have separate request limits. Failures remain visible in warnings and partial results.

## Delegate from another agent

Pass one JSON object via stdin or a file; `task` always returns one JSON line on stdout. Progress goes to stderr; `--quiet` suppresses it. Never interpolate a research prompt into a shell command.

```bash
bookworm --quiet task examples/request.json
bookworm --json --quiet run "point cloud visibility" --source openalex --mode pipeline
bookworm --json schema
```

Minimal task input:

```json
{"protocol_version":"1","prompt":"point cloud visibility","source":"openalex","mode":"agent","tier":"quick","max_steps":12,"max_tokens":100000}
```

Unknown fields and invalid options are rejected before retrieval. `bookworm schema` includes the request and result schemas. See [SKILL.md](SKILL.md) for the delegation workflow and [examples/delegate.py](examples/delegate.py) for safe subprocess invocation.

A run result contains `ok`, `complete`, `stop_reason`, `error`, absolute artifact paths, `summary` (core papers, counts and any available PRISMA record), `warnings`, and per-run `usage`. **`ok: true` means a result was produced; only `complete: true` means the requested workflow finished.** It never guarantees exhaustive coverage.

| Exit code | Meaning | Caller action |
| --- | --- | --- |
| 0 | Completed command | Read warnings and inspect results |
| 2 | Invalid input / missing configuration | Correct the request or configuration |
| 3 | Partial result or exhausted budget | Use the returned paths; optionally continue with a new task |
| 69 | Upstream failure before a usable result | Check service/network configuration |
| 75 | Workspace busy | Retry after its active retrieval ends |
| 130 | Interrupted | Recover from returned corpus/checkpoint when available |
| 1 | Internal failure | Inspect the run and report the error |

Errors without a usable result return `ok: false` and a typed `error`. CLI parse errors also follow this protocol in JSON mode. `--help` is human-readable. Signal termination is handled as interruption where possible; SIGKILL cannot produce a final stdout response.

## Python API

```python
from bookworm import service
from bookworm.config import configure
from bookworm.requests import ResearchRequest

configure('/path/to/workspace')
result = service.run(ResearchRequest(
    prompt='point cloud visibility', source='openalex', mode='pipeline',
    max_papers=30, obsidian=False,
))
print(result.model_dump_json())
```

The service never prints; pass `log=callback` for progress. Invalid requests/configuration raise typed exceptions or Pydantic validation errors. A failure after checkpoint creation returns an explicit partial `RunResult`. Run one CLI process per delegated task; retrievals sharing a workspace are serialized by a file lock. Configuration selection is process-wide, so use separate processes for different workspaces.

## Outputs and recovery

Each run has its own directory under `artifacts/runs/`:

| File | Purpose |
| --- | --- |
| `corpus.json` | Papers, scores, provenance, citation edges, version snapshots and metadata |
| `corpus.bib`, `corpus.ris`, `corpus.csv` | Reference manager / spreadsheet exports |
| `request.json`, `result.json` | Recorded input and structured output |
| `checkpoint.json` | Recoverable intermediate state; may include unscored candidates |
| `trace.jsonl` | API/tool events and usage; no API keys |
| `.bookworm-run.json` | Running / completed / incomplete status |
| `obsidian/` | Optional notes, table, citation Canvas and overview |

**Earlier runs are never automatically deleted.** Failed and interrupted runs retain their cache and checkpoints. New `run`/`task` calls keep caches; low-level commands also honor the cache retention setting. Legacy `--keep-history` is accepted but history is always kept. `artifacts/archive/` and existing research materials remain outside the runtime lifecycle.

```bash
bookworm run "补充近期研究" --corpus /absolute/path/to/corpus.json --max-papers 100
bookworm run "继续未完成检索" --corpus /absolute/path/to/checkpoint.json --max-papers 100
```

Continuation creates a **new** run, parses the new request, and starts a fresh budget. It is a new exploration from saved papers, not an exact replay of the old model conversation. Existing papers are preserved even if the new task specifies a different year range; that range restricts new additions. Unscored candidates in a checkpoint require screening before being treated as accepted literature. Explicit seed papers may lie outside the year range and may be retracted; inspect warnings.

## Other commands

```bash
bookworm intent "research need" --source openalex
bookworm stats path/to/corpus.json
bookworm export path/to/corpus.json --format ris --min-score 7
bookworm deduplicate path/to/corpus.json --out new-directory --no-obsidian
bookworm traverse path/to/corpus.json --start W2741809807 --direction refs
bookworm obsidian path/to/corpus.json --vault /path/to/vault
bookworm eval path/to/corpus.json --gold 'survey DOI'
```

`stats`, `export`, `deduplicate`, `traverse` and `obsidian` work offline without keys. Obsidian exports preserve text below the generated notes delimiter, including legacy SCIBooster notes. Export into a new Bookworm folder by default, or specify the former folder explicitly to update it.

## Architecture and development

- `bookworm/cli.py`: option parsing, human rendering and the one-line JSON protocol.
- `bookworm/requests.py`, `service.py`: validated task input, orchestration and result models.
- `bookworm/agent/`: autonomous tool loop; tools reuse retrieval/scoring modules.
- `bookworm/pipeline/`: intent, queries, screening, snowballing, gaps and relations.
- `bookworm/sources/`, `llm/`: provider clients, cache, centralized prompts and accounting.
- `models.py`, `store.py`, `dedup.py`: corpus schema, persistence and conservative version merging.
- `runtime.py`, `persistence.py`, `trace.py`: locking, atomic writes, checkpoints and audit events.
- `obsidian.py`, `layout/`: offline graph and notes export, including vendored licenses.

```bash
python -m pytest -q
ruff check bookworm tests/unit
node --test tests/unit/test_network_layout.cjs
python -m build
```

Offline tests cover orchestration with fake providers, protocol errors, continuation, interruption, budgets, data integrity, deduplication, reference export and layout. CI builds and installs the wheel as well. Research archives, secrets, caches, historical experiment scripts and removed renderer code are not shipped.

## Limits

Relevance scores use titles and abstracts, not full-text review. Citation-first snowballing favors established work; recent and Chinese-language literature may be underrepresented. Unknown publication years are retained with a warning. Metadata version matching is conservative and leaves uncertain matches for review. Citation edges come from OpenAlex; optional semantic edge labels are model inferences. PRISMA-style pipeline counts describe recorded retrieval stages, not certification of a systematic review; agent exploration has no fabricated PRISMA counts. Survey-reference recall is comparative and can be inflated by using that survey as a seed.
