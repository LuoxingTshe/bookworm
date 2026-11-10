"""Error taxonomy shared by the service layer and the CLI.

Each error carries a stable `code` (reported in `--json` output) and a process exit code, so a calling
agent can branch on the failure kind without parsing messages. They subclass the builtin exception
they replace, so existing `except RuntimeError` / `except ValueError` callers keep working.
"""

from __future__ import annotations


class BookwormError(Exception):
    code = "error"
    exit_code = 1


class ConfigError(BookwormError, RuntimeError):
    """Missing API key or other environment problem: fix .env, not the request."""

    code = "config"
    exit_code = 2


class InputError(BookwormError, ValueError):
    """The request itself is invalid (bad option value, output path not new, unresolvable gold set)."""

    code = "invalid_input"
    exit_code = 2


class BusyError(BookwormError, RuntimeError):
    """Another retrieval holds the results directory lock; retry later."""

    code = "busy"
    exit_code = 75  # EX_TEMPFAIL


class BudgetError(BookwormError):
    """An observed token or request budget is exhausted."""

    code = "budget"
    exit_code = 3


class UpstreamError(BookwormError, RuntimeError):
    """A retrieval or model service could not complete the operation."""

    code = "upstream"
    exit_code = 69


