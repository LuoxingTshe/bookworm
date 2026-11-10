"""Run tracing: writes every LLM/API call to trace.jsonl and accumulates usage."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .models import Usage


class Tracer:
    def __init__(self, path: Path | None = None, usage: Usage | None = None):
        self.path = path
        self.usage = usage or Usage()
        self.max_tokens: int | None = None
        self.warnings: list[str] = []

    def check_budget(self) -> None:
        from .errors import BudgetError
        if self.max_tokens is not None and self.usage.deepseek_prompt_tokens + self.usage.deepseek_completion_tokens >= self.max_tokens:
            raise BudgetError(f"Observed token budget exhausted ({self.max_tokens})")

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
        self.log("warning", message=message)

    def log(self, kind: str, **data: Any) -> None:
        if self.path is None:
            return
        rec = {"t": round(time.time(), 3), "kind": kind, **data}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


NULL_TRACER = Tracer()
