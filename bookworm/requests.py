"""Versioned task input shared by CLI, stdin delegation and the Python API."""
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol_version: Literal["1"] = "1"
    prompt: str = Field(min_length=1)
    mode: Literal["agent", "pipeline"] = "agent"
    source: Literal["auto", "wos", "openalex"] = "auto"
    tier: Literal["quick", "standard", "deep"] = "quick"
    seeds: list[str] = Field(default_factory=list)
    years: tuple[int | None, int | None] = (None, None)
    corpus: Path | None = None
    out: Path | None = None
    max_papers: int | None = Field(None, ge=1, le=10000)
    max_steps: int = Field(12, ge=1, le=200)
    max_tokens: int = Field(100000, ge=1)
    threshold: float = Field(6, ge=0, le=10)
    obsidian: bool = False
    top: int = Field(10, ge=0, le=100)

    @field_validator("prompt")
    @classmethod
    def nonblank_prompt(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("prompt must not be blank")
        return value

    @field_validator("years")
    @classmethod
    def valid_years(cls, value):
        start, end = value
        if any(y is not None and not 1 <= y <= 9999 for y in value):
            raise ValueError("years must be between 1 and 9999")
        if start is not None and end is not None and start > end:
            raise ValueError("year range starts after it ends")
        return value

    @model_validator(mode="after")
    def continuation_mode(self):
        if self.corpus is not None and self.mode != "agent":
            raise ValueError("corpus continuation requires mode=agent")
        if self.max_papers is not None and len(set(self.seeds)) > self.max_papers:
            raise ValueError("seed count exceeds max_papers")
        return self
