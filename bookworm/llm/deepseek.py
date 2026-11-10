"""DeepSeek client (OpenAI-compatible): JSON output, tool calling, retries, token accounting."""

from __future__ import annotations

import json
import re
import threading
from typing import Any, Protocol

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import Settings, get_settings
from ..errors import ConfigError
from . import prompts as P
from ..trace import Tracer

_RETRYABLE = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)


class LLM(Protocol):
    """Interface the pipeline depends on; tests can inject a fake implementation."""

    def chat_json(self, system: str, user: str, *, purpose: str = "") -> dict: ...

    def chat_tools(self, messages: list[dict], tools: list[dict]) -> Any: ...


class DeepSeek:
    def __init__(self, settings: Settings | None = None, tracer: Tracer | None = None):
        self.s = settings or get_settings()
        if not self.s.deepseek_api_key:
            raise ConfigError("DEEPSEEK_API_KEY is not set (see .env.example)")
        self.client = OpenAI(api_key=self.s.deepseek_api_key, base_url=self.s.deepseek_base_url,
                             timeout=self.s.request_timeout, max_retries=0)
        self.tracer = tracer or Tracer()
        self._lock = threading.Lock()

    def close(self):
        self.client.close()

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _complete(self, purpose: str, **kwargs: Any):
        self.tracer.check_budget()
        resp = self.client.chat.completions.create(
            model=self.s.deepseek_model, temperature=self.s.deepseek_temperature, **kwargs
        )
        with self._lock:
            u = self.tracer.usage
            u.deepseek_calls += 1
            if resp.usage:
                u.deepseek_prompt_tokens += resp.usage.prompt_tokens or 0
                u.deepseek_completion_tokens += resp.usage.completion_tokens or 0
            self.tracer.log(
                "llm",
                purpose=purpose,
                prompt_tokens=getattr(resp.usage, "prompt_tokens", None),
                completion_tokens=getattr(resp.usage, "completion_tokens", None),
            )
        return resp

    def chat_json(self, system: str, user: str, *, purpose: str = "") -> dict:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(2):
            resp = self._complete(purpose, messages=msgs, response_format={"type": "json_object"})
            content = resp.choices[0].message.content or ""
            try:
                return parse_json(content)
            except ValueError:
                if attempt == 1:
                    raise
                msgs += [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": P.JSON_RETRY},
                ]
        raise AssertionError("unreachable")

    def chat_tools(self, messages: list[dict], tools: list[dict]):
        """Returns the assistant message object (with .content / .tool_calls)."""
        return self._complete("agent", messages=messages, tools=tools, tool_choice="auto").choices[0].message


def parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"no JSON object in LLM output: {text[:200]}")
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError as e:
            raise ValueError(str(e)) from e
    if not isinstance(obj, dict):
        raise ValueError("LLM JSON output is not an object")
    return obj
