"""DeepSeek tool-calling loop with step/token budgets."""

from __future__ import annotations

from typing import Callable

from ..llm import prompts as P
from .tools import TOOLS, AgentContext, dispatch

Log = Callable[[str], None]


def run_agent(ctx: AgentContext, prompt: str, max_steps: int = 30, max_tokens: int = 400_000, log: Log = print) -> None:
    """The corpus size cap is ctx.max_papers, enforced by add_to_corpus (None = no cap)."""
    system = P.fill(
        P.AGENT_SYSTEM, prompt=prompt, intent=ctx.intent.model_dump_json(), threshold=ctx.threshold,
        cap_line=(f"Hard size cap: {ctx.max_papers} papers (add_to_corpus refuses beyond it; remove weaker papers "
                  "to make room)." if ctx.max_papers is not None else "No size cap."),
    )
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": P.fill(P.AGENT_START, size=len(ctx.corpus), steps=max_steps)},
    ]
    usage = ctx.tracer.usage
    nudges = 0
    ctx.stop_reason = "max_steps"
    tools = [t for t in TOOLS if ctx.wos is not None or t["function"]["name"] != "search_wos"]
    for step in range(1, max_steps + 1):
        if usage.deepseek_prompt_tokens + usage.deepseek_completion_tokens >= max_tokens:
            ctx.stop_reason = "budget"
            log(f"[yellow]Token budget reached ({max_tokens}), stopping[/]")
            break
        if step == max_steps - 1:
            messages.append({"role": "user", "content": P.AGENT_FINISH})
        msg = ctx.llm.chat_tools(messages, tools)
        calls = msg.tool_calls or []
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            **({"tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in calls
            ]} if calls else {}),
        })
        if msg.content:
            log(f"[dim]{step:>2} 💭 {msg.content.strip()[:200]}[/]")
        if not calls:
            nudges += 1
            if nudges >= 2:
                ctx.stop_reason = "no_tool_calls"
                break
            messages.append({"role": "user", "content": P.AGENT_CONTINUE})
            continue
        nudges = 0
        for c in calls:
            if usage.deepseek_prompt_tokens + usage.deepseek_completion_tokens >= max_tokens:
                ctx.stop_reason = "budget"
                break
            log(f"{step:>2} 🔧 {c.function.name}({c.function.arguments[:150]})")
            result = dispatch(ctx, c.function.name, c.function.arguments)
            messages.append({"role": "tool", "tool_call_id": c.id, "content": result})
            if ctx.finished:
                ctx.stop_reason = "finished"
                break
        log(f"   corpus = {len(ctx.corpus)}")
        if ctx.finished:
            break
    from ..dedup import deduplicate

    deduplicate(ctx.corpus, ctx.tracer)
