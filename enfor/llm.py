"""Chat generation with resumable JSONL logs.

``generate`` sends requests [(key, user text)] with a fixed system prompt and few-shot prefix and appends each reply to
``log`` as {"key", "text", "stop", "usage"}; an interrupted run resumes. Open-weights models run locally with vLLM
(temperature 1.0, seed = request index); model names starting with ``claude`` go to the Anthropic API.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from enfor.env import MODELS_DIR

CHUNK = 2048
FINAL = "assistantfinal"   # gpt-oss: the reasoning channel precedes the final answer in the decoded text
_ENGINES: Dict[str, object] = {}


def read_log(log: Path) -> Dict[str, dict]:
    """key -> record of the finished replies in ``log`` (failed calls are left out, so they are retried)."""
    done: Dict[str, dict] = {}
    if Path(log).exists():
        for line in open(log):
            try:
                r = json.loads(line)
            except ValueError:
                continue   # a line cut by an interrupted write
            if r.get("text") is None and r["key"] in done:
                continue
            done[r["key"]] = r
    return {k: r for k, r in done.items() if r.get("text") is not None}


def _engine(model: str, gpu_memory_utilization: float = 0.85, max_model_len: int = 16384):
    if model not in _ENGINES:
        from vllm import LLM

        _ENGINES[model] = LLM(model=str(MODELS_DIR / model), seed=0, enable_prefix_caching=True,
                              gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len)
    return _ENGINES[model]


def _local(model, system, prefix, todo, log, effort, max_tokens):
    from vllm import SamplingParams

    llm = _engine(model)
    head = [{"role": "system", "content": system}] + list(prefix)
    with open(log, "a") as out:
        for start in range(0, len(todo), CHUNK):
            part = todo[start:start + CHUNK]
            convs = [head + [{"role": "user", "content": u}] for _, (_, u) in part]
            params = [SamplingParams(temperature=1.0, max_tokens=max_tokens, seed=i) for i, _ in part]
            outs = llm.chat(convs, params, use_tqdm=False, chat_template_kwargs={"reasoning_effort": effort})
            for (_, (key, _)), o in zip(part, outs):
                c = o.outputs[0]
                text = c.text.rsplit(FINAL, 1)[1] if FINAL in c.text else c.text
                out.write(json.dumps({"key": key, "text": text, "stop": c.finish_reason,
                                      "usage": {"output_tokens": len(c.token_ids)}}) + "\n")
            out.flush()


async def _claude(model, system, prefix, todo, log, effort, max_tokens, concurrency=16):
    import anthropic

    client = anthropic.AsyncAnthropic(max_retries=8, timeout=300.0)
    sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()
    with open(log, "a") as out:
        async def one(key: str, user: str):
            async with sem:
                r = await client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                                 messages=list(prefix) + [{"role": "user", "content": user}],
                                                 output_config={"effort": effort})
            text = next((b.text for b in r.content if b.type == "text"), None)
            usage = {"output_tokens": getattr(r.usage, "output_tokens", 0) or 0}
            async with lock:
                out.write(json.dumps({"key": key, "text": text, "stop": r.stop_reason, "usage": usage}) + "\n")
                out.flush()

        await asyncio.gather(*(one(k, u) for _, (k, u) in todo))


def generate(model: str, system: str, prefix: List[dict], requests: Sequence[Tuple[str, str]], log: Path,
             effort: str = "low", max_tokens: int = 4096) -> Dict[str, Optional[str]]:
    """key -> reply text for every request (None if it failed)."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    done = read_log(log)
    todo = [(i, r) for i, r in enumerate(requests) if r[0] not in done]
    if todo:
        if model.startswith("claude"):
            asyncio.run(_claude(model, system, prefix, todo, log, effort, max_tokens))
        else:
            _local(model, system, prefix, todo, log, effort, max_tokens)
        done = read_log(log)
    return {k: done[k].get("text") if k in done else None for k, _ in requests}
