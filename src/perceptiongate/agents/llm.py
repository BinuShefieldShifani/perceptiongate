"""Minimal LLM clients over plain HTTP (standard library only).

PGATE_LLM selects the provider:
  anthropic  Claude API. Needs ANTHROPIC_API_KEY. Model: PGATE_LLM_MODEL.
  openai     Any OpenAI-compatible endpoint: Ollama, Groq, vLLM, OpenAI.
             PGATE_LLM_BASE_URL (default Ollama, http://localhost:11434/v1),
             PGATE_LLM_API_KEY if the endpoint needs one, PGATE_LLM_MODEL.
  none       No LLM: the comment uses the deterministic summary.
PGATE_LLM_TIMEOUT sets the per-call timeout in seconds (default 300; CPU inference of
larger models can take minutes).
If PGATE_LLM is unset: anthropic when ANTHROPIC_API_KEY is set, openai when
PGATE_LLM_BASE_URL is set, otherwise none.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Reply:
    text: str
    model: str
    seconds: float
    usage: dict[str, Any] = field(default_factory=dict)


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> Reply: ...


def _timeout() -> float:
    return float(os.environ.get("PGATE_LLM_TIMEOUT", "300"))


def _post(
    url: str, headers: dict[str, str], body: dict[str, Any], timeout: float | None = None
) -> Any:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout or _timeout()) as resp:  # noqa: S310 - fixed scheme
        return json.loads(resp.read())


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        self.key = os.environ["ANTHROPIC_API_KEY"]
        self.model = model or os.environ.get("PGATE_LLM_MODEL", "claude-haiku-5-5")

    def complete(self, system: str, user: str) -> Reply:
        start = time.monotonic()
        data = _post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": self.key, "anthropic-version": "2023-06-01"},
            {
                "model": self.model,
                "max_tokens": 800,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        text = "".join(b.get("text", "") for b in data.get("content", []))
        return Reply(text, self.model, time.monotonic() - start, data.get("usage", {}))


class OpenAICompatibleLLM:
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        self.base = os.environ.get("PGATE_LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
        self.key = os.environ.get("PGATE_LLM_API_KEY", "")
        self.model = model or os.environ.get("PGATE_LLM_MODEL", "qwen2.5:3b")

    def complete(self, system: str, user: str) -> Reply:
        start = time.monotonic()
        headers = {"authorization": f"Bearer {self.key}"} if self.key else {}
        data = _post(
            f"{self.base}/chat/completions",
            headers,
            {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        text = data["choices"][0]["message"]["content"]
        return Reply(text, self.model, time.monotonic() - start, data.get("usage", {}))


def make_llm() -> LLM | None:
    choice = os.environ.get("PGATE_LLM")
    if choice is None:
        if os.environ.get("ANTHROPIC_API_KEY"):
            choice = "anthropic"
        elif os.environ.get("PGATE_LLM_BASE_URL"):
            choice = "openai"
        else:
            choice = "none"
    if choice == "anthropic":
        return AnthropicLLM()
    if choice == "openai":
        return OpenAICompatibleLLM()
    if choice == "none":
        return None
    raise ValueError(f"unknown PGATE_LLM={choice!r}")
