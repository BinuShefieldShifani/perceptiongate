"""Ollama path tested against a fake server speaking the same HTTP API, over real sockets."""

import importlib.util
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from perceptiongate.agents import ollama_server
from perceptiongate.agents.analyst import deterministic_summary, explain
from perceptiongate.agents.grounding import check_text
from perceptiongate.agents.llm import OpenAICompatibleLLM

ROOT = Path(__file__).resolve().parents[2]


class FakeOllama(BaseHTTPRequestHandler):
    reply = "unset"
    delay = 0.0
    requests: list[dict[str, Any]] = []

    def log_message(self, *args: Any) -> None:
        pass

    def _send(self, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._send({"version": "0.0-test"})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        FakeOllama.requests.append(body)
        time.sleep(FakeOllama.delay)
        self._send(
            {
                "choices": [{"message": {"content": FakeOllama.reply}}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 150},
            }
        )


@pytest.fixture
def fake_server(monkeypatch: pytest.MonkeyPatch) -> Any:
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("PGATE_LLM_BASE_URL", f"{url}/v1")
    monkeypatch.setenv("PGATE_LLM_MODEL", "qwen2.5:14b")
    FakeOllama.requests, FakeOllama.delay = [], 0.0
    yield url
    server.shutdown()


FACTS: dict[str, Any] = {
    "candidate": "short-25ep",
    "decision": "FAIL",
    "config_changes": ["train.epochs: 50 -> 25"],
    "blocking_failures": [],
    "needs_review": [
        {
            "check": "recall-by-condition",
            "metric": "recall",
            "slice": s,
            "baseline": b,
            "candidate": c,
            "change_points": d,
            "interval_points": [lo, hi],
            "allowed_drop_points": 1.5,
        }
        for s, b, c, d, lo, hi in [
            ("day", 0.561, 0.542, -1.9, -2.3, -1.5),
            ("night", 0.573, 0.547, -2.6, -3.1, -2.0),
            ("snow", 0.552, 0.523, -3.0, -3.8, -2.1),
        ]
    ],  # fmt: skip
    "real_drops_within_tolerance": [],
    "improvements": [],
    "too_noisy_to_decide": [],
    "per_class_recall": {"largest_drops": [], "largest_gains": []},
    "blocking_interval_level_percent": 98.75,
    "review_interval_level_percent": 95.0,
}


def test_client_talks_to_ollama_over_http(fake_server: str) -> None:
    FakeOllama.reply = "Gate decision: FAIL. Recall fell 3.0 points in snow."
    exp = explain(FACTS, OpenAICompatibleLLM())
    assert exp.source == "llm"
    sent = FakeOllama.requests[0]
    assert sent["model"] == "qwen2.5:14b"
    assert sent["temperature"] == 0
    assert '"decision": "FAIL"' in sent["messages"][1]["content"]
    assert exp.trace["attempts"][0]["usage"]["completion_tokens"] == 150


def test_slow_model_times_out_and_falls_back(
    fake_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PGATE_LLM_TIMEOUT", "0.5")
    FakeOllama.delay = 2.0
    FakeOllama.reply = "FAIL"
    exp = explain(FACTS, OpenAICompatibleLLM())
    assert exp.source == "deterministic"
    assert "timed out" in exp.trace["attempts"][0]["error"].lower()


def test_server_helpers(fake_server: str) -> None:
    assert ollama_server.server_version(fake_server) == "0.0-test"
    assert ollama_server.wait_for_server(fake_server, seconds=2) == "0.0-test"
    assert ollama_server.server_version("http://127.0.0.1:9") is None
    with pytest.raises(RuntimeError):
        ollama_server.wait_for_server("http://127.0.0.1:9", seconds=1)


def test_warm_up_sends_a_tiny_generate_request(fake_server: str) -> None:
    seconds = ollama_server.warm_up("qwen2.5:14b", url=fake_server)
    assert seconds >= 0
    sent = FakeOllama.requests[-1]
    assert sent["model"] == "qwen2.5:14b"
    assert sent["options"]["num_predict"] == 1


def test_start_does_nothing_when_server_already_runs(fake_server: str, tmp_path: Path) -> None:
    assert ollama_server.start(tmp_path / "log", url=fake_server) == "0.0-test"
    assert not (tmp_path / "log").exists()  # no second server was launched


def test_install_and_pull_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []

    def run(cmd: Any, **kw: Any) -> None:
        calls.append(cmd)

    monkeypatch.setattr(ollama_server.shutil, "which", lambda name: None)
    ollama_server.install(run=run)
    assert calls[-1] == ollama_server.INSTALL_CMD
    monkeypatch.setattr(ollama_server.shutil, "which", lambda name: "/usr/bin/ollama")
    calls.clear()
    ollama_server.install(run=run)
    assert calls == []  # already installed
    ollama_server.pull("qwen2.5:7b", run=run)
    assert calls == [["ollama", "pull", "qwen2.5:7b"]]


def test_code_summary_groups_conditions_and_stays_grounded() -> None:
    text = deterministic_summary(FACTS)
    assert "recall dropped in day, night, snow, by 1.9 to 3.0 points" in text
    assert text.count("Needs review") == 1
    assert check_text(text, FACTS).ok


def test_kaggle_agent_summary_row() -> None:
    spec = importlib.util.spec_from_file_location("ka", ROOT / "scripts" / "kaggle_agent.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    first = {"seconds": 4.2, "numbers_checked": 12, "unverified_numbers": ["5.4"]}
    second = {"seconds": 3.9, "numbers_checked": 11, "unverified_numbers": []}
    trace = {"attempts": [first, second]}
    row = mod.summarize("qwen2.5:7b", trace)
    assert row["first_try_grounded"] is False
    assert row["used"] == "llm"
    assert row["attempts"] == 2
