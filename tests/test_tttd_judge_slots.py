"""The judge scorer's in-flight cap: requests wait for a judge slot before their timeout starts."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from recipes.tttd.examples.tttd.harness.scorer import JudgeScorer, judge_slots

SOLUTION = "```python\ndef run():\n    return 1\n```"
EXAMPLE = Path(__file__).resolve().parents[1] / "recipes" / "tttd" / "examples" / "tttd"


class _SlowJudge(BaseHTTPRequestHandler):
    """Takes 0.2 s per program and records how many it is running at once."""

    lock = threading.Lock()
    running = 0
    peak = 0

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        with self.lock:
            type(self).running += 1
            type(self).peak = max(type(self).peak, type(self).running)
        time.sleep(0.2)
        with self.lock:
            type(self).running -= 1
        body = json.dumps({"score": 1.0, "reason": "ok"}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        return None


@pytest.fixture()
def judge_url():
    _SlowJudge.running = _SlowJudge.peak = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowJudge)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _score_all(scorer: JudgeScorer, count: int) -> list[float]:
    rewards: list[float] = []
    threads = [threading.Thread(target=lambda: rewards.append(scorer(SOLUTION).reward)) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return rewards


@pytest.mark.unit
def test_capped_scorer_never_sends_more_than_the_judge_slots(judge_url: str) -> None:
    # Eight programs of 0.2 s through two slots take four rounds (0.8 s). The
    # 0.5 s timeout would fail the later ones if it counted the wait.
    rewards = _score_all(JudgeScorer(judge_url, timeout_s=0.5, max_in_flight=2), 8)
    assert rewards == [1.0] * 8
    assert _SlowJudge.peak == 2


@pytest.mark.unit
def test_uncapped_scorer_sends_everything_at_once(judge_url: str) -> None:
    _score_all(JudgeScorer(judge_url, timeout_s=5), 6)
    assert _SlowJudge.peak == 6


@pytest.mark.unit
def test_cap_must_be_positive() -> None:
    with pytest.raises(ValueError):
        JudgeScorer("http://127.0.0.1:1", max_in_flight=0)


@pytest.mark.unit
def test_judge_slots_come_from_each_task_config(tmp_path: Path) -> None:
    assert judge_slots(EXAMPLE / "harbor" / "erdos_min_overlap") == 32
    assert judge_slots(EXAMPLE / "harbor" / "circle_packing_26") == 64
    assert judge_slots(tmp_path) is None
