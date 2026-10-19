"""Drop-in stand-in for TypeSafeClient backed by a local strands-decider server.

strands-decider (Strands Labs, Apache-2.0) serves the same System One request shape
as Jev at POST /v1/systemone. This client exposes the one method JevController uses,
`system_one(state, questions, model=...)`, and returns an object with the attributes
the controller reads, so the controller, gates and runner stay unchanged.

Start the server in its own environment (it needs torch, the simulator does not):

    .venv-decider-main/bin/strands-decider serve StrandsAgents/strands-decider-E4B-gemma4-v1-2610 \
        --device mps --port 8765

then set DECIDER_URL=http://127.0.0.1:8765 for the runner.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from urllib import request


def _dump(question):
    """typesafe_sdk question objects are pydantic models with the same wire shape."""
    return question.model_dump() if hasattr(question, "model_dump") else dict(question)


class DeciderClient:
    def __init__(self, base_url="http://127.0.0.1:8765", timeout=60.0):
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.timeout = timeout

    def system_one(self, state, questions, model=None):
        # `model` is accepted for signature parity; the server hosts a single model.
        body = {"state": state, "questions": {k: _dump(q) for k, q in questions.items()}}
        req = request.Request(self.url, data=json.dumps(body).encode(),
                              headers={"content-type": "application/json"})
        with request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        answers = {}
        for name, a in data["answers"].items():
            answers[name] = SimpleNamespace(
                choice=a.get("choice"), confidence=a.get("confidence"),
                probabilities=a.get("probabilities", {}), noul=a.get("noul"))
        usage = data.get("usage", {})
        return SimpleNamespace(
            model=data.get("model", "strands-decider"), answers=answers, request_id=None,
            latency_ms=data.get("latency_ms"),
            usage=SimpleNamespace(input_tokens=usage.get("input_tokens", 0),
                                  output_tokens=usage.get("output_tokens", 0)))
