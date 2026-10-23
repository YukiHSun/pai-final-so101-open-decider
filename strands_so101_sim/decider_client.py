"""Drop-in stand-in for TypeSafeClient backed by a local strands-decider server.

strands-decider (Strands Labs, Apache-2.0) serves the same System One request shape
as Jev at POST /v1/systemone. This client exposes the one method JevController uses,
`system_one(state, questions, model=...)`, and returns an object with the attributes
the controller reads, so the controller, gates and runner stay unchanged.

Start the server in its own environment (it needs torch, the simulator does not):

    .venv-decider-main/bin/strands-decider serve StrandsAgents/strands-decider-E4B-gemma4-v1-2610 \
        --device mps --port 8765

then set DECIDER_URL=http://127.0.0.1:8765 for the runner.

Hosted decision models on OpenRouter use the same request and answer shape at
POST https://openrouter.ai/api/alpha/decisions, with a bearer key and a model name:

    DECIDER_URL=https://openrouter.ai/api/alpha/decisions
    DECIDER_MODEL=baseten/mammoth-1:free      # OPENROUTER_API_KEY (or DECIDER_API_KEY) from .env
"""
from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace
from urllib import error, request

RETRY_STATUS = {429, 500, 502, 503, 504}


def _dump(question):
    """typesafe_sdk question objects are pydantic models with the same wire shape."""
    return question.model_dump() if hasattr(question, "model_dump") else dict(question)


class DeciderClient:
    def __init__(self, base_url="http://127.0.0.1:8765", timeout=60.0, api_key=None, model=None, retries=4):
        base = base_url.rstrip("/")
        # A full decisions endpoint (OpenRouter) is used as given; a bare server URL gets /v1/systemone.
        self.url = base if base.endswith(("/decisions", "/v1/systemone")) else base + "/v1/systemone"
        self.timeout = timeout
        self.api_key = api_key
        self.model = model
        self.retries = retries

    @classmethod
    def from_env(cls):
        url = os.environ["DECIDER_URL"].strip()
        key = os.getenv("DECIDER_API_KEY", "").strip()
        if not key and "openrouter.ai" in url:
            key = os.getenv("OPENROUTER_API_KEY", "").strip()
        return cls(url, api_key=key or None, model=os.getenv("DECIDER_MODEL", "").strip() or None)

    def _post(self, body):
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        for attempt in range(self.retries + 1):
            req = request.Request(self.url, data=json.dumps(body).encode(), headers=headers)
            try:
                with request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except error.HTTPError as exc:
                if exc.code not in RETRY_STATUS or attempt == self.retries:
                    raise
                time.sleep(2 ** attempt)

    def system_one(self, state, questions, model=None):
        # The per-call `model` is the Jev name the controller always passes; only a model
        # configured on this client is sent, so single-model local servers see no model field.
        body = {"state": state, "questions": {k: _dump(q) for k, q in questions.items()}}
        if self.model:
            body["model"] = self.model
        data = self._post(body)
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
