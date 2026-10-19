"""DeciderClient wire-format tests; no server or model needed."""
import io
import json
import os
import unittest
from unittest.mock import patch

from strands_so101_sim.claude_jev_controller import JevController, check_gates
from strands_so101_sim.decider_client import DeciderClient
from strands_so101_sim.tests.test_claude_jev import observation, state

REPLY = {
    "model": "strands-decider-2B-qwen3.5-v1-2610",
    "answers": {
        "next_skill": {"type": "choice", "choice": "lift", "confidence": 0.9,
                       "probabilities": {"lift": 0.9, "done": 0.1}},
        "grasp_secure": {"type": "noul", "noul": 0.95},
    },
    "usage": {"input_tokens": 500, "output_tokens": 1},
    "latency_ms": 120.0,
}


class DeciderClientTest(unittest.TestCase):
    def setUp(self):
        self.sent = []

        def fake_urlopen(req, timeout):
            self.sent.append((req.full_url, json.loads(req.data)))
            return io.BytesIO(json.dumps(REPLY).encode())

        patcher = patch("strands_so101_sim.decider_client.request.urlopen", fake_urlopen)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_controller_runs_unchanged_on_decider(self):
        ctl = JevController(client=DeciderClient("http://127.0.0.1:8765/"))
        decision = ctl.decide(observation(), state("close_gripper", "closed"))
        check_gates(decision)
        url, body = self.sent[0]
        self.assertEqual(url, "http://127.0.0.1:8765/v1/systemone")
        self.assertNotIn("model", body)
        self.assertEqual(body["questions"]["next_skill"]["type"], "choice")
        self.assertEqual(set(body["questions"]["grasp_secure"]["criteria"]), {"true", "false"})
        self.assertEqual(body["state"]["last_completed_skill"], "close_gripper")
        self.assertEqual((decision.action, decision.gate["probability"]), ("lift", 0.95))
        self.assertEqual(decision.model, REPLY["model"])

    def test_env_var_selects_decider(self):
        with patch.dict(os.environ, {"DECIDER_URL": "http://127.0.0.1:9"}, clear=False):
            self.assertIsInstance(JevController().client, DeciderClient)


if __name__ == "__main__":
    unittest.main()
