"""Serve recorded Elephant-1 answers on POST /v1/systemone, so the SO-101 kit scripts run unchanged.

Each request body is matched exactly (canonical JSON) against the bodies answered on Colab by
answer_bodies.py; an unknown body is a 404, so a kit run can never silently get a wrong answer.

    python scripts/elephant1_colab/replay_server.py outputs/elephant1_colab/answers/*.json --port 8745
    DECIDER_URL=http://127.0.0.1:8745 .venv/bin/python compare_decider_vs_jev.py out.json
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def key(body):
    return json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def placeholder(body):
    """A uniform answer of the right shape (collect mode only)."""
    answers = {}
    for name, q in body["questions"].items():
        if q["type"] == "noul":
            answers[name] = dict(type="noul", noul=0.5)
        else:
            keys = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
            probs = {k: 1 / len(keys) for k in keys}
            answers[name] = dict(type=q["type"], choice=keys[0], probabilities=probs, confidence=0.0)
            if q["type"] == "score":
                answers[name].update(score=(len(keys) - 1) / 2)
    return dict(model="placeholder", answers=answers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers", nargs="+")
    ap.add_argument("--port", type=int, default=8745)
    ap.add_argument("--collect", metavar="BODIES_JSON",
                    help="collect mode: save every unanswered body to this file and answer it with a "
                         "uniform placeholder, so a kit run can be driven once only to gather its bodies "
                         "(its results are meaningless)")
    args = ap.parse_args()
    collected = []
    table, meta = {}, {}
    for path in args.answers:
        data = json.load(open(path))
        meta = {k: v for k, v in data.items() if k != "rows"}
        for row in data["rows"]:
            if "response" in row:
                table[key(row["body"])] = (row["response"], row.get("ms"))
    print(f"{len(table)} recorded answers ({meta.get('model')} on {meta.get('device')})", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, dict(status="ok", replay=True, answers=len(table), **meta))
            else:
                self._send(404, dict(error="not found"))

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self._send(404, dict(error="not found"))
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
            hit = table.get(key(body))
            if hit is None and args.collect:
                if key(body) not in {key(b) for b in collected}:
                    collected.append(body)
                    with open(args.collect, "w") as f:
                        json.dump(collected, f, ensure_ascii=False, indent=1)
                print(f"COLLECT ({len(collected)} new bodies)", flush=True)
                return self._send(200, placeholder(body))
            if hit is None:
                print("MISS: body not answered on Colab", flush=True)
                return self._send(404, dict(error="this body was not answered on Colab"))
            response, ms = hit
            print(f"hit ({ms} ms on Colab)", flush=True)
            self._send(200, response)

        def log_message(self, *a):
            pass

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
