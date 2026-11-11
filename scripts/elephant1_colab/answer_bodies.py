"""Answer recorded /v1/systemone bodies with the official Elephant-1 engine (runs inside the Colab kernel).

Reads every /content/elephant/bodies/<name>.json (a JSON list of {"state", "questions"} bodies) and
writes /content/elephant/answers/<name>.json with, per body, the request, the engine's response and
its timing. The model is loaded once and kept in the kernel (`ENGINE`), so a second run reuses it.

The model is loaded from a local directory on purpose: qev reads readout.json and the
block-bidirectional attention override in config.json only for a local path, and would otherwise
serve the checkpoint with plain causal attention.
"""
import json
import sys
import time
from pathlib import Path

# The editable install's .pth is only read at interpreter start; this kernel started before it.
sys.path.insert(0, "/content/elephant/OneJev")

REPO = "baselabs/elephant-1"
REVISION = "b3f8a93462fb1051a9de031e94a7ff380e254072"
WORK = Path("/content/elephant")
MODEL_DIR = WORK / "elephant-1"

if "ENGINE" not in globals():
    from huggingface_hub import snapshot_download

    t = time.perf_counter()
    snapshot_download(REPO, revision=REVISION, local_dir=str(MODEL_DIR),
                      allow_patterns=["*.json", "*.jinja", "model.safetensors"])
    print(f"downloaded in {time.perf_counter() - t:.0f}s", flush=True)

    import torch
    from qev.engine import DecisionEngine

    t = time.perf_counter()
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_gb >= 30:
        ENGINE = DecisionEngine(str(MODEL_DIR), device="cuda:0")
        HEAD_UPCAST = "qev (on GPU)"
    else:
        # On a 24 GB card (L4) qev's own upcast (clone in bf16, then cast, all on the GPU) needs
        # about 6 GB at once and runs out of memory. Load the head in bf16, then do the same exact
        # bf16 -> fp32 cast through the CPU and install the same pre-hook as qev's _upcast_head;
        # the resulting head and forward are identical. The fork mode is then selected again
        # with the fp32 head, as qev does at load.
        ENGINE = DecisionEngine(str(MODEL_DIR), device="cuda:0", head_dtype="bfloat16")
        head = ENGINE.model.get_output_embeddings()
        assert getattr(head, "bias", None) is None
        w = head.weight.detach().to("cpu")
        head.weight = None
        torch.cuda.empty_cache()
        head.weight = torch.nn.Parameter(w.to(torch.float32).to("cuda:0"), requires_grad=False)
        del w
        head.register_forward_pre_hook(lambda module, args: (args[0].to(torch.float32),) + tuple(args[1:]))
        ENGINE.head_dtype = "float32"
        ENGINE.info.extra["head_dtype"] = "float32"
        ENGINE._select_fork_mode()
        HEAD_UPCAST = "same cast via CPU (24 GB card)"
    info = ENGINE.info
    print(f"GPU {torch.cuda.get_device_name(0)} {total_gb:.0f} GB, head fp32 ({HEAD_UPCAST}), "
          f"memory {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
    print(f"loaded in {time.perf_counter() - t:.0f}s: attention={info.extra['attention']} "
          f"readout={info.extra['readout']} fork={info.fork_mode} (diff {info.fork_max_abs_diff}) "
          f"calibration={info.extra['calibration']}", flush=True)
    if info.extra["attention"] != "noncausal":
        raise SystemExit("attention override not applied: the checkpoint would be served causally")

import torch  # noqa: E402
from qev.schema import SystemOneRequest  # noqa: E402

out_dir = WORK / "answers"
out_dir.mkdir(exist_ok=True)
info = ENGINE.info
for path in sorted((WORK / "bodies").glob("*.json")):
    bodies = json.loads(path.read_text())
    rows = []
    for i, body in enumerate(bodies):
        t = time.perf_counter()
        try:
            resp, meta = ENGINE.decide(SystemOneRequest(**body))
            torch.cuda.synchronize()
            row = dict(body=body, response=resp.model_dump(exclude_none=True), meta=meta)
        except Exception as exc:  # noqa: BLE001
            row = dict(body=body, error=f"{type(exc).__name__}: {exc}")
        row["ms"] = round((time.perf_counter() - t) * 1000, 1)
        rows.append(row)
        print(f"{path.stem} {i}: {'ERROR ' + row['error'] if 'error' in row else 'ok'} {row['ms']:.0f} ms", flush=True)
    result = dict(model=REPO, revision=REVISION, runtime="OneJev qev @81ce62f + serve.patch",
                  device=torch.cuda.get_device_name(0), torch=torch.__version__,
                  transformers=info.transformers_version, attention=info.extra["attention"],
                  readout=info.extra["readout"], fork_mode=info.fork_mode,
                  fork_max_abs_diff=info.fork_max_abs_diff, calibration=info.extra["calibration"],
                  head_dtype=info.extra["head_dtype"], head_upcast=HEAD_UPCAST, rows=rows)
    (out_dir / path.name).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"wrote {out_dir / path.name} ({len(rows)} bodies)", flush=True)
print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
print("ANSWERS_DONE", flush=True)
