"""Install the Elephant-1 serving runtime on a Colab GPU VM (runs inside the Colab kernel).

Mirrors the model repo's own setup.sh (baselabs/elephant-1 @ b3f8a93): OneJev @ 81ce62f with the
model's serve.patch applied, plus the pinned transformers stack and the linear-attention kernels.
Colab's preinstalled CUDA torch is kept (setup.sh does the same with the docker image's torch).

Expects /content/elephant/serve.patch and /content/elephant/constraints.txt (uploaded by run_colab.sh).
"""
import subprocess
import sys
from pathlib import Path

ONEJEV = "81ce62f1597c91e46767d4d02d6ac2e18534fe94"
WORK = Path("/content/elephant")
SRC = WORK / "OneJev"
# Only the packages the runtime needs are pinned; the full constraints file also pins Colab's
# own packages (numpy, jupyter, ...) and would fight the image.
KEEP = ("transformers", "tokenizers", "safetensors", "accelerate", "huggingface_hub", "hf-xet",
        "fla-core", "flash-linear-attention", "causal-conv1d", "pydantic", "pydantic_core", "einops")


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


pins = [line for line in (WORK / "constraints.txt").read_text().splitlines()
        if line.split("==")[0].strip().lower() in {k.lower() for k in KEEP}]
(WORK / "pins.txt").write_text("\n".join(pins) + "\n")
print("pins:", ", ".join(pins), flush=True)

if not SRC.exists():
    run(["git", "clone", "-q", "https://github.com/OmniJev/OneJev.git", str(SRC)])
run(["git", "-C", str(SRC), "checkout", "-q", "-f", ONEJEV])
run(["git", "-C", str(SRC), "clean", "-fdq"])
run(["git", "-C", str(SRC), "apply", str(WORK / "serve.patch")])

pip = [sys.executable, "-m", "pip", "install", "-q", "-c", str(WORK / "pins.txt")]
run(pip + ["-e", str(SRC), "transformers", "tokenizers", "safetensors", "accelerate", "huggingface_hub",
           "flash-linear-attention", "einops"])
# causal-conv1d compiles CUDA code; without it transformers falls back to a torch convolution.
try:
    run(pip + ["--no-build-isolation", "causal-conv1d"], timeout=1800)
except Exception as exc:  # noqa: BLE001
    print(f"causal-conv1d not installed ({exc}); the torch fallback convolution will be used", flush=True)

import importlib.metadata as md  # noqa: E402

for name in ("torch", "transformers", "flash-linear-attention", "causal-conv1d", "qev"):
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}: not installed")
print("SETUP_DONE", flush=True)
