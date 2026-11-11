#!/usr/bin/env bash
# Answer recorded /v1/systemone bodies with the official Elephant-1 runtime on a Colab GPU.
#
#   scripts/elephant1_colab/run_colab.sh [GPU]          # GPU: L4 (default) or A100 / H100 / G4
#
# Uploads the model repo's serve.patch and pins (fetched from the Hub at the pinned revision),
# the two VM scripts and every
# outputs/elephant1_colab/bodies/*.json. On the VM, setup_vm.py (OneJev @81ce62f + serve.patch)
# and then answer_bodies.py run as one detached background job, so a dropped CLI connection
# does not stop them; this script only polls the job's log with short execs. The answers are
# downloaded to outputs/elephant1_colab/answers/. The VM is released at the end (trap), also on
# errors; KEEP=1 keeps it. An existing session named $SESSION is reused.
set -euo pipefail

GPU=${1:-L4}
SESSION=${SESSION:-elephant1}
POLL=${POLL:-60}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
BODIES=${BODIES:-"$ROOT/outputs/elephant1_colab/bodies"}
ANSWERS="$ROOT/outputs/elephant1_colab/answers"
mkdir -p "$ANSWERS"

# The model repo's own serve.patch and pins, fetched at the pinned revision.
REV=b3f8a93462fb1051a9de031e94a7ff380e254072
for f in serve.patch constraints.txt; do
  [ -s "$HERE/$f" ] || curl -sfL -o "$HERE/$f" "https://huggingface.co/baselabs/elephant-1/resolve/$REV/$f"
done

say() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
quiet() { grep -a -v -E '^\[colab\] (A new version|Run |You can run|To silence)|^\s*$' || true; }

# Short exec with a hard timeout and retries: the CLI can hang after "Connection was lost".
vm() {
  local code=$1 tries=${2:-5} out
  for ((i = 1; i <= tries; i++)); do
    if out=$(printf '%s\n' "$code" | timeout 120 colab exec -s "$SESSION" 2>&1); then
      printf '%s\n' "$out" | quiet
      return 0
    fi
    echo "(exec failed, try $i/$tries)" >&2
    sleep 10
  done
  return 1
}

put() {
  for ((i = 1; i <= 5; i++)); do
    timeout 300 colab upload -s "$SESSION" "$1" "$2" 2>&1 | quiet && return 0
    sleep 10
  done
  return 1
}

release() {
  if [ "${KEEP:-0}" = 1 ]; then
    say "keeping session $SESSION (colab stop -s $SESSION when done)"
  else
    say "releasing the VM"
    timeout 120 colab stop -s "$SESSION" 2>&1 | quiet
  fi
}

if ! timeout 120 colab sessions 2>/dev/null | grep -q "\[$SESSION\]"; then
  say "allocating a $GPU VM ($SESSION)"
  timeout 600 colab new -s "$SESSION" --gpu "$GPU" 2>&1 | quiet
  sleep 20
fi
trap release EXIT

vm 'import subprocess, os; os.makedirs("/content/elephant/bodies", exist_ok=True); print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout)'

say "uploading the patch, pins, scripts and bodies"
for f in serve.patch constraints.txt setup_vm.py answer_bodies.py; do
  put "$HERE/$f" "/content/elephant/$f"
done
for f in "$BODIES"/*.json; do
  put "$f" "/content/elephant/bodies/$(basename "$f")"
done

say "starting the job on the VM (setup, model download, answers)"
vm 'import subprocess
subprocess.Popen("cd /content/elephant && (python setup_vm.py && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python answer_bodies.py; echo JOB_EXIT=$?) > job.log 2>&1",
                 shell=True, start_new_session=True)
print("started")'

status='import pathlib
p = pathlib.Path("/content/elephant/job.log")
lines = p.read_text(errors="replace").splitlines() if p.exists() else []
print("\n".join(l[:200] for l in lines[-4:]))'
while true; do
  sleep "$POLL"
  say "job log"
  out=$(vm "$status" 2 || true)
  printf '%s\n' "$out"
  if grep -q "JOB_EXIT=" <<<"$out"; then
    grep -q "JOB_EXIT=0" <<<"$out" || { say "the job failed; full log:"; vm 'print(open("/content/elephant/job.log").read()[-6000:])'; exit 1; }
    break
  fi
done

say "downloading the answers and the job log"
for f in "$BODIES"/*.json; do
  name=$(basename "$f")
  timeout 300 colab download -s "$SESSION" "/content/elephant/answers/$name" "$ANSWERS/$name" 2>&1 | quiet
done
timeout 300 colab download -s "$SESSION" /content/elephant/job.log "$ANSWERS/job.log" 2>&1 | quiet
ls -la "$ANSWERS"
