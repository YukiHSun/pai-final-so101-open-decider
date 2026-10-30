#!/usr/bin/env python3
"""Can a small local vision model replace Claude as the "eye"?

Every recorded step image (outputs/claude_*/NN_*.png) has the observation Claude
wrote for it. Duplicate images (the simulator is deterministic) are merged and
their labels decided by majority. Two Ollama models answer the same questions:

  d1      d1:2b through /v1/systemone: an eye call with the image and one short `noul`
          per observation field (P(yes)), then a text-only head call for `next_skill`.
  vl      lfm3-vl:2b through /api/chat: answer every field as yes/no/unclear JSON.
  jev     Jev (TypeSafe API) with image assets: the same eye/head calls as d1, plus a `box`
          question for the cube in each view, scored against the red pixels of the cube.

    .venv/bin/python eval_eye_models.py [d1|vl|jev|both]   # Ollama on 127.0.0.1:11434
    D1_MODEL=d1-omni:2b .venv/bin/python eval_eye_models.py d1
    JEV_MODEL=jev-1.13.0 .venv/bin/python eval_eye_models.py jev   # TYPESAFE_API_KEY in .env

Writes outputs/eye_eval_<model>.json with per-image answers, agreement and latency.
"""
from __future__ import annotations

import base64
import glob
import io
import json
import os
import sys
import time
from collections import Counter
from urllib import request

from PIL import Image

from strands_so101_sim.claude_jev_controller import TASKS

ROOT = os.path.dirname(os.path.abspath(__file__))
OLLAMA = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
D1_MODEL = os.getenv("D1_MODEL", "d1:2b")
VL_MODEL = os.getenv("VL_MODEL", "lfm3-vl:2b")
JEV_URL = os.getenv("JEV_URL", "https://api.typesafe.ai/v1/systemone")
JEV_MODEL = os.getenv("JEV_MODEL", "jev-1.14.0")
# How the 1280x480 step image (oblique | top-down) is shown to the model. A small cube in a
# side-by-side frame was often missed, so each view can be sent on its own and upscaled 2x.
EYE_VIEW = os.getenv("EYE_VIEW", "both2")  # full | left2 | both2
VIEW_TEXT = {"full": "The image shows two camera views of a simulated SO-101 robot arm: left is oblique, right is top-down.",
             "left2": "The image is an oblique camera view of a simulated SO-101 robot arm.",
             "both2": "The first image is an oblique camera view and the second a top-down view of a simulated SO-101 robot arm."}


def views(data):
    if EYE_VIEW == "full":
        return [data]
    frame = Image.open(io.BytesIO(data)).convert("RGB")
    halves = [frame.crop((0, 0, 640, 480)), frame.crop((640, 0, 1280, 480))]
    out = []
    for half in halves[: 1 if EYE_VIEW == "left2" else 2]:
        buf = io.BytesIO()
        half.resize((1280, 960), Image.LANCZOS).save(buf, "PNG")
        out.append(buf.getvalue())
    return out


def b64(images):
    return [base64.b64encode(x).decode() for x in images]


def post(path, body, timeout=300, headers=None):
    url = path if path.startswith("https://") else OLLAMA + path
    req = request.Request(url, data=json.dumps(body).encode(),
                          headers={"content-type": "application/json", **(headers or {})})
    started = time.monotonic()
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read()), time.monotonic() - started


def typesafe_key():
    key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not key and os.path.exists(os.path.join(ROOT, ".env")):
        for line in open(os.path.join(ROOT, ".env")):
            if line.startswith("TYPESAFE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    return key


def cube_box(view):
    """Ground-truth box of the red cube in one view (normalized), or None if hardly visible."""
    img = Image.open(io.BytesIO(view)).convert("RGB")
    w, h = img.size
    pts = [(i % w, i // w) for i, (r, g, b) in enumerate(img.getdata()) if r > 90 and r > 2.5 * g and r > 2.5 * b]
    if len(pts) < 40:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return dict(left=min(xs) / w, right=(max(xs) + 1) / w, top=min(ys) / h, bottom=(max(ys) + 1) / h)


def iou(a, b):
    ix = max(0.0, min(a["right"], b["right"]) - max(a["left"], b["left"]))
    iy = max(0.0, min(a["bottom"], b["bottom"]) - max(a["top"], b["top"]))
    area = lambda x: (x["right"] - x["left"]) * (x["bottom"] - x["top"])
    inter = ix * iy
    return inter / (area(a) + area(b) - inter)


def labelled_images():
    """One entry per unique image; each field's label is the majority over runs."""
    by_image = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "outputs/claude_*/session.json"))
                       + glob.glob(os.path.join(ROOT, "outputs/claude_*/*/session.json"))):
        run = os.path.dirname(path)
        if "negative" in run:  # observations there were deliberately wrong
            continue
        session = json.load(open(path))
        records = list(session["decisions"]) + [r for r in session.get("rejected", []) if r]
        images = sorted(glob.glob(os.path.join(run, "[0-9][0-9]_*.png")))
        for image, rec in zip(images, records):
            data = open(image, "rb").read()
            key = (session.get("task", "cube_lift"), hash(data))
            entry = by_image.setdefault(key, dict(task=key[0], image=os.path.relpath(image, ROOT), data=data,
                                                  phase=rec["phase_before"], votes=[], next_skill=[]))
            entry["votes"].append({k: v for k, v in rec["claude_observation"].items() if k != "notes"})
            entry["next_skill"].append(rec["action"])
    items = []
    for entry in by_image.values():
        fields = entry["votes"][0].keys()
        entry["labels"] = {f: Counter(v[f] for v in entry["votes"]).most_common(1)[0][0] for f in fields}
        entry["expected_skill"] = Counter(entry["next_skill"]).most_common(1)[0][0]
        items.append(entry)
    return items


def ask_d1(item):
    """Eye call (image + one short yes/no per field), then a text-only head call for the next skill."""
    spec = TASKS[item["task"]]
    questions = {f: {"type": "noul", "instructions": q.split(" (yes")[0],
                     "criteria": {"true": "Yes, clearly visible in the images.", "false": "No, or it cannot be seen."}}
                 for f, q in spec.observation_fields.items()}
    eye, eye_s = post("/v1/systemone", {"model": D1_MODEL, "state": VIEW_TEXT[EYE_VIEW], "questions": questions,
                                        "images": b64(views(item["data"]))})
    probs = {f: round(eye["answers"][f]["noul"], 3) for f in spec.observation_fields}
    answers = {f: ("yes" if p >= 0.5 else "no") for f, p in probs.items()}
    state = {"task": spec.goal, "last_completed_skill": item["phase"],
             "visual_observation": {spec.observation_fields[f]: a for f, a in answers.items()}}
    head, head_s = post("/v1/systemone", {"model": D1_MODEL, "state": state, "questions": {
        "next_skill": {"type": "choice", "instructions": spec.skill_instructions, "criteria": spec.skill_criteria}}})
    choice = head["answers"]["next_skill"]
    return dict(answers=answers, p_yes=probs, next_skill=choice["choice"],
                next_skill_confidence=round(choice["confidence"], 3), seconds=round(eye_s, 2),
                head_seconds=round(head_s, 2), usage=eye.get("usage"))


def ask_jev(item):
    """Same eye/head split as d1, sent to Jev with the views as image assets."""
    spec = TASKS[item["task"]]
    headers = {"Authorization": f"Bearer {typesafe_key()}", "TypeSafe-Danger-Zone-Images": "true"}
    imgs = views(item["data"])
    names = ["oblique_view", "top_down_view"][: len(imgs)]
    assets = {f"img-{n}": {"type": "image", "format": "png", "data_base64": d} for n, d in zip(names, b64(imgs))}
    questions = {f: {"type": "noul", "instructions": q.split(" (yes")[0],
                     "criteria": {"true": "Yes, clearly visible in the images.", "false": "No, or it cannot be seen."}}
                 for f, q in spec.observation_fields.items()}
    if JEV_MODEL != "jev-1.13.0":  # box questions need Jev 1.14
        questions.update({f"box_{n}": {"type": "box", "instructions": f"the red cube in `{n}`"} for n in names})
    state = {"scene": VIEW_TEXT[EYE_VIEW], **{n: f"img-{n}" for n in names}}
    eye, eye_s = post(JEV_URL, {"model": JEV_MODEL, "assets": assets, "state": state, "questions": questions},
                      headers=headers)
    probs = {f: round(eye["answers"][f]["noul"], 3) for f in spec.observation_fields}
    answers = {f: ("yes" if p >= 0.5 else "no") for f, p in probs.items()}
    boxes = {}
    for n, img in zip(names, imgs):
        got = eye["answers"].get(f"box_{n}")
        truth = cube_box(img)
        if got:
            boxes[n] = dict(box=got["box"], confidence=round(got["confidence"], 3), truth=truth,
                            iou=round(iou(got["box"], truth), 3) if truth else None)
    head = jev_head(spec, item["phase"], answers, headers["Authorization"])
    # Reference: the same head and gate asked with Claude's observation, to see whether eye errors flip the gate.
    ref = jev_head(spec, item["phase"], item["labels"], headers["Authorization"])
    return dict(answers=answers, p_yes=probs, boxes=boxes, next_skill=head["action"],
                next_skill_confidence=head["confidence"], gate=head["gate"], gate_with_claude_eye=ref["gate"],
                seconds=round(eye_s, 2), head_seconds=head["seconds"], usage=eye.get("usage"))


def jev_head(spec, phase, answers, auth):
    """Text-only head call as in the kit: next_skill plus the gate question for this phase, if any."""
    state = {"task": spec.goal, "last_completed_skill": phase,
             "visual_observation": {spec.observation_fields[f]: a for f, a in answers.items()}}
    questions = {"next_skill": {"type": "choice", "instructions": spec.skill_instructions,
                                "criteria": spec.skill_criteria}}
    gate = spec.gates.get(phase)
    if gate:
        questions[gate.name] = {"type": "noul", "instructions": gate.question,
                                "criteria": {"true": gate.yes, "false": gate.no}}
    head, seconds = post(JEV_URL, {"model": JEV_MODEL, "state": state, "questions": questions},
                         headers={"Authorization": auth})
    choice = head["answers"]["next_skill"]
    out = dict(action=choice["choice"], confidence=round(choice["confidence"], 3), seconds=round(seconds, 2), gate=None)
    if gate:
        prob = round(head["answers"][gate.name]["noul"], 3)
        out["gate"] = dict(name=gate.name, guards=gate.guards, probability=prob,
                           passes=not (choice["choice"] == gate.guards and prob < 0.60))
    return out


def ask_vl(item):
    spec = TASKS[item["task"]]
    listing = "\n".join(f'- "{f}": {q}' for f, q in spec.observation_fields.items())
    prompt = (f"{VIEW_TEXT[EYE_VIEW]} Answer each question about the images with exactly one of \"yes\", \"no\" or \"unclear\". "
              f"Reply with one JSON object whose keys are the question ids.\n{listing}")
    reply, seconds = post("/api/chat", {"model": VL_MODEL, "stream": False, "format": "json",
                                        "options": {"temperature": 0.1, "top_p": 1.0, "top_k": 50,
                                                    "repeat_penalty": 1.1},
                                        "messages": [{"role": "user", "content": prompt,
                                                      "images": b64(views(item["data"]))}]})
    text = reply["message"]["content"]
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raw = {}
    answers = {f: str(raw.get(f, "missing")).strip().lower() for f in spec.observation_fields}
    return dict(answers=answers, raw=text[:400], seconds=round(seconds, 2))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    items = labelled_images()
    for name, ask in (("d1", ask_d1), ("vl", ask_vl), ("jev", ask_jev)):
        if which != name and not (which == "both" and name != "jev"):
            continue
        rows, per_field = [], Counter()
        totals = Counter()
        for item in items:
            result = ask(item)
            agree = {f: result["answers"][f] == lab for f, lab in item["labels"].items() if lab != "unclear"}
            for f, ok in agree.items():
                per_field[(f, ok)] += 1
            totals["fields"] += len(agree)
            totals["agree"] += sum(agree.values())
            if "next_skill" in result:
                totals["skill_ok"] += result["next_skill"] == item["expected_skill"]
            rows.append(dict(task=item["task"], image=item["image"], phase=item["phase"], labels=item["labels"],
                             expected_skill=item["expected_skill"], agree=agree, **result))
            wrong = [f for f, ok in agree.items() if not ok]
            extra = f" next={result['next_skill']}({result['next_skill_confidence']})" if "next_skill" in result else ""
            print(f"  {name} {item['task']:10s} {item['phase']:15s} {sum(agree.values())}/{len(agree)} "
                  f"{result['seconds']:.2f}s{extra} wrong={wrong}")
        fields = sorted({f for f, _ in per_field})
        summary = dict(model=dict(d1=D1_MODEL, vl=VL_MODEL, jev=JEV_MODEL)[name], images=len(items),
                       agreement=round(totals["agree"] / totals["fields"], 3),
                       per_field={f: f"{per_field[(f, True)]}/{per_field[(f, True)] + per_field[(f, False)]}" for f in fields},
                       median_seconds=sorted(r["seconds"] for r in rows)[len(rows) // 2])
        if name in ("d1", "jev"):
            summary["next_skill_correct"] = f"{totals['skill_ok']}/{len(items)}"
        ious = sorted(b["iou"] for r in rows for b in r.get("boxes", {}).values() if b["iou"] is not None)
        if ious:
            summary["box_iou_median"] = ious[len(ious) // 2]
            summary["box_iou_at_least_0.5"] = f"{sum(x >= 0.5 for x in ious)}/{len(ious)}"
        print(json.dumps(summary, indent=1))
        summary["eye_view"] = EYE_VIEW
        tag = summary["model"].replace(":", "-") + "_" + EYE_VIEW
        out = os.path.join(ROOT, "outputs", f"eye_eval_{tag}.json")
        json.dump(dict(summary=summary, rows=rows), open(out, "w"), indent=1, ensure_ascii=False)
        print("saved", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
