"""compare_prompt_order.py, with a skill outside the task's skills recorded as a wrong, stopped
answer instead of ending the run (the kit's controller raises on it). The kit is not changed.

usage (from the kit root): python scripts/run_order_tolerant.py <wording> <tag>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import compare_prompt_order as cpo  # noqa: E402
from strands_so101_sim.claude_jev_controller import JevControllerError  # noqa: E402

original_ask = cpo.ask
off_task = []


def tolerant_ask(ctl, rec, wording):
    try:
        return original_ask(ctl, rec, wording)
    except JevControllerError as exc:
        if "Unknown action" not in str(exc):
            raise
        action = str(exc).rsplit(": ", 1)[-1]
        off_task.append(dict(task=ctl.spec.name, phase=rec["phase_before"], action=action))
        return dict(action=f"{action} (not a {ctl.spec.name} skill)", confidence=0.0,
                    probabilities={}, gate=None, verdict="stop", latency_ms=None, model="clef-flash")


cpo.ask = tolerant_ask
sys.argv = ["compare_prompt_order.py", *sys.argv[1:]]
cpo.main()
print("off-task answers:", off_task)
