"""Score one candidate's answers, blind, against the use case's own rubric.

THE JUDGE IS PINNED AND MUST NOT BE A CANDIDATE. Left floating, evaluating a
candidate would make it its own examiner — and that has two distinct failure
modes, both invisible in the output:

    CONFOUNDING       a score that moved could be a better writer or a more
                      generous reader, and the number cannot separate them.
    SELF-PREFERENCE   models score their own family's prose higher. With the
                      judge floating, every candidate marks its own paper.

`check_judge_not_candidate` refuses the run outright rather than warning,
because a leaderboard produced this way looks completely normal — the bias
shows up as half a point in the wrong place and nothing reveals it after the
fact.

BLIND AND ONE AT A TIME. The judge sees the test case, the reference answer (if
any) and one candidate's output — never which model wrote it, and never two
candidates' answers side by side. A judge shown two answers reasons about which
is "the improved one"; shown one, it can only read it.

THE RUBRIC IS VERBATIM FROM THE USE CASE. This module invents no criteria of
its own — it renders whatever `config.py` loaded and asks the judge to score
exactly that, because a generic rubric judging an arbitrary task is the thing
that makes a ranking meaningless without anyone noticing.
"""
from typing import Optional

from . import client as client_module
from .catalogue import key as logical_key
from .config import Criterion, TestCase, UseCase


class JudgeIsCandidate(Exception):
    """The judge is also being benchmarked. Refused, not reported."""


def check_judge_not_candidate(judge_model: str, candidates: list) -> None:
    judge_key = logical_key(judge_model)
    clash = [c for c in candidates if logical_key(c) == judge_key]
    if clash:
        raise JudgeIsCandidate(
            f"the judge ({judge_model}) is also a candidate ({', '.join(clash)}). "
            f"It would mark its own paper, and self-preference does not show up "
            f"in the score. Pick a judge outside the field.")


_PROMPT = """You are scoring ONE model's answer to a task, against a rubric someone who
owns this task wrote themselves. Score only what the rubric asks for.

TASK GIVEN TO THE MODEL:
{input}
{reference_block}
THE MODEL'S ANSWER:
{output}

RUBRIC — score each of these 1 (fails) to 5 (excellent), and say why in one
sentence citing something specific in the answer:
{rubric_block}

Return ONLY valid JSON, no markdown fences:
{{
  "scores": {{{score_keys}}},
  "reasons": {{{reason_keys}}},
  "wouldShip": true
}}

"wouldShip": would you hand this answer to a real user as-is, unedited?
"""


def _rubric_block(criteria: list) -> str:
    return "\n".join(f"  - {{\"{c.id}\"}}: {c.description} (weight {c.weight})"
                     for c in criteria)


async def score_one(model_id: str, judge_model: str, uc: UseCase, tc: TestCase,
                    output: str) -> dict:
    """One judged answer. Returns {weighted, scores, reasons, wouldShip} or a
    failure dict — a failed judgement is recorded, not raised, so one bad call
    does not lose the whole candidate's leaderboard row."""
    criteria = tc.rubric or uc.rubric
    reference_block = (f"\nA REFERENCE ANSWER, for comparison only — the model's "
                       f"answer need not match it word for word:\n{tc.reference}\n"
                       if tc.reference else "")
    prompt = _PROMPT.format(
        input=tc.input, reference_block=reference_block, output=output,
        rubric_block=_rubric_block(criteria),
        score_keys=", ".join(f'"{c.id}": 3' for c in criteria),
        reason_keys=", ".join(f'"{c.id}": "why"' for c in criteria))

    try:
        data = await client_module.call_json(
            prompt, model=judge_model, label=f"judge[{model_id}:{tc.id}]",
            required=("scores",), temperature=0.1, max_tokens=1200)
    except Exception as exc:                        # noqa: BLE001
        return {"testCase": tc.id, "status": "judge_failed", "error": str(exc)[:300]}

    scores = {}
    for c in criteria:
        try:
            scores[c.id] = max(1.0, min(5.0, float((data.get("scores") or {}).get(c.id, 3))))
        except (TypeError, ValueError):
            scores[c.id] = 3.0
    total_weight = sum(c.weight for c in criteria) or 1.0
    weighted = round(sum(scores[c.id] * c.weight for c in criteria) / total_weight, 3)

    return {"testCase": tc.id, "status": "ok", "scores": scores,
            "weighted": weighted, "reasons": data.get("reasons") or {},
            "wouldShip": bool(data.get("wouldShip"))}
