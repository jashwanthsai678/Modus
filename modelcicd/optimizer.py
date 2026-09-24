"""Teacher -> student prompt optimization — an ADDITIVE feature, opt in only.

Every other module in this project answers "which EXISTING model should I
use". This one asks a different question about a use case that has already
been benched: can a CHEAP candidate be made to score like the run's best
("gold") candidate, not by training anything, but by rewriting the prompt
it's given? The gold candidate's own answers — and the judge's own stated
reasons for scoring them well — are the teaching material; a model is asked
to turn that into a better prompt for the cheap candidate, which is then
re-run and re-scored, in a bounded, self-refining loop.

NOTHING HERE CHANGES `resolve()`, `state.py`'s schema, OR ANY EXISTING
LEADERBOARD. This module only ever reads a bench that already ran
(`bench.py`/`rank.py`'s saved output) and produces its own separate
artifact under `out/optimize/`. Turning an optimized prompt into what a
use case actually asks for in production is a distinct, later, human
decision — the same boundary this project already draws between
`state.approve()` and `apply-code-patch`.

A COPY OF THE USE CASE IS ALL THAT'S EVER MUTATED. `sandbox.run_one` reads
its system prompt off the `UseCase` object it's given, never off disk —
so `dataclasses.replace(uc, system_prompt=...)` produces a throwaway
candidate prompt with the SAME rubric/test cases/assertions, and the real
`UseCase` (and the `use_case.yaml` it came from) is never touched.
"""
import dataclasses
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import client as client_module
from . import config as config_module
from . import judge as judge_module
from . import rank as rank_module
from . import sandbox as sandbox_module
from .config import UseCase

# Tier order to search, cheapest first — same list `rank.py` already sorts
# boards by, so "the cheap candidate" means the same thing here it does on
# every leaderboard in this project.
_TIER_ORDER = rank_module.TIER_ORDER


class NothingToOptimize(ValueError):
    """Raised when a run has no gold/cheap pair to work from — e.g. only
    one tier ran, or the gold candidate IS the cheapest tier's best."""


def select_gold_and_cheap(board: dict, *, cheap_model: Optional[str] = None
                          ) -> tuple:
    """(gold_model, gold_score, cheap_model) from an already-ranked board.

    Gold is simply `rank.best_overall` — the same "best candidate,
    regardless of tier" this project already computes for the pending-
    candidate check in `state.record_run`. Cheap is either named
    explicitly, or the top-scoring row of the cheapest tier that actually
    ran and isn't the gold model itself."""
    gold = rank_module.best_overall(board)
    if not gold:
        raise NothingToOptimize(
            "this run has no scored candidate to learn from — run a bench "
            "with at least one successful candidate first.")

    if cheap_model:
        return gold["model"], gold["score"], cheap_model

    for tier in _TIER_ORDER:
        for row in board.get("tiers", {}).get(tier, []):
            if row["score"] is not None and row["model"] != gold["model"]:
                return gold["model"], gold["score"], row["model"]

    raise NothingToOptimize(
        f"nothing cheaper than the gold candidate ({gold['model']}) to "
        f"optimize — either only one tier ran, or the gold candidate is "
        f"already the cheapest tier's best. Pass --cheap-model explicitly "
        f"to target a specific candidate anyway.")


def _gold_examples(bench_result: dict, gold_model: str, uc: UseCase) -> list:
    """Each test case's {input, output, reasons, scores} for the gold
    model — the teaching material `propose_prompt` reads. Skips test cases
    the gold candidate itself failed; there's nothing to learn from a
    failure."""
    by_model = {r["model"]: r for r in bench_result.get("results") or []}
    entry = by_model.get(gold_model)
    if not entry:
        raise NothingToOptimize(
            f"gold model {gold_model!r} has no results in this bench file "
            f"— pass --from-run pointing at the run that actually scored it.")
    by_tc = {tc["testCase"]: tc for tc in entry.get("testCases") or []}

    examples = []
    for tc in uc.test_cases:
        e = by_tc.get(tc.id)
        if e and e.get("status") == "ok" and e.get("output"):
            examples.append({"id": tc.id, "input": tc.input, "output": e["output"],
                             "reasons": e.get("reasons") or {},
                             "scores": e.get("scores") or {}})
    return examples


_PROPOSE_PROMPT = """You are improving the SYSTEM PROMPT given to a cheaper, weaker model, so \
its answers get as close as possible to a stronger model's answers below — \
without becoming the stronger model. You may rewrite instructions, add \
constraints, or add worked examples (few-shot) drawn from the material \
below. Keep the same task and the same required output shape.

RUBRIC the answers are scored against (do not invent new criteria):
{rubric_block}

CURRENT SYSTEM PROMPT:
{current_prompt}

STRONG MODEL'S OWN ANSWERS, AND WHY A JUDGE SCORED THEM WELL:
{gold_block}
{feedback_block}
Return ONLY valid JSON, no markdown fences:
{{"prompt": "the full revised system prompt"}}
"""


def _rubric_block(uc: UseCase) -> str:
    seen, lines = set(), []
    for c in uc.rubric or []:
        if c.id not in seen:
            lines.append(f"  - {c.id}: {c.description} (weight {c.weight})")
            seen.add(c.id)
    return "\n".join(lines)


def _gold_block(examples: list) -> str:
    parts = []
    for e in examples:
        reasons = "; ".join(f"{k}: {v}" for k, v in (e["reasons"] or {}).items())
        parts.append(f"- input: {e['input']}\n  strong model's answer: {e['output']}"
                     + (f"\n  why it scored well: {reasons}" if reasons else ""))
    return "\n".join(parts)


def _feedback_block(prior_feedback: Optional[list]) -> str:
    if not prior_feedback:
        return ""
    parts = ["\nTHE CHEAP MODEL'S LAST ATTEMPT WITH YOUR PREVIOUS PROMPT, AND WHAT "
            "FELL SHORT — fix these specifically:"]
    for f in prior_feedback:
        reasons = "; ".join(f"{k}: {v}" for k, v in (f.get("reasons") or {}).items())
        parts.append(f"- input: {f['input']}\n  cheap model's answer: {f.get('output')}"
                     + (f"\n  judge's reasons: {reasons}" if reasons else ""))
    return "\n".join(parts) + "\n"


async def propose_prompt(uc: UseCase, gold_examples: list, current_prompt: str, *,
                         optimizer_model: str, prior_feedback: Optional[list] = None,
                         cache=None) -> str:
    """One call: read the gold answers + the judge's own reasoning (that
    reasoning already IS the analysis of "what made them good" — nothing
    is gained by spending a second call re-deriving it), and write a
    revised prompt for the cheap model. From the second iteration on, the
    cheap model's own last attempt and where the judge marked it down are
    folded in too, so each iteration is a self-refine step, not a blind
    retry."""
    prompt = _PROPOSE_PROMPT.format(
        rubric_block=_rubric_block(uc), current_prompt=current_prompt,
        gold_block=_gold_block(gold_examples),
        feedback_block=_feedback_block(prior_feedback))
    data = await client_module.call_json(
        prompt, model=optimizer_model, label=f"optimizer[{uc.name}]",
        required=("prompt",), temperature=0.4,
        max_tokens=max(uc.max_tokens * 2, 1500), cache=cache)
    revised = (data.get("prompt") or "").strip()
    return revised or current_prompt


async def _score_with_prompt(uc: UseCase, tc, cheap_model: str, prompt: str, *,
                             provider: str, judge_model: str, cache=None) -> dict:
    """One test case, one candidate prompt: generate through a throwaway
    prompt-swapped copy of `uc`, then judge against the REAL `uc` — the
    rubric/assertions/test cases never change, only the prompt varies."""
    trial_uc = dataclasses.replace(uc, system_prompt=prompt)
    outcome = await sandbox_module.run_one(cheap_model, trial_uc, tc, provider=provider,
                                           cache=cache)
    if outcome["status"] != "ok":
        return {**outcome, "weighted": None, "reasons": {}}
    scored = await judge_module.score_one(cheap_model, judge_model, uc, tc,
                                          outcome["output"], cache=cache)
    return {**outcome, **scored}


async def _run_iteration(uc: UseCase, cheap_model: str, prompt: str, *,
                         provider: str, judge_model: str, cache=None) -> dict:
    per_test_case = []
    for tc in uc.test_cases:
        per_test_case.append(await _score_with_prompt(
            uc, tc, cheap_model, prompt, provider=provider,
            judge_model=judge_model, cache=cache))
    scored = [e["weighted"] for e in per_test_case if e.get("weighted") is not None]
    mean = round(sum(scored) / len(scored), 3) if scored else None
    return {"prompt": prompt, "meanScore": mean, "perTestCase": per_test_case}


async def execute(uc: UseCase, board: dict, bench_result: dict, *,
                  cheap_model: Optional[str] = None,
                  optimizer_model: Optional[str] = None,
                  judge_model: Optional[str] = None,
                  max_iterations: int = 3, target_gap: float = 0.20,
                  provider_by_model: Optional[dict] = None, cache=None) -> dict:
    """The bounded optimize -> test -> measure -> re-optimize loop.

    Stops the moment the cheap candidate's mean score comes within
    `target_gap` of the gold score, or once `max_iterations` is spent,
    whichever comes first — same "deepen only as far as needed" spirit as
    `bench.rejudge_for_spread`'s tie-zone-only re-scoring."""
    gold_model, gold_score, resolved_cheap = select_gold_and_cheap(
        board, cheap_model=cheap_model)
    gold_examples = _gold_examples(bench_result, gold_model, uc)
    if not gold_examples:
        raise NothingToOptimize(
            f"gold model {gold_model!r} has no successful answers to learn from.")

    optimizer_model = optimizer_model or gold_model
    judge_model = judge_model or bench_result.get("judge") or uc.judge_model
    provider = (provider_by_model or {}).get(resolved_cheap, "openrouter")

    current_prompt = uc.system_prompt
    prior_feedback = None
    iterations: list = []
    best_index: Optional[int] = None
    reached = False

    for i in range(max_iterations):
        proposed = await propose_prompt(
            uc, gold_examples, current_prompt, optimizer_model=optimizer_model,
            prior_feedback=prior_feedback, cache=cache)
        iteration = await _run_iteration(
            uc, resolved_cheap, proposed, provider=provider,
            judge_model=judge_model, cache=cache)
        iterations.append(iteration)
        current_prompt = proposed

        if best_index is None or (iteration["meanScore"] or -1) > (
                iterations[best_index]["meanScore"] or -1):
            best_index = i

        if iteration["meanScore"] is not None and iteration["meanScore"] >= gold_score - target_gap:
            reached = True
            break

        prior_feedback = [{"input": tc.input, "output": e.get("output"),
                           "reasons": e.get("reasons")}
                          for tc, e in zip(uc.test_cases, iteration["perTestCase"])
                          if e.get("status") == "ok"]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return {
        "schema": 1, "stamp": stamp, "ranAt": datetime.now(timezone.utc).isoformat(),
        "useCase": uc.name, "goldModel": gold_model, "goldScore": gold_score,
        "cheapModel": resolved_cheap, "optimizerModel": optimizer_model,
        "judgeModel": judge_model, "targetGap": target_gap,
        "maxIterations": max_iterations, "iterations": iterations,
        "bestIteration": (best_index + 1) if best_index is not None else None,
        "bestScore": iterations[best_index]["meanScore"] if best_index is not None else None,
        "reachedTarget": reached,
        # SAME FINGERPRINT FUNCTION AS EVERY OTHER SAVED RUN. The rubric,
        # test cases, checks and judge never change within an optimize run
        # — only the prompt does, and every iteration's own prompt is
        # already recorded on it — so this says what the SCORES here are
        # comparable to (a normal bench run of this same use case), not a
        # new kind of fingerprint.
        "measurement": config_module.measurement(uc),
    }


def save(result: dict, *, out_dir: Path) -> Path:
    """Atomic write (temp file + rename, same pattern as `state._save`) to
    a SIBLING directory of `out/`, not inside it —
    `out/optimize/<stamp>_<useCase>.json` — so this can never be picked up
    by `dashboard._runs_for`'s existing glob over `out/*_<use_case>.json`
    and never mistaken for a regular bench run."""
    d = Path(out_dir) / "optimize"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{result['stamp']}_{result['useCase']}.json"
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_latest(out_dir: Path, use_case: str) -> Optional[dict]:
    d = Path(out_dir) / "optimize"
    if not d.exists():
        return None
    suffix = f"_{use_case}.json"
    files = sorted((p for p in d.glob("*.json") if p.name.endswith(suffix)),
                   key=lambda p: p.name, reverse=True)
    return load(files[0]) if files else None
