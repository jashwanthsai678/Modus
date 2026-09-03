"""Run every candidate through the sandbox, then judge every answer, blind.

CANDIDATES RUN CONCURRENTLY; JUDGING RUNS CONCURRENTLY WITH ONE PINNED MODEL.
There is no shared mutable state to corrupt here — each candidate call carries
its own model id as an argument, not a module global — so unlike a design that
patches a global per call, concurrency is simply safe.
"""
import asyncio
from datetime import datetime, timezone
from typing import Optional

from . import judge as judge_module
from . import sandbox as sandbox_module
from .config import UseCase

CANDIDATE_CONCURRENCY = 4
JUDGE_CONCURRENCY = 6


async def _judge_candidate(model_id: str, sandboxed: dict, judge_model: str,
                           uc: UseCase) -> dict:
    by_id = {tc.id: tc for tc in uc.test_cases}
    gate = asyncio.Semaphore(JUDGE_CONCURRENCY)

    async def one(entry: dict) -> dict:
        if entry["status"] != "ok":
            return {**entry}
        async with gate:
            return {**entry,
                    **await judge_module.score_one(
                        model_id, judge_model, uc, by_id[entry["testCase"]],
                        entry["output"])}

    judged = await asyncio.gather(*(one(e) for e in sandboxed["results"]))
    scored = [j for j in judged if j.get("status") == "ok"]
    mean = round(sum(j["weighted"] for j in scored) / len(scored), 3) if scored else None
    return {"model": model_id, "testCases": list(judged),
            "mean": mean, "wouldShipRate":
                round(sum(1 for j in scored if j.get("wouldShip")) / len(scored), 3)
                if scored else None,
            "counts": {"ok": len(scored),
                       "generation_failed": sum(1 for j in judged
                                                if j["status"] == "failed"),
                       "judge_failed": sum(1 for j in judged
                                           if j["status"] == "judge_failed")}}


async def run(candidates: list, uc: UseCase, *, judge_model: Optional[str] = None
             ) -> dict:
    """Every candidate: every test case answered, then every answer judged.

    IF THE USE CASE HAS A LIVE ENDPOINT, IT RIDES ALONG AS ONE MORE ROW. Same
    test-case inputs, same judge, same rubric — so the leaderboard shows what
    the use case's own application returns RIGHT NOW next to every candidate.
    It is a baseline capture, never a candidate: excluded from the
    judge-is-a-candidate clash check below, and never itself benchmarked."""
    judge_model = judge_model or uc.judge_model
    judge_module.check_judge_not_candidate(judge_model, candidates)

    gate = asyncio.Semaphore(CANDIDATE_CONCURRENCY)

    async def one_candidate(model_id: str) -> dict:
        async with gate:
            sandboxed = await sandbox_module.run_candidate(model_id, uc)
        return await _judge_candidate(model_id, sandboxed, judge_model, uc)

    async def one_endpoint() -> dict:
        sandboxed = await sandbox_module.run_endpoint_candidate(uc)
        return await _judge_candidate(sandbox_module.ENDPOINT_LABEL, sandboxed,
                                      judge_model, uc)

    tasks = [one_candidate(m) for m in candidates]
    if uc.endpoint:
        tasks.append(one_endpoint())

    results = await asyncio.gather(*tasks)
    return {
        "schema": 1, "ranAt": datetime.now(timezone.utc).isoformat(),
        "useCase": uc.name, "judge": judge_model,
        "testCases": len(uc.test_cases), "candidates": len(candidates),
        "results": list(results),
    }
