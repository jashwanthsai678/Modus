"""Run every test case against one candidate model.

ONE CALL PER TEST CASE, NOT A REPEATED SUITE. A screening pass over forty
candidates and ten test cases costs 400 calls this way; running each three times
for statistical confidence costs 1,200. The trade is stated plainly in
`judge.py`'s report: this ranks confidently at the extremes and only suggests in
the middle. Deepen on the shortlist a candidate makes it to, never on the whole
field.

A FAILED CANDIDATE IS A RESULT, NOT AN EXCEPTION. A model that cannot produce
usable output on a test case has told you something true about it. Every
candidate that fails is recorded with its error and appears on the leaderboard,
rather than silently vanishing from it.
"""
import asyncio
from datetime import datetime, timezone
from typing import Optional

from . import client as client_module
from . import endpoint_client as endpoint_client_module
from .config import TestCase, UseCase

ENDPOINT_LABEL = "current (your endpoint)"


async def run_one(model_id: str, uc: UseCase, tc: TestCase) -> dict:
    """One candidate, one test case, one call."""
    started = datetime.now(timezone.utc)
    try:
        output = await client_module.call_json(
            tc.input, model=model_id, label=f"sandbox[{model_id}:{tc.id}]",
            system=uc.system_prompt, required=(),
            temperature=0.4, max_tokens=uc.max_tokens)
    except Exception as exc:                        # noqa: BLE001
        return {"testCase": tc.id, "status": "failed",
                "seconds": (datetime.now(timezone.utc) - started).total_seconds(),
                "error": str(exc)[:300]}
    return {"testCase": tc.id, "status": "ok",
            "seconds": (datetime.now(timezone.utc) - started).total_seconds(),
            "output": output}


async def run_candidate(model_id: str, uc: UseCase) -> dict:
    """Every test case for one candidate, in sequence.

    Sequential, not concurrent, per candidate — this is a screening tool run
    occasionally on a schedule, not a latency-sensitive service, and serial
    calls are one fewer way to trip a shared rate limit across many candidates
    running at once. Candidates themselves ARE run concurrently, one level up in
    `bench.py`.
    """
    results = []
    for tc in uc.test_cases:
        results.append(await run_one(model_id, uc, tc))
    ok = [r for r in results if r["status"] == "ok"]
    return {"model": model_id, "results": results,
            "counts": {"ok": len(ok), "failed": len(results) - len(ok)}}


async def run_endpoint_one(uc: UseCase, tc: TestCase) -> dict:
    """One test case against the use case's OWN live endpoint — the baseline,
    not a candidate. Same result shape as `run_one` so nothing downstream
    (judging, ranking) needs to know the difference."""
    started = datetime.now(timezone.utc)
    try:
        output = await endpoint_client_module.call(
            uc.endpoint, tc.input, label=f"endpoint[{tc.id}]",
            timeout_s=uc.endpoint.timeout_seconds)
    except Exception as exc:                        # noqa: BLE001
        return {"testCase": tc.id, "status": "failed",
                "seconds": (datetime.now(timezone.utc) - started).total_seconds(),
                "error": str(exc)[:300]}
    return {"testCase": tc.id, "status": "ok",
            "seconds": (datetime.now(timezone.utc) - started).total_seconds(),
            "output": output}


async def run_endpoint_candidate(uc: UseCase) -> dict:
    """Every test case against the use case's own endpoint, sequentially —
    mirrors `run_candidate`, so `bench.py` can treat this exactly like one
    more candidate's results."""
    results = []
    for tc in uc.test_cases:
        results.append(await run_endpoint_one(uc, tc))
    ok = [r for r in results if r["status"] == "ok"]
    return {"model": ENDPOINT_LABEL, "results": results,
            "counts": {"ok": len(ok), "failed": len(results) - len(ok)}}
