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
                           uc: UseCase, *, cache=None) -> dict:
    by_id = {tc.id: tc for tc in uc.test_cases}
    gate = asyncio.Semaphore(JUDGE_CONCURRENCY)

    async def one(entry: dict) -> dict:
        if entry["status"] != "ok":
            return {**entry}
        async with gate:
            return {**entry,
                    **await judge_module.score_one(
                        model_id, judge_model, uc, by_id[entry["testCase"]],
                        entry["output"], cache=cache)}

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
                                           if j["status"] == "judge_failed"),
                       "rate_limited": sum(1 for j in judged
                                           if j["status"] == "rate_limited")}}


async def run(candidates: list, uc: UseCase, *, judge_model: Optional[str] = None,
             provider_by_model: Optional[dict] = None, cache=None) -> dict:
    """Every candidate: every test case answered, then every answer judged.

    EACH CANDIDATE IS CALLED THROUGH WHICHEVER PLATFORM IT ACTUALLY CAME
    FROM. `provider_by_model` (built by `catalogue.provider_map`) maps a
    candidate id to "openrouter"/"groq"/"fireworks"; a candidate missing
    from it — e.g. one named directly via `--models` — falls back to
    OpenRouter, today's only behavior. The JUDGE always calls through
    OpenRouter regardless, unchanged from before this existed.

    IF THE USE CASE HAS A LIVE ENDPOINT, IT RIDES ALONG AS ONE MORE ROW. Same
    test-case inputs, same judge, same rubric — so the leaderboard shows what
    the use case's own application returns RIGHT NOW next to every candidate.
    It is a baseline capture, never a candidate: excluded from the
    judge-is-a-candidate clash check below, and never itself benchmarked.

    `cache`, WHEN GIVEN, IS RECORDED IN THE RESULT — not just used. A run
    whose answers were replayed from a previous run's cache is a different
    kind of evidence from one where every answer was bought fresh, and the
    trend chart compares them side by side. `cacheStats` in the returned
    dict says how many of this run's calls were replays, so that
    distinction survives into the saved run file instead of living only in
    whoever remembered ticking the box. The LIVE endpoint baseline is
    deliberately never cached: its whole purpose is to capture what the
    application returns RIGHT NOW."""
    judge_model = judge_model or uc.judge_model
    judge_module.check_judge_not_candidate(judge_model, candidates)

    gate = asyncio.Semaphore(CANDIDATE_CONCURRENCY)

    async def one_candidate(model_id: str) -> dict:
        provider = (provider_by_model or {}).get(model_id, "openrouter")
        async with gate:
            sandboxed = await sandbox_module.run_candidate(
                model_id, uc, provider=provider, cache=cache)
        return await _judge_candidate(model_id, sandboxed, judge_model, uc, cache=cache)

    async def one_endpoint() -> dict:
        sandboxed = await sandbox_module.run_endpoint_candidate(uc)
        return await _judge_candidate(sandbox_module.ENDPOINT_LABEL, sandboxed,
                                      judge_model, uc, cache=cache)

    tasks = [one_candidate(m) for m in candidates]
    if uc.endpoint:
        tasks.append(one_endpoint())

    results = await asyncio.gather(*tasks)
    return {
        "schema": 1, "ranAt": datetime.now(timezone.utc).isoformat(),
        "useCase": uc.name, "judge": judge_model,
        "testCases": len(uc.test_cases), "candidates": len(candidates),
        "cacheStats": cache.stats() if cache is not None else None,
        "results": list(results),
    }


async def rejudge_for_spread(bench_result: dict, uc: UseCase, judge_model: str,
                             model_ids: list, *, repeats: int = 3) -> dict:
    """Re-scores the ALREADY-GENERATED outputs of just the given candidates
    (the tie-zone ones a caller identified from `rank.build`'s shortlists)
    several times each, to surface judge-side noise the single-sample
    pipeline can't see on its own. No candidate is re-run — every output
    here already exists in `bench_result`; this only spends extra judge
    calls, and only on the shortlist, never the whole field.

    Returns {model_id: mean_spread}, averaged across that model's test
    cases — a model with no scoreable test cases in the given result is
    simply absent from the returned dict, not an error."""
    by_id = {tc.id: tc for tc in uc.test_cases}
    by_model = {r["model"]: r for r in bench_result.get("results") or []}
    gate = asyncio.Semaphore(JUDGE_CONCURRENCY)

    async def one_test_case(model_id: str, tc_result: dict) -> Optional[float]:
        tc = by_id.get(tc_result.get("testCase"))
        if tc_result.get("status") != "ok" or not tc:
            return None
        async with gate:
            r = await judge_module.score_repeated(
                model_id, judge_model, uc, tc, tc_result["output"], repeats=repeats)
        return r["spread"]

    async def one_model(model_id: str) -> tuple:
        entry = by_model.get(model_id)
        if not entry:
            return model_id, None
        spreads = await asyncio.gather(
            *(one_test_case(model_id, tc) for tc in entry.get("testCases") or []))
        real = [s for s in spreads if s is not None]
        return model_id, (round(sum(real) / len(real), 3) if real else None)

    pairs = await asyncio.gather(*(one_model(m) for m in model_ids))
    return {model_id: spread for model_id, spread in pairs if spread is not None}


async def resample_candidates_for_spread(uc: UseCase, model_ids: list, judge_model: str,
                                         *, provider_by_model: Optional[dict] = None,
                                         repeats: int = 2) -> dict:
    """RE-GENERATES each given candidate's answer to every test case, several
    more times each, judging every new answer — a different source of noise
    than `rejudge_for_spread`, which only re-scores an answer that already
    exists. This is the one the single-sample pipeline genuinely can't see
    on its own: the same model asked the same question twice can give a
    meaningfully different answer, and a leaderboard built from one sample
    can't tell a real edge from that kind of luck.

    THIS SPENDS REAL EXTRA GENERATION CALLS, NOT JUST JUDGE CALLS — unlike
    `rejudge_for_spread`, it is never wired in automatically. A caller opts
    in explicitly, and even then only for a tie-zone shortlist a caller
    already identified, never the whole field.

    TAKES NO `cache` PARAMETER, ON PURPOSE — and must never grow one, for
    the same reason `judge.score_repeated` doesn't. This function's entire
    measurement is how far apart two IDENTICAL calls land. Served from a
    cache, every repeat would return the first answer, `max - min` would be
    exactly 0.0, and the leaderboard would print "candidate spread: 0.00"
    — reading as a rock-steady model — about a re-generation that never
    happened. The calls below therefore pass no cache, and that is load-
    bearing, not an omission.

    Returns {model_id: mean_spread}, same shape as `rejudge_for_spread`."""
    gate = asyncio.Semaphore(CANDIDATE_CONCURRENCY)

    async def one_sample(model_id: str, tc, provider: str) -> Optional[float]:
        async with gate:
            outcome = await sandbox_module.run_one(model_id, uc, tc, provider=provider)
        if outcome.get("status") != "ok":
            return None
        scored = await judge_module.score_one(model_id, judge_model, uc, tc, outcome["output"])
        return scored["weighted"] if scored.get("status") == "ok" else None

    async def one_test_case(model_id: str, tc, provider: str) -> Optional[float]:
        samples = await asyncio.gather(
            *(one_sample(model_id, tc, provider) for _ in range(repeats)))
        real = [s for s in samples if s is not None]
        return (max(real) - min(real)) if len(real) >= 2 else None

    async def one_model(model_id: str) -> tuple:
        provider = (provider_by_model or {}).get(model_id, "openrouter")
        spreads = await asyncio.gather(
            *(one_test_case(model_id, tc, provider) for tc in uc.test_cases))
        real = [s for s in spreads if s is not None]
        return model_id, (round(sum(real) / len(real), 3) if real else None)

    pairs = await asyncio.gather(*(one_model(m) for m in model_ids))
    return {model_id: spread for model_id, spread in pairs if spread is not None}
