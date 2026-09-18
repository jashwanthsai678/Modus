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

from . import catalogue as catalogue_module
from . import client as client_module
from . import endpoint_client as endpoint_client_module
from .config import TestCase, UseCase

ENDPOINT_LABEL = "current (your endpoint)"


def _provider_spec(provider: str) -> dict:
    """Falls back to OpenRouter for an unrecognized provider name rather
    than raising — a stale/unknown provider tag on an old candidate id
    should degrade to the safe default, not break the whole bench."""
    return catalogue_module.PROVIDERS.get(provider) or catalogue_module.PROVIDERS["openrouter"]


def _generation_call(model_id: str, uc: UseCase, tc: TestCase, provider: str) -> dict:
    """Every argument that defines one generation call, in ONE place.

    THE COST PREVIEW AND THE ACTUAL CALL MUST AGREE. `run_one` below makes
    the call from this dict; `cached_count` fingerprints the same dict to
    tell a person how many of a run's calls are already paid for. Built in
    two places, they would drift the first time a parameter changed, and
    the symptom would be a run form promising "28 already cached" and then
    buying all 30 — a quiet lie about money, which is the one thing this
    project's cost previews exist to avoid."""
    spec = _provider_spec(provider)
    return {"prompt": tc.input, "model": model_id, "base_url": spec["chat_url"],
            "system": uc.system_prompt, "required": (),
            "temperature": 0.4, "max_tokens": uc.max_tokens}


def cached_count(model_ids: list, uc: UseCase, cache, *,
                 provider_by_model: Optional[dict] = None) -> int:
    """How many of this run's generation calls a cache could already serve.

    A LOCAL FILE CHECK, NOTHING SPENT. Reads the cache the same way the run
    will, so `run_form`'s cost line can say what a `--cache` run will
    actually buy rather than quoting the full price and being wrong."""
    if cache is None:
        return 0
    hits = 0
    for model_id in model_ids:
        provider = (provider_by_model or {}).get(model_id, "openrouter")
        for tc in uc.test_cases:
            call = _generation_call(model_id, uc, tc, provider)
            key = cache.fingerprint(
                model=call["model"], prompt=call["prompt"], system=call["system"],
                temperature=call["temperature"], max_tokens=call["max_tokens"],
                base_url=call["base_url"], required=call["required"])
            if cache.get(key) is not None:
                hits += 1
    # A preview must not be mistaken for the run itself in the run file's
    # own cache accounting, so the counters it moved are rolled back.
    cache.hits = cache.misses = 0
    return hits


async def run_one(model_id: str, uc: UseCase, tc: TestCase, *,
                  provider: str = "openrouter", cache=None) -> dict:
    """One candidate, one test case, one call — through WHICHEVER platform
    this candidate actually came from, not always OpenRouter.

    `cache`, when given, reuses this exact (model, prompt, system, input)
    answer if it was already bought — the expensive half of a run, and the
    half a rubric edit doesn't change. `None` (the default) always calls.
    NOT PASSED by `bench.resample_candidates_for_spread`, which exists to
    measure how much this call's answer MOVES between identical calls; a
    cache would hand it the same answer every time and it would report a
    spread of exactly zero. See `cache.py`."""
    spec = _provider_spec(provider)
    call = _generation_call(model_id, uc, tc, provider)
    started = datetime.now(timezone.utc)
    usage: dict = {}
    try:
        output = await client_module.call_json(
            call.pop("prompt"), label=f"sandbox[{model_id}:{tc.id}]",
            api_key_env=spec["key_env"], cache=cache, usage_sink=usage, **call)
    except client_module.RateLimitedError as exc:
        # A REAL, OBSERVED signal — this candidate hit a rate limit during
        # even a light bench. Recorded distinctly from a generic failure so
        # the leaderboard can say so, next to the use case's own usage
        # estimate if one was given.
        return {"testCase": tc.id, "status": "rate_limited",
                "seconds": (datetime.now(timezone.utc) - started).total_seconds(),
                "error": str(exc)[:300]}
    except Exception as exc:                        # noqa: BLE001
        return {"testCase": tc.id, "status": "failed",
                "seconds": (datetime.now(timezone.utc) - started).total_seconds(),
                "error": str(exc)[:300]}
    return {"testCase": tc.id, "status": "ok",
            "seconds": (datetime.now(timezone.utc) - started).total_seconds(),
            "usage": usage or None, "output": output}


async def run_candidate(model_id: str, uc: UseCase, *,
                        provider: str = "openrouter", cache=None) -> dict:
    """Every test case for one candidate, in sequence.

    Sequential, not concurrent, per candidate — this is a screening tool run
    occasionally on a schedule, not a latency-sensitive service, and serial
    calls are one fewer way to trip a shared rate limit across many candidates
    running at once. Candidates themselves ARE run concurrently, one level up in
    `bench.py`.
    """
    results = []
    for tc in uc.test_cases:
        results.append(await run_one(model_id, uc, tc, provider=provider, cache=cache))
    counts = {"ok": 0, "failed": 0, "rateLimited": 0}
    for r in results:
        counts["ok" if r["status"] == "ok" else
               "rateLimited" if r["status"] == "rate_limited" else "failed"] += 1
    return {"model": model_id, "results": results, "counts": counts}


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
