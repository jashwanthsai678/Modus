"""One tiny call per model, before the run, to catch what is already dead.

WHAT THIS PREVENTS, TWICE OBSERVED IN THIS PROJECT. A model id that a
platform has deprecated keeps looking fine everywhere it's read from — it
sits in the catalogue, passes every guardrail, gets picked as a candidate —
and only fails when a real call is made. Both times it happened here the
symptom was a generic "could not be scored" on a leaderboard row, minutes
into a run, after the concurrency slot and the wall-clock were already spent.
A dead judge is worse still: every generation call gets paid for and then
every one of them comes back `judge_failed`, so the whole run is wasted.

WHAT THIS DOES NOT CLAIM, AND THE REASON IT'S WORDED SO NARROWLY. This
project already learned that an easy request succeeding does not prove a
harder one will — a model that answered a trivial probe went on to fail the
real scan prompt outright, because it narrated its way past the token limit
before reaching any JSON. So a passing probe here means exactly one thing:
"this id exists, the key works, and the platform answered." It is NOT a
prediction that the model will complete the actual test cases. A health check
that implied otherwise would be worse than none, because someone would trust
it.

ONLY A PROVABLE DEATH EXCLUDES A CANDIDATE, AND THE HTTP STATUS DOES NOT
TELL YOU THAT. This was checked live against OpenRouter, and the first
version of this module — which excluded on 404 — was wrong in a way that
would have blocked every run:

    404  "This model is unavailable for free. The paid version is available
          now - use this slug instead: minimax/minimax-m3"
                                        GONE. Permanent. Message names the fix.
    404  "Provider returned error"   metadata: {provider_name: "Nvidia"}
                                        NOT GONE. The upstream host had a
                                        moment. This one was the configured
                                        JUDGE, so excluding it would have
                                        refused the whole run.
    400  "vendor/x is not a valid model ID"
                                        GONE. Never existed. And it's a 400,
                                        so a 404-only rule misses the single
                                        most clear-cut case.

Two 404s meaning opposite things is not an edge case, it's the normal
situation, so the decision has to read the platform's message. That is
string matching against another service's prose, which is fragile — so it is
deliberately one-directional: a message must match a known-permanent pattern
to be FATAL, and everything unrecognized falls through to "unknown", which
never excludes anything. A new phrasing therefore costs a wasted run, not a
live model wrongly dropped.

Everything else (a timeout, a 429, an unparseable reply, a network blip) is
REPORTED but never excluded, because "unknown is never resolved in the
candidate's favor" cuts both ways: silently shrinking the field on a guess is
the same class of bug as silently assuming an unknown price is free. A rate
limit in particular would exclude half the free tier at a busy moment, which
is a fact about the minute, not the model.

NOTHING IS CACHED HERE, DELIBERATELY. The entire point is to know what's dead
RIGHT NOW. A cached "alive" is exactly the stale-and-flattering assumption
this module exists to catch, and a cached "dead" would keep excluding a model
that came back. The probes are a few tokens each; freshness is worth more.

WHATEVER IS EXCLUDED IS SHOWN. A candidate dropped before the bench never
appears on the leaderboard at all, which is the same silent-disappearance
this project has already had to fix once elsewhere. `runner.execute` records
these results into the saved run file, and the run page lists every exclusion
with its reason.
"""
import asyncio
from datetime import datetime, timezone
from typing import Optional

from . import catalogue as catalogue_module
from . import client as client_module

PROBE_CONCURRENCY = 6

# GENEROUS ON PURPOSE, at the cost of a few tokens. The failure this project
# actually hit was a model narrating its reasoning before emitting JSON and
# running out of room — at a stingy limit that reads as "broken model" when
# it's really "budget too small". 200 tokens makes a false accusation
# unlikely; the probe is still trivially cheap.
PROBE_MAX_TOKENS = 200

# ONE SHOT, NO RETRIES. `call_json`'s retry exists to rescue a real answer
# worth paying for. Here it would triple the cost of a check whose whole
# value is being cheap, and a first-shot failure lands in "unknown" — which
# never excludes anything — so nothing is lost by not retrying.
PROBE_ATTEMPTS = 1

# Exercises json_object mode, because every real call in this project uses
# it. A model that can't do JSON at all fails here rather than on test case
# one of ten.
_PROBE_PROMPT = 'Reply with exactly this JSON object and nothing else: {"ok": true}'

# The one status that justifies dropping a candidate before the run.
FATAL = "unavailable"

# Phrases that mean a model id is PERMANENTLY unusable, matched
# case-insensitively against the platform's own error message. Every one of
# these was observed live; none is inferred from documentation.
#
# ADDING TO THIS LIST IS THE ONLY WAY A CANDIDATE BECOMES EXCLUDABLE, which
# is the safe direction. A phrasing this list doesn't know lands in
# "unknown", the candidate still runs, and the cost is one wasted run — not
# a working model silently dropped from someone's leaderboard.
_GONE_PHRASES = (
    "is not a valid model",       # 400: the id does not exist at all
    "unavailable for free",       # the free slug was retired; message names the paid one
    "use this slug instead",      # explicitly renamed
    "no longer available",
    "has been deprecated",
    "is deprecated",
)

# Observed 404s that are NOT deaths. Listed only as documentation of the trap
# — the code needs no exclusion list, because anything not matching
# `_GONE_PHRASES` already falls through to "unknown".
#
#   "Provider returned error"  — the upstream host erred; the model is fine.
#   "No endpoints found"       — no provider is serving it AT THE MOMENT.
#                                Could be capacity, could be permanent; not
#                                distinguishable from here, so not fatal.


def classify(message: str) -> str:
    """FATAL only for a message that states the id is permanently unusable;
    "unknown" for everything else, including a bare "Provider returned
    error". One-directional on purpose — see `_GONE_PHRASES`."""
    lowered = (message or "").lower()
    return FATAL if any(p in lowered for p in _GONE_PHRASES) else "unknown"


async def probe(model_id: str, *, provider: str = "openrouter",
                timeout_s: float = 20.0) -> dict:
    """One tiny call. Returns {model, provider, status, detail, seconds}.

    `status` is one of:
        ok            answered, and the answer parsed
        unavailable   the platform's message says this id is permanently
                      unusable — see `classify`. The ONLY fatal status.
        rate_limited  429 — transient, reported, never excluded
        unknown       anything else, INCLUDING a 404 whose message is just
                      "Provider returned error" — reported, never excluded
    """
    spec = catalogue_module.PROVIDERS.get(provider) or catalogue_module.PROVIDERS["openrouter"]
    started = datetime.now(timezone.utc)

    def done(status: str, detail: str) -> dict:
        return {"model": model_id, "provider": provider, "status": status,
                "detail": detail,
                "seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 2)}

    try:
        await client_module.call_json(
            _PROBE_PROMPT, model=model_id, label=f"health[{model_id}]",
            base_url=spec["chat_url"], api_key_env=spec["key_env"],
            required=("ok",), temperature=0.0, max_tokens=PROBE_MAX_TOKENS,
            attempts=PROBE_ATTEMPTS, timeout_s=timeout_s)
    except client_module.ModelUnavailableError as exc:
        # The platform rejected it at the model level. Whether that's
        # permanent is decided from its own message, never from the status
        # code — the two 404s in this module's docstring mean opposite
        # things. The message is kept verbatim because it is often the fix
        # ("use this slug instead: minimax/minimax-m3").
        return done(classify(str(exc)), str(exc)[:200])
    except client_module.RateLimitedError as exc:
        return done("rate_limited", str(exc)[:200])
    except Exception as exc:                        # noqa: BLE001
        # Includes a missing API key, a timeout, and a model that answered
        # with something unparseable. All genuinely ambiguous, all reported
        # rather than acted on.
        return done("unknown", str(exc)[:200])
    return done("ok", "answered a trivial JSON request")


async def probe_all(model_ids: list, *, provider_by_model: Optional[dict] = None,
                    concurrency: int = PROBE_CONCURRENCY,
                    timeout_s: float = 20.0) -> list:
    """Every candidate, concurrently but bounded — same discipline as the
    bench itself, so a health check can't be the thing that trips a shared
    rate limit."""
    gate = asyncio.Semaphore(concurrency)

    async def one(model_id: str) -> dict:
        async with gate:
            return await probe(model_id,
                               provider=(provider_by_model or {}).get(model_id, "openrouter"),
                               timeout_s=timeout_s)

    return list(await asyncio.gather(*(one(m) for m in model_ids)))


def partition(results: list) -> tuple:
    """(usable_model_ids, fatal_results) — the split `runner.execute` acts on.

    Only `FATAL` results are dropped. A `rate_limited` or `unknown` candidate
    stays in the field and takes its chances in the bench, where a failure
    becomes a visible leaderboard row with its own error rather than a
    silent absence."""
    usable = [r["model"] for r in results if r["status"] != FATAL]
    fatal = [r for r in results if r["status"] == FATAL]
    return usable, fatal


def summary(results: list) -> str:
    """One line for the CLI, naming counts by status — never just 'N
    unhealthy', because 'one is gone, two were rate limited' and 'three are
    gone' call for completely different reactions."""
    if not results:
        return "no candidates probed"
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    parts = [f"{counts[s]} {s}" for s in ("ok", FATAL, "rate_limited", "unknown")
             if counts.get(s)]
    return f"probed {len(results)}: " + ", ".join(parts)
