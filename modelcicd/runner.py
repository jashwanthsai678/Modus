"""The reusable core of one bench run — candidate selection through to the
saved run file, the state update, and the notification.

ONE IMPLEMENTATION, THREE CALLERS. `cli.py run` (interactive, asks before it
spends), `scheduler.py` (automatic, spends without asking — same as `--yes`),
and any future caller all go through `execute()` so there is exactly one place
that turns a candidate list into a leaderboard, a saved file, and a state
update. `cli.py` keeps the printing and the interactive confirmation; nothing
about what actually happens once someone says yes lives in two places.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import bench as bench_module
from . import cache as cache_module
from . import catalogue as catalogue_module
from . import config as config_module
from . import guardrails as guardrails_module
from . import health as health_module
from . import notify as notify_module
from . import rank as rank_module
from . import state as state_module
from .config import UseCase


def select_candidates(uc: UseCase, cat: dict, *, providers: Optional[list] = None,
                      tier: Optional[str] = None, models: Optional[str] = None,
                      limit: Optional[int] = None) -> list:
    """Which candidate model ids a run should bench — an explicit list if
    given, otherwise the use case's own guardrails applied to the catalogue.

    `providers` narrows the field to a project's chosen marketplace(s) (e.g.
    a project connected with `--providers groq`) — `None` searches everything
    the catalogue has, same as an unscoped use case always has.

    THE JUDGE IS EXCLUDED HERE, NOT JUST REFUSED LATER. A guardrails-derived
    field naturally CAN contain the judge model (this got more likely once
    the default judge became a free model, sitting in the same free tier
    candidates are drawn from) — dropping it here, before `limit` is applied,
    means benching still runs instead of refusing outright over something
    this function can just avoid by construction. An EXPLICIT `--models`
    list is left alone: naming the judge on purpose is still refused loudly
    by `judge.check_judge_not_candidate`, not silently corrected."""
    if models:
        return [m.strip() for m in models.split(",") if m.strip()]
    result = guardrails_module.apply(cat, uc.guardrails)
    passed = guardrails_module.within_tiers(
        result["passed"], [tier] if tier else uc.guardrails.tiers)
    if providers:
        wanted = set(providers)
        passed = [m for m in passed if set(m.get("providers") or []) & wanted]
    judge_key = catalogue_module.key(uc.judge_model)
    passed = [m for m in passed if m.get("key") != judge_key]
    if limit:
        passed = passed[:limit]
    return [cid for m in passed if (cid := catalogue_module.cheapest_host_id(m))]


async def execute(uc: UseCase, cat: dict, candidates: list, *,
                  state_root: Optional[Path] = None,
                  out_dir: Optional[Path] = None,
                  resample_candidates: bool = False,
                  use_cache: bool = False,
                  cache_root: Optional[Path] = None,
                  health_check: bool = True) -> dict:
    """Runs the bench, ranks it, deepens judging on just the tie zone,
    saves the run file, records state, and notifies if a candidate is
    pending AND hasn't already been notified about. Returns everything a
    caller needs to report what happened, without needing to redo any of
    it.

    `resample_candidates` OPTS IN TO RE-GENERATING (not just re-scoring)
    the tie-zone shortlist a few more times each, to surface CANDIDATE-side
    noise on top of the judge-side noise `rejudge_for_spread` already
    always checks. Off by default — unlike judge-spread, this spends real
    extra generation calls, so it's never turned on silently.

    `use_cache` REUSES ANSWERS ALREADY BOUGHT, and is off by default for
    the mirror-image reason. A cached run is cheap but it is a REPLAY: the
    same rubric edit iterated ten times costs one set of generation calls
    instead of ten, which is the point, but a run whose answers came from
    last week is not a fresh measurement of what those models do today.
    Off by default means the number on the leaderboard is a measurement
    unless someone deliberately asked for a replay — and `cacheStats` in
    the saved run file records which it was either way.

    The cache object is created HERE, per run, and passed down — never a
    module-level switch. The dashboard runs benches in worker threads, and
    two concurrent runs (one replaying, one measuring) must not be able to
    reach into each other's setting.

    `health_check` PROBES EVERY CANDIDATE AND THE JUDGE FIRST, one tiny call
    each, and drops only the ones the platform says are GONE (404). On by
    default because the failure it prevents — minutes of a run spent
    discovering a deprecated id, or worse, every generation call paid for and
    then thrown away by a dead judge — costs far more than the probes. It
    never claims a passing model will succeed on the real test cases; see
    `health.py`. Whatever it excludes is returned and saved, never silently
    dropped."""
    cache = cache_module.Cache(cache_root) if use_cache else None
    provider_by_model = catalogue_module.provider_map(cat)

    health = []
    excluded = []
    if health_check and candidates:
        # THE JUDGE IS PROBED TOO, and it's the one that can waste an entire
        # run: if the judge is gone, every candidate answer gets bought and
        # then scored as `judge_failed`. Refused up front instead, loudly,
        # the same way `check_judge_not_candidate` refuses rather than warns.
        judge_model = uc.judge_model
        health = await health_module.probe_all(
            candidates + [judge_model], provider_by_model=provider_by_model)
        judge_health = next((h for h in health if h["model"] == judge_model), None)
        if judge_health and judge_health["status"] == health_module.FATAL:
            raise RuntimeError(
                f"the judge ({judge_model}) is gone: {judge_health['detail']}. "
                f"Nothing was spent. Every candidate would have been generated "
                f"and then failed to score, so this refuses instead of running. "
                f"Point `judgeModel` at a live model and try again.")

        candidate_health = [h for h in health if h["model"] != judge_model]
        usable, fatal = health_module.partition(candidate_health)
        excluded = fatal
        candidates = [m for m in candidates if m in set(usable)]
        if not candidates:
            raise RuntimeError(
                f"every candidate is gone from its platform "
                f"({', '.join(h['model'] for h in fatal)}). Nothing was spent. "
                f"Re-poll the catalogue (`catalogue --refresh`) — these ids were "
                f"probably deprecated since it was last fetched.")

    result = await bench_module.run(
        candidates, uc, provider_by_model=provider_by_model, cache=cache)

    state = state_module.load(uc.name, state_root)
    board = rank_module.build(result, cat, approved_model=state.get("approvedModel"),
                              estimated_calls_per_day=uc.estimated_calls_per_day)

    # A close call is exactly where judge noise matters most — re-score just
    # the candidates that made a tie-band shortlist, against their SAME
    # already-generated outputs (no extra candidate calls), and show the
    # spread. Everyone outside the tie zone was never in question.
    tie_zone = sorted({m for band in (board.get("shortlists") or {}).values() for m in band})
    if tie_zone:
        spread_map = await bench_module.rejudge_for_spread(
            result, uc, result.get("judge") or uc.judge_model, tie_zone)
        rank_module.attach_judge_spread(board, spread_map)
        if resample_candidates:
            candidate_spread_map = await bench_module.resample_candidates_for_spread(
                uc, tie_zone, result.get("judge") or uc.judge_model,
                provider_by_model=catalogue_module.provider_map(cat))
            rank_module.attach_candidate_spread(board, candidate_spread_map)

    text = rank_module.report(board)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = out_dir or (Path(__file__).resolve().parent.parent / "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stamp}_{uc.name}.json"
    # `health` IS SAVED WITH THE RUN, not just returned. A candidate dropped
    # before the bench never appears on the leaderboard, and a candidate that
    # vanishes with no explanation is the exact failure this project has
    # already had to fix once in bulk-create. The run page reads this back
    # and lists every exclusion with its reason.
    out_path.write_text(
        json.dumps({"bench": result, "board": board, "health": health},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")

    # WHAT CHANGED SINCE THE LAST RUN, read BEFORE this run is appended to
    # history — afterwards the newest entry is this run itself and there is
    # nothing left to compare against.
    prior = state_module.load(uc.name, state_root).get("history") or []
    measurement_changes = config_module.measurement_changes(
        board.get("measurement"), (prior[-1] if prior else {}).get("measurement"))

    new_state = state_module.record_run(uc.name, board, root=state_root)
    notified = False
    if new_state.get("pending"):
        pending_model = new_state["pending"]["model"]
        # Only notify when the pending candidate actually CHANGED since the
        # last email — otherwise a scheduled run re-finding the same
        # unreviewed candidate would re-send the same email every time it
        # checks, forever, until someone acts on it.
        if pending_model != new_state.get("notifiedModel"):
            notified = notify_module.send_pending(uc.name, new_state, text, to=uc.notify.email)
            if notified:
                state_module.mark_notified(uc.name, pending_model, root=state_root)

    return {"stamp": stamp, "board": board, "state": new_state, "report": text,
            "out_path": out_path, "notified": notified,
            "cache_stats": result.get("cacheStats"),
            "measurement_changes": measurement_changes,
            "health": health, "excluded": excluded}
