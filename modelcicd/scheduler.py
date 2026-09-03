"""Re-runs the loop on its own, on whatever cadence each use case set.

WHAT "AUTOMATIC" MEANS HERE. `schedule.intervalDays` on a use case is the only
thing that makes it eligible — a use case with nothing set is manual-only,
exactly as today. Being due only ever produces a PENDING candidate through the
same `runner.execute` path `cli.py run` uses; nothing here calls
`state.approve()`. A model still only reaches production when a human clicks
Approve — the schedule changes how often the platform LOOKS, never who
decides.

TWO WAYS TO TRIGGER IT, BOTH SUPPORTED. `scheduler run-due` is a one-shot
check-and-exit, meant for an external cron / Task Scheduler entry. `scheduler
serve` is a long-running loop for someone who would rather leave one process
running than configure an external trigger. Either way it is the exact same
`run_due()` underneath.
"""
import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from . import config as config_module
from . import project as project_module
from . import runner as runner_module
from . import state as state_module


def due(state: dict, interval_days: Optional[int], *, now: Optional[datetime] = None) -> bool:
    """Whether a use case's next automatic run is due. Never having run counts
    as due; a use case with no interval configured is never due."""
    if not interval_days:
        return False
    now = now or datetime.now(timezone.utc)
    last = None
    history = state.get("history") or []
    if history:
        last = history[-1].get("ranAt")
    last = last or state.get("approvedAt")
    if not last:
        return True
    try:
        last_at = datetime.fromisoformat(last)
    except ValueError:
        return True
    if last_at.tzinfo is None:
        last_at = last_at.replace(tzinfo=timezone.utc)
    return now - last_at >= timedelta(days=interval_days)


def run_due(*, _cat=None) -> list:
    """Runs every use case, across every project, that is due right now.
    Returns one summary dict per use case actually run."""
    from . import catalogue as catalogue_module

    ran = []
    cat = _cat if _cat is not None else catalogue_module.load_or_poll()
    for proj in project_module.list_all():
        for uc_path in project_module.list_use_cases(proj.slug):
            uc = config_module.load(uc_path)
            if not uc.schedule_interval_days:
                continue
            state_root = project_module.state_dir(proj.slug)
            st = state_module.load(uc.name, state_root)
            if not due(st, uc.schedule_interval_days):
                continue
            candidates = runner_module.select_candidates(uc, cat, providers=proj.providers)
            if not candidates:
                ran.append({"project": proj.slug, "useCase": uc.name,
                           "skipped": "no candidates pass the guardrails"})
                continue
            result = asyncio.run(runner_module.execute(
                uc, cat, candidates, state_root=state_root,
                out_dir=project_module.out_dir(proj.slug)))
            ran.append({"project": proj.slug, "useCase": uc.name,
                       "pending": result["state"].get("pending")})
    return ran


def serve(interval_minutes: int = 60) -> None:
    """Loops forever, checking every use case's schedule every
    `interval_minutes`. Ctrl+C to stop."""
    print(f"[modelcicd] scheduler running — checking every {interval_minutes} "
          f"minute(s) for due use cases (Ctrl+C to stop).")
    while True:
        ran = run_due()
        if ran:
            for r in ran:
                print(f"[modelcicd] ran {r['project']}/{r['useCase']}: "
                     f"{r.get('pending') or r.get('skipped') or 'no pending candidate'}")
        time.sleep(max(60, interval_minutes * 60))
