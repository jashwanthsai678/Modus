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
from . import catalogue as catalogue_module
from . import guardrails as guardrails_module
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
    the catalogue has, same as an unscoped use case always has."""
    if models:
        return [m.strip() for m in models.split(",") if m.strip()]
    result = guardrails_module.apply(cat, uc.guardrails)
    passed = guardrails_module.within_tiers(
        result["passed"], [tier] if tier else uc.guardrails.tiers)
    if providers:
        wanted = set(providers)
        passed = [m for m in passed if set(m.get("providers") or []) & wanted]
    if limit:
        passed = passed[:limit]
    return [cid for m in passed if (cid := catalogue_module.cheapest_host_id(m))]


async def execute(uc: UseCase, cat: dict, candidates: list, *,
                  state_root: Optional[Path] = None,
                  out_dir: Optional[Path] = None) -> dict:
    """Runs the bench, ranks it, saves the run file, records state, and
    notifies if a candidate is pending. Returns everything a caller needs to
    report what happened, without needing to redo any of it."""
    result = await bench_module.run(candidates, uc)

    state = state_module.load(uc.name, state_root)
    board = rank_module.build(result, cat, approved_model=state.get("approvedModel"))
    text = rank_module.report(board)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = out_dir or (Path(__file__).resolve().parent.parent / "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stamp}_{uc.name}.json"
    out_path.write_text(
        json.dumps({"bench": result, "board": board}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    new_state = state_module.record_run(uc.name, board, root=state_root)
    notified = False
    if new_state.get("pending"):
        notified = notify_module.send_pending(uc.name, new_state, text, to=uc.notify.email)

    return {"stamp": stamp, "board": board, "state": new_state, "report": text,
            "out_path": out_path, "notified": notified}
