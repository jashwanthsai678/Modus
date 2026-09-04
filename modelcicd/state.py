"""What is currently approved for each use case, and the history that got it there.

ONE JSON FILE PER USE CASE, NOT A DATABASE. This is a self-hosted, single-tenant
tool — the state that matters is "what model is live right now" and "what did we
try before", and a flat file both a person and `resolver.py` can read without a
server answers that completely. A database earns its cost when there are many
tenants to isolate; there is one, here.

APPROVAL IS THE ONLY THING THAT WRITES `approvedModel`. Every other function in
this module is additive — a new run's board is appended to history, never
replacing what came before — so that "what did we approve, and when, and why"
stays answerable after twenty cycles.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "state"


def _path(use_case: str, root: Optional[Path] = None) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in use_case)
    return Path(root or DEFAULT_DIR) / f"{safe}.json"


def load(use_case: str, root: Optional[Path] = None) -> dict:
    p = _path(use_case, root)
    if not p.exists():
        return {"useCase": use_case, "approvedModel": None,
                "approvedScore": None, "approvedAt": None,
                "history": [], "pending": None,
                "notifiedModel": None, "rejected": []}
    return json.loads(p.read_text(encoding="utf-8"))


def _save(state: dict, root: Optional[Path] = None) -> Path:
    p = _path(state["useCase"], root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def record_run(use_case: str, board: dict, *, root: Optional[Path] = None) -> dict:
    """Append one bench run's leaderboard to history, and flag a candidate for
    approval if it beats what is currently live.

    THE PENDING CANDIDATE IS A PROPOSAL, NOT A PROMOTION. Recording it here does
    not change what `resolver.py` returns — only `approve()` does that. A run
    that finds a better model changes nothing about production until a human
    says so.
    """
    from . import rank as rank_module

    state = load(use_case, root)
    state["history"] = (state.get("history") or [])[-49:] + [{
        "ranAt": board.get("ranAt"),
        "tiers": {name: [{"model": r["model"], "score": r["score"]}
                         for r in band]
                 for name, band in board.get("tiers", {}).items()},
    }]

    best = rank_module.best_overall(board)
    approved_score = state.get("approvedScore")
    state["pending"] = None
    if best and best["model"] != state.get("approvedModel"):
        if approved_score is None or best["score"] - approved_score >= state.get(
                "minImprovement", 0.20):
            state["pending"] = {"model": best["model"], "score": best["score"],
                                "foundAt": board.get("ranAt")}
    _save(state, root)
    return state


def approve(use_case: str, model_id: Optional[str] = None, *,
           root: Optional[Path] = None) -> dict:
    """Promote a model to approved. Defaults to whatever is pending.

    THE ONLY FUNCTION THAT CHANGES WHAT resolve() RETURNS. Everything upstream
    of this — the bench, the ranking, the email — only ever proposes.
    """
    state = load(use_case, root)
    target = model_id or (state.get("pending") or {}).get("model")
    if not target:
        raise ValueError(f"nothing pending for {use_case!r} and no model_id "
                         f"given — run a bench first, or name a model explicitly.")
    score = (state.get("pending") or {}).get("score") if target == (
        state.get("pending") or {}).get("model") else None
    state["approvedModel"] = target
    state["approvedScore"] = score
    state["approvedAt"] = datetime.now(timezone.utc).isoformat()
    state["pending"] = None
    return _save(state, root) and load(use_case, root)


def set_min_improvement(use_case: str, value: float, *, root: Optional[Path] = None) -> None:
    state = load(use_case, root)
    state["minImprovement"] = value
    _save(state, root)


def mark_notified(use_case: str, model: str, *, root: Optional[Path] = None) -> None:
    """Records that this candidate has already been emailed about, so a
    scheduled run that finds the SAME pending candidate again doesn't send
    the same email every time it checks — only a genuinely new pending
    candidate (a different model) notifies again."""
    state = load(use_case, root)
    state["notifiedModel"] = model
    _save(state, root)


def reject(use_case: str, *, root: Optional[Path] = None) -> dict:
    """Dismisses the pending candidate WITHOUT approving it — the only
    other function, besides `approve`, allowed to clear `pending`, and it
    never touches `approvedModel`. Recorded, not silently forgotten:
    appended to `rejected` the same way `history` never discards a past
    run. Clears `notifiedModel` too, so if this exact model resurfaces as
    pending again later, it notifies once more rather than staying quiet
    forever because of a dismissal from months ago."""
    state = load(use_case, root)
    pending = state.get("pending")
    if not pending:
        raise ValueError(f"nothing pending for {use_case!r} to reject.")
    state["rejected"] = (state.get("rejected") or [])[-49:] + [
        {"model": pending["model"], "rejectedAt": datetime.now(timezone.utc).isoformat()}]
    state["pending"] = None
    state["notifiedModel"] = None
    _save(state, root)
    return load(use_case, root)


def all_pending(*, projects_root=None, unscoped_root: Optional[Path] = None) -> list:
    """Every pending candidate across every project, plus unscoped use
    cases — the one place both the CLI's `pending` command and the
    dashboard's `/pending` page read from, so they can never disagree about
    what's waiting for review."""
    from . import project as project_module

    def _pending_in(state_dir: Path, project_slug: Optional[str]) -> list:
        if not state_dir.exists():
            return []
        found = []
        for p in sorted(state_dir.glob("*.json")):
            try:
                st = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if st.get("pending"):
                found.append({"project": project_slug, "useCase": st["useCase"],
                             "pending": st["pending"]})
        return found

    results = _pending_in(Path(unscoped_root or DEFAULT_DIR), None)
    for proj in project_module.list_all(projects_root):
        results += _pending_in(project_module.state_dir(proj.slug, projects_root), proj.slug)
    return results
