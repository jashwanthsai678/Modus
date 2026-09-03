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
                "history": [], "pending": None}
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
