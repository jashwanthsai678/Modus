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

CONCURRENT WRITERS TO THE SAME USE CASE ARE REAL, NOT HYPOTHETICAL. A
scheduled run and a manual `approve` can land within the same second; two CI
jobs could be pointed at the same use case by mistake. Every function that
changes a use case's state acquires `_FileLock` first — a sidecar `.lock`
file, scoped to that one use case, created with a mode that fails if it
already exists, so two writers can never both believe they hold it. Without
this, two callers reading the same state, both computing a change from it,
and whichever saves last silently winning is a real, silent lost update —
this is scoped per use case (each has its own file and its own lock), so
concurrent work on DIFFERENT use cases never contends at all. `_save` writes
to a temp file and atomically renames it into place, so a crash mid-write
can never leave a half-written, corrupted state file behind either.
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "state"

_LOCK_TIMEOUT_S = 10.0
_LOCK_POLL_S = 0.05


class _FileLock:
    """An exclusive lock for one use case's state file, dependency-free —
    matches this project's habit of staying light on third-party packages.
    Retries briefly on contention rather than failing immediately, since a
    lock held for a few milliseconds by another writer is the expected
    case, not an error."""

    def __init__(self, state_path: Path):
        self._path = state_path.with_suffix(state_path.suffix + ".lock")

    def __enter__(self) -> "_FileLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + _LOCK_TIMEOUT_S
        while True:
            try:
                fd = os.open(str(self._path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"could not acquire the state lock at {self._path} within "
                        f"{_LOCK_TIMEOUT_S}s — another process may be stuck holding "
                        f"it; delete that file by hand if you're sure nothing else "
                        f"is using it.") from None
                time.sleep(_LOCK_POLL_S)

    def __exit__(self, *exc_info) -> bool:
        try:
            self._path.unlink()
        except OSError:
            pass
        return False


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
    """Writes to a temp file, then atomically renames it into place — a
    reader (or a crash) can never observe a half-written state file.
    Callers that mutate state must already hold that use case's `_FileLock`;
    this function alone does not make a read-modify-write sequence safe."""
    p = _path(state["useCase"], root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f"{p.suffix}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)
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

    with _FileLock(_path(use_case, root)):
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
    with _FileLock(_path(use_case, root)):
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
        _save(state, root)
    return load(use_case, root)


def set_min_improvement(use_case: str, value: float, *, root: Optional[Path] = None) -> None:
    with _FileLock(_path(use_case, root)):
        state = load(use_case, root)
        state["minImprovement"] = value
        _save(state, root)


def mark_notified(use_case: str, model: str, *, root: Optional[Path] = None) -> None:
    """Records that this candidate has already been emailed about, so a
    scheduled run that finds the SAME pending candidate again doesn't send
    the same email every time it checks — only a genuinely new pending
    candidate (a different model) notifies again."""
    with _FileLock(_path(use_case, root)):
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
    with _FileLock(_path(use_case, root)):
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
