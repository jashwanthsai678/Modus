"""The one function someone's application actually calls.

    from modelcicd.resolver import resolve
    model = resolve("prep_material_writer")
    # call your own LLM client with `model`, exactly as before

THIS IS THE ENTIRE INTEGRATION SURFACE. Everything else in this project —
polling, sandboxing, judging, ranking, emailing — exists to answer one question
well: which id should this line return? An application drops this import in
once, in place of a hardcoded model string, and from then on the model it uses
is controlled by whatever was last APPROVED — never by whatever a bench run
merely found. Nothing upstream of `state.approve()` can change this function's
answer.

FAILS SAFE, NOT SILENT. A use case with nothing approved yet returns `fallback`
if one was given, and raises if not — an application that hardcodes no fallback
and gets no answer needs to know immediately, not fail a support ticket three
requests later.

NO NETWORK CALL, NO LATENCY ADDED. This reads a small JSON file already on
disk. An application calling it on every request pays the cost of one file
read, not a request to some other service — self-hosted means there IS no other
service between the app and its model.
"""
from pathlib import Path
from typing import Optional

from . import state as state_module


def resolve(use_case: str, *, fallback: Optional[str] = None,
           root: Optional[Path] = None) -> str:
    """The currently-approved model id for one use case."""
    current = state_module.load(use_case, root).get("approvedModel")
    if current:
        return current
    if fallback:
        return fallback
    raise LookupError(
        f"no model is approved for use case {use_case!r} yet, and no fallback "
        f"was given. Run `python -m modelcicd.cli run --use-case {use_case}` "
        f"and approve a candidate, or pass resolve(..., fallback='your-current-model').")


def status(use_case: str, *, root: Optional[Path] = None) -> dict:
    """What is approved, and what is waiting for a human — for a health check
    or a status page, without pulling in the rest of the project."""
    state = state_module.load(use_case, root)
    return {"useCase": use_case, "approvedModel": state.get("approvedModel"),
            "approvedAt": state.get("approvedAt"), "pending": state.get("pending")}
