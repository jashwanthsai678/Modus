"""The one place this project ever writes into a user's OWN application code.

A SEPARATE, DELIBERATELY LATER STEP FROM `state.approve()`. Approving a model
is a state-file write inside this repo — reversible, cheap, and already the
one function everything else calls to promote a candidate. Patching a
`codeTarget` is a write into someone else's real source, which is a bigger
and harder-to-reverse thing to do unattended: every caller of `apply()` (see
`cli.py`'s `apply-code-patch` and `dashboard.py`'s code-patch routes) shows a
`preview()` first and asks for its own explicit confirmation. Nothing here is
ever triggered automatically by `state.approve()`.

THE NARROWEST EDIT THAT COULD WORK. `apply()` is one exact literal
`str.replace` — never a regex, never AST-aware. A model string either
appears verbatim or it doesn't; guessing harder about where "the model" is
written in someone else's code is exactly the kind of cleverness that reads
right in a preview and touches the wrong thing.
"""
from pathlib import Path
from typing import Optional

import yaml


def _resolve(repo_path, relative_file: str) -> Path:
    """Containment-checked join — the same guard `project.read_repo_file`
    uses, so a patch can never land outside the connected repo."""
    repo_root = Path(repo_path).resolve()
    target = (repo_root / relative_file).resolve()
    try:
        target.relative_to(repo_root)
    except ValueError:
        raise ValueError(f"{relative_file!r} is outside the connected repo.") from None
    return target


def preview(repo_path, relative_file: str, old_model: str, new_model: str) -> dict:
    """How many times `old_model` appears in the target file, verbatim.
    Raises rather than guessing if the file is missing or the string isn't
    there anymore — the file may have changed since `currentModel` was last
    recorded, and re-seeding it is a decision for the person, not this
    function."""
    target = _resolve(repo_path, relative_file)
    if not target.is_file():
        raise FileNotFoundError(f"no file at {relative_file!r} in the connected repo.")
    text = target.read_text(encoding="utf-8")
    count = text.count(old_model)
    if count == 0:
        raise ValueError(
            f"{old_model!r} was not found in {relative_file} — it may have "
            f"changed since this was last recorded. Update it by hand, or "
            f"update this feature's codeTarget.currentModel and try again.")
    return {"file": str(target), "occurrences": count, "old": old_model, "new": new_model}


def apply(repo_path, relative_file: str, old_model: str, new_model: str) -> int:
    """Replaces every exact occurrence of `old_model` with `new_model` in the
    target file. Returns how many were replaced. Call `preview()` first —
    this does not re-check that the string is actually present."""
    target = _resolve(repo_path, relative_file)
    text = target.read_text(encoding="utf-8")
    count = text.count(old_model)
    target.write_text(text.replace(old_model, new_model), encoding="utf-8")
    return count


def update_tracked_model(use_case_yaml_path, new_model: str) -> None:
    """Rewrites just `codeTarget.currentModel` in a saved use_case.yaml after
    a successful `apply()`, so the file stays an honest record of what's
    currently written into the connected code — nothing else in the
    document is touched."""
    p = Path(use_case_yaml_path)
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    doc.setdefault("codeTarget", {})["currentModel"] = new_model
    p.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
