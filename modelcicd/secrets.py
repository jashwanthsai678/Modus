"""The one place this project writes an API key — into `.env`, the same file
`OPENROUTER_API_KEY` already lives in. No new secret-storage mechanism: the
interface (CLI and dashboard) can now ASK for a key instead of requiring you
to hand-edit this file, but where it ends up is exactly where it always did.

NEVER RETURNS A VALUE. `status()` reports whether a key is set, never what
it is — nothing here is a place a secret gets displayed back.
"""
import os
from pathlib import Path
from typing import Optional

from . import auth as auth_module
from . import catalogue as catalogue_module

DEFAULT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _key_env(provider: str) -> str:
    spec = catalogue_module.PROVIDERS.get(provider)
    if not spec:
        raise ValueError(f"unknown provider {provider!r}.")
    return spec["key_env"]


def _write_env_var(name: str, value: str, *, env_path: Optional[Path] = None) -> None:
    """THE ACTUAL one place this project writes to `.env` — `set_key` and
    `set_dashboard_key` are both thin callers of this, so there is exactly
    one implementation of "replace this line, or append it, and leave every
    other line untouched" rather than two copies that could drift. Also
    updates `os.environ` immediately, so a dashboard process already
    running picks up the change on its very next request without needing a
    restart — true for a provider key, and just as true for the dashboard's
    own auth key."""
    path = Path(env_path or DEFAULT_ENV_PATH)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    prefix = f"{name}="
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            lines[i] = f"{name}={value}"
            break
    else:
        lines.append(f"{name}={value}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ[name] = value


def set_key(provider: str, value: str, *, env_path: Optional[Path] = None) -> None:
    """Writes `value` for `provider`'s key into `.env`."""
    _write_env_var(_key_env(provider), value, env_path=env_path)


def set_dashboard_key(value: str, *, env_path: Optional[Path] = None) -> None:
    """Writes `MODELCICD_API_KEY` into `.env` — the one shared secret that
    gates the dashboard once it's reachable by more than just you (see
    `auth.py`). Setting this is what turns auth ON; there is no separate
    switch."""
    if not value.strip():
        raise ValueError("a dashboard key can't be empty — that would set MODELCICD_API_KEY "
                         "to a blank string, which auth.configured() treats as unset, silently "
                         "leaving the dashboard open.")
    _write_env_var(auth_module.API_KEY_ENV, value.strip(), env_path=env_path)


def status() -> dict:
    """{provider: bool} for every registered provider — whether a key is
    set, never the key itself. Checks the live environment (which `.env` was
    already loaded into at process start, and which `set_key` updates
    immediately) rather than re-reading any particular file, so this always
    reflects what a real call would actually use right now."""
    return {p: bool((os.environ.get(spec["key_env"]) or "").strip())
           for p, spec in catalogue_module.PROVIDERS.items()}
